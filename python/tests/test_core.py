"""Unit tests that need no voice model: catalogue, loudness, subtitles, WAV, downloads, CLI."""
import hashlib
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import local_tts
from local_tts import downloader as download, loudness
from local_tts.catalog import AUTO_USE, find_voice, openai_voice_map, voices
from local_tts.errors import LocalTTSError
from local_tts.speech import resample, srt, subtitle_cues, vtt, wav_bytes


# ---------------------------------------------------------------- catalogue
def test_catalogue_size_and_fields():
    every = voices()
    assert len(every) == 1683
    assert all(v.use in ("none", "credit", "copyleft", "unclear") for v in every)
    assert {v.locale for v in every} >= {"en-US", "en-GB", "es-ES", "es-MX", "de-DE", "fr-FR"}


def test_auto_voice_has_clear_licence_and_is_best():
    for lang in ("en-US", "es", "de-DE", "fr-FR"):
        v = find_voice("auto", lang)
        assert v.use in AUTO_USE
        assert v.overall == max(x.overall for x in voices(lang, use=AUTO_USE))


def test_find_voice_by_id_model_and_openai_name():
    best = voices("en-US")[0]
    assert find_voice(best.id).id == best.id
    assert find_voice("en_US-libritts-high").model == "en_US-libritts-high"
    assert find_voice("nova").sex == "F" and find_voice("onyx").sex == "M"
    assert set(openai_voice_map()) >= {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    with pytest.raises(LocalTTSError) as error:
        find_voice("no-such-voice")
    assert error.value.code == "VOICE_NOT_FOUND"


def test_filters_and_credit():
    f = voices("en-US", sex="female", min_score=95)
    assert f and all(v.sex == "F" and v.overall >= 95 for v in f)
    assert all(v.use == "none" for v in voices(use="none"))
    jenny = find_voice("en_IE-jenny-medium")
    assert "Jenny" in jenny.credit and jenny.note
    assert find_voice(voices(use="none")[0].id).credit is None


# ---------------------------------------------------------------- loudness
def sine(freq=997.0, amp=1.0, seconds=5.0, rate=48000):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("rate", [16000, 22050, 48000])
def test_reference_tone(rate):
    # BS.1770: a full-scale 997 Hz sine in one channel reads -3.01 LUFS.
    assert loudness.integrated_loudness(sine(rate=rate), rate) == pytest.approx(-3.01, abs=0.05)


def test_normalize_reaches_target_and_limits_peaks():
    rate = 22050
    x = sine(rate=rate) * 0.05
    x[5000:5100] = 0.9  # a peak that linear gain alone would push over the ceiling
    y, report = loudness.normalize(x, rate, -14.0, -1.5)
    m = loudness.measure(y, rate)
    assert m["integrated_lufs"] == pytest.approx(-14.0, abs=0.15)
    assert m["true_peak_dbtp"] <= -1.4
    assert report["normalized"] and report["limited"]


def test_silence_is_left_alone():
    y, report = loudness.normalize(np.zeros(22050, np.float32), 22050)
    assert not report["normalized"] and not y.any()


# ---------------------------------------------------------------- audio and subtitles
def test_wav_16_and_24_bit():
    x = sine(seconds=0.1, rate=22050) * 0.5
    for depth in (16, 24):
        with wave.open(io.BytesIO(wav_bytes(x, 22050, depth))) as w:
            assert (w.getsampwidth(), w.getframerate(), w.getnframes(), w.getnchannels()) == (depth // 8, 22050, x.size, 1)


def test_resample_length():
    assert resample(np.zeros(22050, np.float32), 22050, 24000).size == 24000


TIMINGS = {"version": 1, "text": "Hello there, my friend. This is a second sentence that is quite a bit longer than the first one, so it splits.",
           "duration": 9000,
           "segments": [[0, 23, 0, 1500], [24, 111, 1800, 8500]], "words": []}


def _words(timings):
    import re
    words, t = [], 0
    for seg in timings["segments"]:
        found = list(re.finditer(r"\S+", timings["text"][seg[0]:seg[1]]))
        step = (seg[3] - seg[2]) / len(found)
        for i, m in enumerate(found):
            words.append([seg[0] + m.start(), seg[0] + m.end(), seg[2] + i * step, seg[2] + (i + 1) * step])
    return words


def test_subtitle_rules():
    timings = dict(TIMINGS, words=_words(TIMINGS))
    cues = subtitle_cues(timings)
    assert cues[0]["text"] == "Hello there, my friend."
    assert all(len(line) <= 60 for c in cues for line in c["text"].split("\n"))
    assert all(c["to"] - c["from"] >= 0 for c in cues)
    assert srt(timings).startswith("1\n00:00:00,000 --> ")
    assert vtt(timings).startswith("WEBVTT\n\n00:00:00.000 --> ")


# ---------------------------------------------------------------- downloads
def test_download_verifies_checksum(tmp_path, monkeypatch):
    payload = b"model bytes" * 100
    src = tmp_path / "src.onnx"
    src.write_bytes(payload)
    good = {"url": src.as_uri(), "size": len(payload), "md5": hashlib.md5(payload).hexdigest()}
    target = tmp_path / "cache" / "m.onnx"
    download._fetch(good["url"], target, good, "m.onnx", None, 10)
    assert target.read_bytes() == payload and download._verified(target, good)
    bad = dict(good, md5="0" * 32)
    with pytest.raises(LocalTTSError) as error:
        download._fetch(bad["url"], tmp_path / "cache" / "n.onnx", bad, "n.onnx", None, 10)
    assert error.value.code == "CHECKSUM_MISMATCH"
    assert not (tmp_path / "cache" / "n.onnx").exists()


def test_offline_refuses_missing_voice(tmp_path):
    with pytest.raises(LocalTTSError) as error:
        download.ensure_model("en_US-libritts-high", root=tmp_path, offline=True)
    assert error.value.code == "OFFLINE"


# ---------------------------------------------------------------- CLI
def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "local_tts", *args], capture_output=True, text=True, timeout=120)


def test_cli_voices_json():
    r = run_cli("voices", "--language", "it", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data[0]["locale"] == "it-IT" and "credit" in data[0]


def test_cli_error_is_reported():
    r = run_cli("say", "hello", "--voice", "nope", "-o", "x.wav", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["error"]["code"] == "VOICE_NOT_FOUND"


def test_version():
    assert local_tts.__version__ and run_cli("--version").stdout.startswith("local-tts ")
