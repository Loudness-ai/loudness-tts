"""Synthesis tests with real voices. They run when the voices are in the cache (set LOUDNESS_TTS_HOME); in CI the
workflow downloads them first with `loudness-tts download`."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from loudness_tts import TTS, loudness
from loudness_tts.downloader import downloaded_models
from loudness_tts.errors import TTSError

EN = "en_US-clean100-medium"
pytestmark = pytest.mark.skipif(EN not in downloaded_models(), reason=f"{EN} is not in the cache")


@pytest.fixture(scope="module")
def tts():
    return TTS(offline=True)


def test_speech_is_normalized_and_timed(tts):
    s = tts.synthesize("Dr. Smith paid $5.50 on 4/5/2026. Call +1 555-123-4567, please.")
    m = loudness.measure(s.samples, s.sample_rate)
    assert m["integrated_lufs"] == pytest.approx(-16.0, abs=0.3)
    assert m["true_peak_dbtp"] <= -1.4
    assert s.duration > 3 and s.metrics["real_time_factor"] < 1
    words = [s.text[a:b] for a, b, _, _ in s.timings["words"]]
    assert "$5.50" in words and "4/5/2026." in words
    starts = [w[2] for w in s.timings["words"]]
    assert starts == sorted(starts)
    assert s.srt().startswith("1\n") and "WEBVTT" in s.vtt()
    import importlib.util
    if importlib.util.find_spec("onnx"):  # pip install "loudness-tts[timings]": timings come from the model itself
        assert s.metrics["timings"] == "exact"


def test_text_preparation_changes_what_is_said(tts):
    raw = tts.synthesize("It costs $5.50.", prepare_text=False)
    prepared = tts.synthesize("It costs $5.50.")
    assert raw.spoken == "It costs $5.50."
    assert prepared.spoken == "It costs 5 dollars and 50 cents."  # digits are read as numbers by the voice


def test_stream_matches_sentences(tts):
    chunks = list(tts.stream("First sentence. Second sentence! Third?"))
    assert len(chunks) >= 3 and chunks[-1].final
    assert all(c.samples.dtype == np.float32 for c in chunks)


def test_options_are_checked(tts):
    with pytest.raises(TTSError):
        tts.synthesize("Hello", speed=5)
    with pytest.raises(TTSError) as error:
        tts.synthesize("x" * 10, timeout=1e-9)
    assert error.value.code == "TIMEOUT"
    stop = threading.Event()
    stop.set()
    with pytest.raises(TTSError) as error:
        tts.synthesize("Hello there.", cancel=stop)
    assert error.value.code == "CANCELLED"


def test_server_openai_compatible(tts):
    from loudness_tts.server import SpeechServer

    server = SpeechServer(("127.0.0.1", 0), tts, api_key="k")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        body = json.dumps({"model": "tts-1", "input": "Hello there.", "voice": "nova", "response_format": "wav"}).encode()
        request = urllib.request.Request(url + "/v1/audio/speech", body, {"Content-Type": "application/json",
                                                                         "Authorization": "Bearer k"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            assert response.headers["Content-Type"] == "audio/wav" and data[:4] == b"RIFF"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url + "/v1/models"), timeout=10)
        assert error.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
