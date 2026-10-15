# SPDX-License-Identifier: GPL-3.0-or-later
"""Results: the spoken audio with its timings, WAV/PCM export and SRT/VTT subtitles."""
from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np


def pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).round().astype("<i2").tobytes()


def wav_bytes(samples: np.ndarray, rate: int, bit_depth: int = 16) -> bytes:
    if bit_depth not in (16, 24):
        raise ValueError("bit_depth must be 16 or 24")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(bit_depth // 8)
        w.setframerate(rate)
        if bit_depth == 16:
            w.writeframes(pcm16(samples))
        else:
            ints = (np.clip(samples, -1.0, 1.0) * 8388607).round().astype("<i4")
            w.writeframes(ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())
    return buf.getvalue()


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Band-limited resampling with an FFT (fine for speech clips)."""
    if rate == target or not samples.size:
        return samples
    n = int(round(samples.size * target / rate))
    spectrum = np.fft.rfft(samples.astype(np.float64))
    keep = min(spectrum.size, n // 2 + 1)
    out = np.fft.irfft(spectrum[:keep], n) * (n / samples.size)
    return out.astype(np.float32)


# ---------------------------------------------------------------------- subtitles (same rules as loudness.ai)
def _clock(ms: float, comma: bool) -> str:
    ms = max(0, int(round(ms)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{',' if comma else '.'}{ms % 1000:03d}"


def _two_lines(line: str) -> str:
    if len(line) <= 42:
        return line
    middle, best = len(line) / 2, -1
    for i, ch in enumerate(line):
        if ch == " " and (best < 0 or abs(i - middle) < abs(best - middle)):
            best = i
    return line[:best] + "\n" + line[best + 1:] if best > 0 else line


def subtitle_cues(timings: dict) -> list:
    """One cue per sentence or line; longer ones split at word gaps to fit two lines of about 42 characters; a cue
    ending mid-thought (comma, semicolon, colon, ellipsis) joins the next when both fit."""
    text, cues = timings["text"], []

    def add(start, end, t0, t1):
        line = re.sub(r"\s+", " ", text[start:end]).strip()
        if line:
            cues.append({"text": _two_lines(line), "from": t0, "to": t1})

    for seg in timings["segments"]:
        words = [w for w in timings["words"] if w[0] >= seg[0] and w[1] <= seg[1]]
        if not words:
            add(seg[0], seg[1], seg[2], seg[3])
            continue
        first = 0
        for i in range(1, len(words)):
            if len(re.sub(r"\s+", " ", text[words[first][0]:words[i][1]])) > 84:
                add(words[first][0], words[i - 1][1], words[first][2], words[i - 1][3])
                first = i
        add(words[first][0], seg[1], words[first][2], max(words[-1][3], seg[3]))
    n = len(cues) - 2
    while n >= 0:
        joined = cues[n]["text"].replace("\n", " ") + " " + cues[n + 1]["text"].replace("\n", " ")
        if re.search(r"(,|;|:|\.\.\.|…)$", cues[n]["text"]) and len(joined) <= 84:
            cues[n:n + 2] = [{"text": _two_lines(joined), "from": cues[n]["from"], "to": cues[n + 1]["to"]}]
        n -= 1
    for i, cue in enumerate(cues):
        cue["to"] = max(cue["to"], cue["from"] + 700)
        if i + 1 < len(cues):
            cue["to"] = min(cue["to"], cues[i + 1]["from"])
    return cues


def srt(timings: dict) -> str:
    return "\n\n".join(f"{i + 1}\n{_clock(c['from'], True)} --> {_clock(c['to'], True)}\n{c['text']}"
                       for i, c in enumerate(subtitle_cues(timings))) + "\n"


def vtt(timings: dict) -> str:
    return "WEBVTT\n\n" + "\n\n".join(f"{_clock(c['from'], False)} --> {_clock(c['to'], False)}\n{c['text']}"
                                      for c in subtitle_cues(timings)) + "\n"


@dataclass
class Speech:
    """Spoken audio (mono float32 samples in -1..1) with word timings and the voice used."""
    samples: np.ndarray
    sample_rate: int
    text: str
    voice: object
    timings: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    spoken: str = ""   # the text as the voice read it, after text preparation ([[...]] marks raw phonemes)

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate

    def wav(self, bit_depth: int = 16) -> bytes:
        return wav_bytes(self.samples, self.sample_rate, bit_depth)

    def pcm(self, sample_rate: Optional[int] = None) -> bytes:
        """Raw 16-bit little-endian mono PCM, optionally resampled."""
        return pcm16(resample(self.samples, self.sample_rate, sample_rate or self.sample_rate))

    def save(self, path: Union[str, Path], bit_depth: int = 16) -> Path:
        """Write a WAV file. For MP3/Opus/FLAC, convert the WAV with any encoder (see ``loudness-tts --format``)."""
        path = Path(path)
        path.write_bytes(self.wav(bit_depth))
        return path

    def srt(self) -> str:
        return srt(self.timings)

    def vtt(self) -> str:
        return vtt(self.timings)

    def save_srt(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(self.srt(), encoding="utf-8")
        return path

    def save_vtt(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(self.vtt(), encoding="utf-8")
        return path

    def to_dict(self) -> dict:
        """Everything except the samples, for logs and JSON output."""
        return {"text": self.text, "spoken": self.spoken, "sample_rate": self.sample_rate, "duration": round(self.duration, 3),
                "voice": self.voice.to_dict() if hasattr(self.voice, "to_dict") else str(self.voice),
                "metrics": self.metrics, "timings": self.timings}


@dataclass
class SpeechChunk:
    """One piece of streamed speech: a spoken sentence with the pause after it."""
    samples: np.ndarray
    sample_rate: int
    index: int
    start: int   # span of the source text it speaks
    end: int
    final: bool = False

    def pcm(self) -> bytes:
        return pcm16(self.samples)
