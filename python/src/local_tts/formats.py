# SPDX-License-Identifier: GPL-3.0-or-later
"""Output formats. WAV and raw PCM are built in; MP3, Opus, AAC and FLAC use an FFmpeg found on the PATH."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

from .errors import LocalTTSError
from .speech import pcm16, resample, wav_bytes

CONTENT_TYPES = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac",
                 "flac": "audio/flac"}
_FFMPEG_ARGS = {"mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
                "opus": ["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"],
                "aac": ["-c:a", "aac", "-b:a", "96k", "-f", "adts"],
                "flac": ["-c:a", "flac", "-f", "flac"]}
PCM_RATE = 24000  # as OpenAI's speech API: 24 kHz, 16-bit, little-endian, mono


def ffmpeg_path():
    return shutil.which("ffmpeg")


def encode(samples: np.ndarray, rate: int, fmt: str) -> bytes:
    if fmt == "wav":
        return wav_bytes(samples, rate)
    if fmt == "pcm":
        return pcm16(resample(samples, rate, PCM_RATE))
    if fmt not in _FFMPEG_ARGS:
        raise LocalTTSError("FORMAT_UNAVAILABLE", f"Unknown format {fmt!r}: use wav, pcm, mp3, opus, aac or flac.")
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise LocalTTSError("FORMAT_UNAVAILABLE", f"{fmt} needs FFmpeg on the PATH; use wav or pcm, or install FFmpeg.")
    run = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1",
                          "-i", "pipe:0", *_FFMPEG_ARGS[fmt], "pipe:1"], input=pcm16(samples), capture_output=True,
                         timeout=300)
    if run.returncode != 0 or not run.stdout:
        raise LocalTTSError("FORMAT_UNAVAILABLE", f"FFmpeg could not encode {fmt}.")
    return run.stdout
