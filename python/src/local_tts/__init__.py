# SPDX-License-Identifier: GPL-3.0-or-later
"""local-tts: fast, natural text to speech on your own computer or server, by Loudness.ai.

    import local_tts
    local_tts.speak("Dr. Smith paid $5.50 on 4/5/2026.", "speech.wav")      # best English (US) voice
    speech = local_tts.synthesize("Hola, ¿qué tal?", language="es-ES")
    speech.save("hola.wav"); speech.save_srt("hola.srt")
    for chunk in local_tts.stream("A long text, read sentence by sentence."):
        play(chunk.pcm())
    local_tts.voices(language="de", sex="female")[:3]

Voices are ranked by Loudness.ai's open voice study (https://loudness.ai/research/piper-tts-voices-ranked/) and
downloaded once, with checksums, from their original source. Docs: https://github.com/Loudness-ai/local-tts
"""
from __future__ import annotations

import threading
from typing import Optional

from ._version import __version__
from .catalog import Voice, find_voice, languages, voices
from .downloader import cache_dir, downloaded_models, ensure_model, remove_model
from .engine import TTS
from .errors import CancelledError, LocalTTSError
from .speech import Speech, SpeechChunk

__all__ = ["TTS", "Speech", "SpeechChunk", "Voice", "LocalTTSError", "CancelledError", "speak", "synthesize", "stream",
           "voices", "languages", "find_voice", "download", "downloaded_models", "remove_model", "cache_dir", "__version__"]

_default: Optional[TTS] = None
_default_lock = threading.Lock()


def _engine() -> TTS:
    global _default
    with _default_lock:
        if _default is None:
            _default = TTS()
        return _default


def synthesize(text: str, **options) -> Speech:
    """Speak ``text`` with a shared engine. Options: voice, language, speed, pause, expressiveness, rhythm,
    prepare_text, normalize, target_lufs, true_peak_db, cancel, timeout (see :meth:`TTS.synthesize`)."""
    return _engine().synthesize(text, **options)


def speak(text: str, path, *, srt: Optional[str] = None, vtt: Optional[str] = None, **options) -> Speech:
    """Speak ``text`` into a WAV file (and optional SRT/VTT subtitles). Returns the :class:`Speech`."""
    speech = synthesize(text, **options)
    speech.save(path)
    if srt:
        speech.save_srt(srt)
    if vtt:
        speech.save_vtt(vtt)
    return speech


def stream(text: str, **options):
    """Speak ``text`` piece by piece for low latency (see :meth:`TTS.stream`)."""
    return _engine().stream(text, **options)


def download(voice: str = "auto", language: Optional[str] = None, progress=None) -> Voice:
    """Download and verify a voice now (useful in Docker builds and offline setups)."""
    info = find_voice(voice, language)
    ensure_model(info.model, progress=progress)
    return info
