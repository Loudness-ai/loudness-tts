# SPDX-License-Identifier: GPL-3.0-or-later
"""Errors raised by loudness-tts. Each carries a stable ``code`` for programs and a plain message for people."""
from __future__ import annotations


class TTSError(Exception):
    """Base error. ``code`` is one of: VOICE_NOT_FOUND, DOWNLOAD_FAILED, CHECKSUM_MISMATCH, OFFLINE, TEXT_INVALID,
    TEXT_TOO_LONG, NOTHING_TO_SPEAK, OPTION_INVALID, CANCELLED, TIMEOUT, ENGINE_FAILED, FORMAT_UNAVAILABLE."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


class CancelledError(TTSError):
    def __init__(self, message: str = "Speech generation was cancelled."):
        super().__init__("CANCELLED", message)
