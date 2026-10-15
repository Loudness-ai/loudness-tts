# SPDX-License-Identifier: GPL-3.0-or-later
"""A local speech server compatible with OpenAI's speech API, so existing clients can switch to local voices by
changing their base URL. Standard library only.

    local-tts serve --port 8880
    curl http://127.0.0.1:8880/v1/audio/speech -H "Content-Type: application/json" \
         -d '{"model": "tts-1", "input": "Hello there!", "voice": "nova", "response_format": "wav"}' -o hello.wav

Endpoints:
  POST /v1/audio/speech   OpenAI-compatible: {input, voice?, model?, response_format?, speed?}. Extra fields: language,
                          pause, expressiveness, rhythm. response_format "pcm" streams 24 kHz 16-bit PCM as it is made.
  POST /v1/tts            JSON answer with base64 WAV, word timings, SRT and VTT, voice and credit line.
  GET  /v1/audio/voices   voices (?language=es&limit=20); GET /v1/models; GET /health
Text is never logged or stored.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ._version import __version__
from .catalog import find_voice, voices
from .engine import TTS
from .errors import LocalTTSError
from .formats import CONTENT_TYPES, PCM_RATE, encode, ffmpeg_path
from .speech import pcm16, resample

MAX_BODY = 1 << 20


def _ascii(text: str) -> str:
    """HTTP headers carry ASCII: accents are dropped and the middle dot becomes a hyphen."""
    import unicodedata
    text = text.replace("\u00b7", "-")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
STATUS = {"VOICE_NOT_FOUND": 400, "TEXT_INVALID": 400, "TEXT_TOO_LONG": 413, "NOTHING_TO_SPEAK": 422,
          "OPTION_INVALID": 400, "FORMAT_UNAVAILABLE": 400, "OFFLINE": 503, "DOWNLOAD_FAILED": 502,
          "CHECKSUM_MISMATCH": 502, "TIMEOUT": 504, "CANCELLED": 499}


class SpeechServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, tts: TTS, *, workers: int = 1, api_key: str = "", cors: str = "",
                 queue_seconds: float = 120.0, timeout: float = 300.0):
        super().__init__(address, Handler)
        self.tts = tts
        self.slots = threading.BoundedSemaphore(max(1, workers))
        self.api_key = api_key
        self.cors = cors
        self.queue_seconds = queue_seconds
        self.timeout = timeout


class Handler(BaseHTTPRequestHandler):
    server: SpeechServer
    server_version = f"local-tts/{__version__}"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # no request contents in logs
        pass

    # -------------------------------------------------------------- helpers
    def _headers(self, status, content_type, length=None, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if self.server.cors:
            self.send_header("Access-Control-Allow-Origin", self.server.cors)
            self.send_header("Access-Control-Expose-Headers", "X-Voice-Id, X-Voice-Credit")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self._headers(status, "application/json", len(body))
        self.wfile.write(body)

    def _error(self, status, code, message):
        self._json(status, {"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error",
                                      "code": code}})

    def _authorized(self):
        if not self.server.api_key:
            return True
        got = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        return hmac.compare_digest(got.encode(), self.server.api_key.encode())

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise LocalTTSError("TEXT_INVALID", "Send a JSON body of up to 1 MB.")
        try:
            body = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            raise LocalTTSError("TEXT_INVALID", "Send valid JSON.") from None
        if not isinstance(body, dict):
            raise LocalTTSError("TEXT_INVALID", "Send a JSON object.")
        return body

    @staticmethod
    def _options(body):
        speed = body.get("speed")
        if speed is not None:
            try:
                speed = min(2.0, max(0.5, float(speed)))  # OpenAI allows 0.25-4; Piper sounds right within 0.5-2
            except (TypeError, ValueError):
                raise LocalTTSError("OPTION_INVALID", "speed must be a number.") from None
        options = {k: body[k] for k in ("language", "pause", "expressiveness", "rhythm") if body.get(k) is not None}
        return {"voice": body.get("voice") or None, "speed": speed if speed is not None else 1.0, **options}

    # -------------------------------------------------------------- routes
    def do_OPTIONS(self):
        self._headers(204, "text/plain", 0, {"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                                             "Access-Control-Allow-Headers": "Authorization, Content-Type"})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            self._json(200, {"status": "ok", "version": __version__, "ffmpeg": bool(ffmpeg_path())})
            return
        if not self._authorized():
            self._error(401, "UNAUTHORIZED", "Missing or wrong API key.")
            return
        if url.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local-tts"}
                                                         for m in ("local-tts", "tts-1", "tts-1-hd", "gpt-4o-mini-tts")]})
        elif url.path in ("/v1/audio/voices", "/v1/voices"):
            q = parse_qs(url.query)
            found = voices(q.get("language", [None])[0], sex=q.get("sex", [None])[0])
            limit = int(q.get("limit", ["100"])[0])
            self._json(200, {"object": "list", "data": [v.to_dict() for v in found[:limit]], "total": len(found)})
        else:
            self._error(404, "NOT_FOUND", "Not found.")

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/v1/audio/speech", "/v1/tts"):
            self._error(404, "NOT_FOUND", "Not found.")
            return
        if not self._authorized():
            self._error(401, "UNAUTHORIZED", "Missing or wrong API key.")
            return
        acquired = False
        try:
            body = self._body()
            text = body.get("input", body.get("text"))
            options = self._options(body)
            voice = find_voice(options["voice"] or "auto", options.get("language"))
            options["voice"] = voice.id
            acquired = self.server.slots.acquire(timeout=self.server.queue_seconds)
            if not acquired:
                self._error(429, "BUSY", "The speech server is busy. Try again shortly.")
                return
            extra = {"X-Voice-Id": voice.id}
            if voice.credit:
                extra["X-Voice-Credit"] = _ascii(voice.credit)
            if path == "/v1/tts":
                speech = self.server.tts.synthesize(text, timeout=self.server.timeout, **options)
                self._json(200, {"audio": base64.b64encode(speech.wav()).decode(), "format": "wav",
                                 "sample_rate": speech.sample_rate, "duration": round(speech.duration, 3),
                                 "voice": voice.to_dict(), "timings": speech.timings, "srt": speech.srt(),
                                 "vtt": speech.vtt(), "metrics": speech.metrics})
                return
            fmt = body.get("response_format") or ("mp3" if ffmpeg_path() else "wav")
            if fmt == "pcm":
                self._stream_pcm(text, options, extra)
                return
            speech = self.server.tts.synthesize(text, timeout=self.server.timeout, **options)
            data = encode(speech.samples, speech.sample_rate, fmt)
            self._headers(200, CONTENT_TYPES[fmt], len(data), extra)
            self.wfile.write(data)
        except LocalTTSError as error:
            self._error(STATUS.get(error.code, 500), error.code, error.message)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:  # noqa: BLE001 - never leak internals to clients
            self._error(500, "ENGINE_FAILED", "Speech generation failed.")
        finally:
            if acquired:
                self.server.slots.release()

    def _stream_pcm(self, text, options, extra):
        """24 kHz 16-bit PCM sent sentence by sentence (chunked), so playback starts before the text is finished."""
        chunks = self.server.tts.stream(text, timeout=self.server.timeout, **options)
        first = next(chunks)  # errors surface before any audio is sent
        self.send_response(200)
        self.send_header("Content-Type", "audio/pcm")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        if self.server.cors:
            self.send_header("Access-Control-Allow-Origin", self.server.cors)
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        for chunk in (first, *chunks):
            data = pcm16(resample(chunk.samples, chunk.sample_rate, PCM_RATE))
            if data:
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")


def serve(host: str = "127.0.0.1", port: int = 8880, *, tts: TTS = None, workers: int = 1, api_key: str = "",
          cors: str = "", preload: bool = True, quiet: bool = False) -> None:
    tts = tts or TTS()
    api_key = api_key or os.environ.get("LOCAL_TTS_API_KEY", "")
    if preload:
        tts.preload()
    server = SpeechServer((host, port), tts, workers=workers, api_key=api_key, cors=cors)
    if not quiet:
        print(f"local-tts {__version__} serving on http://{host}:{port}  (OpenAI-compatible: /v1/audio/speech)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
