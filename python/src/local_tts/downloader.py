# SPDX-License-Identifier: GPL-3.0-or-later
"""Voice files: downloaded once from their original source, checked against the catalogue's size and checksum,
then used offline. Nothing is sent anywhere except the file requests themselves.

Cache folder: $LOCAL_TTS_HOME, else $XDG_CACHE_HOME/local-tts, else ~/.cache/local-tts (Windows: %LOCALAPPDATA%).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ._version import __version__
from .catalog import model_info
from .errors import LocalTTSError

CHUNK = 1 << 20


def cache_dir(override=None) -> Path:
    if override:
        return Path(override).expanduser()
    if os.environ.get("LOCAL_TTS_HOME"):
        return Path(os.environ["LOCAL_TTS_HOME"]).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "local-tts"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "local-tts"


def model_paths(model: str, root=None) -> tuple:
    folder = cache_dir(root) / "voices" / model
    return folder / f"{model}.onnx", folder / f"{model}.onnx.json"


def _digest(spec: dict) -> tuple:
    if "sha256" in spec:
        return "sha256", spec["sha256"]
    return "md5", spec["md5"]


def _verified(path: Path, spec: dict) -> bool:
    """A file is trusted when its size matches and a receipt records the checksum it passed."""
    receipt = path.with_name(path.name + ".ok")
    algo, want = _digest(spec)
    return path.is_file() and path.stat().st_size == spec["size"] and receipt.is_file() and receipt.read_text() == f"{algo}:{want}"


def _fetch(url: str, target: Path, spec: dict, label: str, progress: Optional[Callable], timeout: float) -> None:
    algo, want = _digest(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(3):
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        out = os.fdopen(fd, "wb")
        try:
            h = hashlib.new(algo)
            done = 0
            request = urllib.request.Request(url, headers={"User-Agent": f"local-tts/{__version__}"})
            with urllib.request.urlopen(request, timeout=timeout) as response, out:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    h.update(block)
                    done += len(block)
                    if done > spec["size"]:
                        raise LocalTTSError("CHECKSUM_MISMATCH", f"{label} is larger than expected; download refused.")
                    if progress:
                        progress({"file": label, "bytes": done, "total": spec["size"]})
            if done != spec["size"] or h.hexdigest() != want:
                raise LocalTTSError("CHECKSUM_MISMATCH", f"{label} does not match its published checksum; it was not used.")
            os.replace(tmp, target)
            target.with_name(target.name + ".ok").write_text(f"{algo}:{want}")
            return
        except LocalTTSError:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
        finally:
            out.close()
            if os.path.exists(tmp):
                os.unlink(tmp)
    raise LocalTTSError("DOWNLOAD_FAILED", f"Could not download {label}: {last_error}")


def ensure_model(model: str, *, root=None, offline: bool = False, progress: Optional[Callable] = None,
                 timeout: float = 60.0) -> tuple:
    """Paths of a catalogue model and its config, downloading and verifying them first if needed."""
    info = model_info(model)
    onnx_path, config_path = model_paths(model, root)
    for path, key in ((config_path, "config"), (onnx_path, "model")):
        spec = info["files"][key]
        if _verified(path, spec):
            continue
        if offline or os.environ.get("LOCAL_TTS_OFFLINE") == "1":
            raise LocalTTSError("OFFLINE", f"{model} is not downloaded yet and downloads are turned off.")
        _fetch(spec["url"], path, spec, path.name, progress, timeout)
    return onnx_path, config_path


def downloaded_models(root=None) -> list:
    """Models in the cache that passed their checksum."""
    folder = cache_dir(root) / "voices"
    out = []
    if folder.is_dir():
        for sub in sorted(folder.iterdir()):
            try:
                info = model_info(sub.name)
            except LocalTTSError:
                continue
            onnx_path, config_path = model_paths(sub.name, root)
            if _verified(onnx_path, info["files"]["model"]) and _verified(config_path, info["files"]["config"]):
                out.append(sub.name)
    return out


def remove_model(model: str, root=None) -> bool:
    folder = cache_dir(root) / "voices" / model
    if not folder.is_dir():
        return False
    for p in folder.iterdir():
        p.unlink()
    folder.rmdir()
    return True
