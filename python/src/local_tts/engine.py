# SPDX-License-Identifier: GPL-3.0-or-later
"""The speech engine: Piper voices with Loudness.ai's text preparation, pauses, calmer delivery, click-free joins,
word timings and loudness normalization. Runs locally; nothing leaves the computer except voice downloads.
"""
from __future__ import annotations

import collections
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import numpy as np

from . import loudness, textprep
from .catalog import Voice, find_voice
from .downloader import cache_dir, ensure_model
from .errors import CancelledError, LocalTTSError
from .speech import Speech, SpeechChunk

PAUSES = {"short": 0.12, "normal": 0.35, "long": 0.7}
# Pause after a segment, as a share of the chosen pause; a paragraph gets more.
PAUSE_SHARE = {"clause": 0.4, "ellipsis": 1.0, "sentence": 1.0, "line": 1.0}
# Calmer than each model's own settings: the random variation small VITS voices add to sound and timing is what makes
# them tremble. Models already at 0.5 or lower keep theirs.
CALM = 0.7
FADE_IN_SECONDS = 0.003   # speech meets silence without a click
FADE_OUT_SECONDS = 0.012
TAIL_SECONDS = 0.12       # players do not clip the last word
VOICED_DB = -45.0
_PHONEMIZE = threading.Lock()  # espeak-ng keeps global state


def _espeak_dir() -> Optional[Path]:
    """espeak-ng rejects very long data paths; use a short link to Piper's bundled data when the path is long."""
    from piper.phonemize_espeak import ESPEAK_DATA_DIR

    source = Path(ESPEAK_DATA_DIR)
    if len(str(source)) <= 100:
        return None
    link_dir = Path(tempfile.gettempdir()) / "local-tts-espeak"
    link = link_dir / "espeak-ng-data"
    try:
        link_dir.mkdir(exist_ok=True)
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        return link
    except OSError:
        return None


@dataclass
class _Loaded:
    voice: object                 # piper.PiperVoice
    aligned: bool                 # model reports per-sound durations (exact word timings)
    us_words: bool                # American word forms for US voices trained on British phonemes
    sample_rate: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def _bounded(value, default: float, low: float, high: float, label: str) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LocalTTSError("OPTION_INVALID", f"{label} is not a number.") from None
    if not low <= number <= high:
        raise LocalTTSError("OPTION_INVALID", f"{label} must be between {low} and {high}.")
    return number


def clean_text(value, max_characters: int) -> str:
    if not isinstance(value, str):
        raise LocalTTSError("TEXT_INVALID", "Text must be a string.")
    text = unicodedata.normalize("NFC", value)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise LocalTTSError("TEXT_INVALID", "There is no text to speak.")
    if len(text) > max_characters:
        raise LocalTTSError("TEXT_TOO_LONG", f"Text is longer than {max_characters} characters; raise max_characters or split it.")
    return text


def _faded(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = audio.copy()
    fade_in = min(audio.size // 4, int(rate * FADE_IN_SECONDS))
    fade_out = min(audio.size // 4, int(rate * FADE_OUT_SECONDS))
    if fade_in:
        audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, endpoint=False, dtype=np.float32)
    if fade_out:
        audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return audio


def _voiced(audio: np.ndarray) -> tuple:
    level = np.abs(audio)
    if not audio.size or level.max() <= 0:
        return 0, int(audio.size)
    loud = np.nonzero(level > level.max() * 10 ** (VOICED_DB / 20))[0]
    return (int(loud[0]), int(loud[-1]) + 1) if len(loud) else (0, int(audio.size))


def _pause_after(kind: str, pause: float) -> float:
    if kind == "paragraph":
        return max(2 * pause, pause + 0.3)
    return pause * PAUSE_SHARE.get(kind, 0.0)


class TTS:
    """A reusable speech engine. Keep one per process: voices stay loaded between calls.

        tts = TTS()                                   # best English (US) voice, downloaded on first use
        speech = tts.synthesize("Dr. Smith paid $5.50 on 4/5/2026.")
        speech.save("speech.wav"); speech.save_srt("speech.srt")
        TTS(language="es-ES").synthesize("Llegamos el 15/10/2026.").save("es.wav")

    voice: a voice id, a model name, "auto" (the best voice for ``language``) or a local path to a Piper ``.onnx``.
    threads: CPU threads per synthesis (default: up to 4). cache_size: voices kept loaded.
    timings: "auto" (exact word timings when the optional ``onnx`` package is installed, else estimated), True or False.
    """

    def __init__(self, voice: Optional[str] = "auto", *, language: Optional[str] = None, threads: Optional[int] = None,
                 cache_size: int = 2, cache_dir: Optional[Union[str, Path]] = None, offline: bool = False,
                 timings: Union[str, bool] = "auto", max_characters: int = 100_000,
                 progress: Optional[Callable] = None):
        self.default_voice = voice
        self.default_language = language
        self.threads = threads or max(1, min(4, os.cpu_count() or 1))
        self.cache_size = max(1, cache_size)
        self.root = cache_dir
        self.offline = offline
        self.timings = timings
        self.max_characters = max_characters
        self.progress = progress
        self._cache: "collections.OrderedDict[str, _Loaded]" = collections.OrderedDict()
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ voices
    def voice(self, voice: Optional[str] = None, language: Optional[str] = None) -> Voice:
        """The catalogue voice that ``voice``/``language`` (or the engine defaults) resolve to."""
        return find_voice(voice or self.default_voice, language or self.default_language)

    def preload(self, voice: Optional[str] = None, language: Optional[str] = None) -> Voice:
        """Download (if needed) and load a voice now, so the first request is fast."""
        info = self._resolve(voice, language)
        self._load(info)
        return info

    def _resolve(self, voice, language):
        value = voice or self.default_voice
        if isinstance(value, (str, Path)) and str(value).endswith(".onnx"):
            return self._custom_voice(Path(value), language or self.default_language or "en-US")
        return self.voice(voice, language)

    @staticmethod
    def _custom_voice(path: Path, locale: str) -> Voice:
        if not path.is_file():
            raise LocalTTSError("VOICE_NOT_FOUND", f"No model file at {path}.")
        cfg = json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))
        return Voice(id=str(path), model=str(path), speaker=None, name=path.stem, language=locale, locale=locale,
                     quality=cfg.get("audio", {}).get("quality") or "unknown", sex=None, overall=0, natural=0, clear=0,
                     rank=0, of=0, licence="see the model card", use="unclear", dataset="", source="local",
                     sample_rate=cfg.get("audio", {}).get("sample_rate", 22050))

    def _load(self, info: Voice) -> _Loaded:
        key = info.model
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        import onnxruntime
        from piper import PiperVoice
        from piper.config import PiperConfig

        if info.source == "local":
            onnx_path, config_path = Path(info.model), Path(f"{info.model}.json")
        else:
            onnx_path, config_path = ensure_model(info.model, root=self.root, offline=self.offline, progress=self.progress)
        model_file, aligned = self._aligned(onnx_path)
        onnxruntime.set_default_logger_severity(3)  # errors only: no hardware-discovery chatter on stderr
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle threads sleep
        options.enable_cpu_mem_arena = False   # a voice leaving the cache gives its memory back
        config = PiperConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        session = onnxruntime.InferenceSession(str(model_file), sess_options=options, providers=["CPUExecutionProvider"])
        kwargs = {"session": session, "config": config, "download_dir": cache_dir(self.root) / "resources"}
        espeak = _espeak_dir()
        if espeak:
            kwargs["espeak_data_dir"] = espeak
        loaded = _Loaded(voice=PiperVoice(**kwargs), aligned=aligned,
                         us_words=info.locale == "en-US" and getattr(config, "espeak_voice", "") == "en",
                         sample_rate=config.sample_rate)
        with self._cache_lock:
            self._cache[key] = loaded
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return loaded

    def _aligned(self, onnx_path: Path) -> tuple:
        """A copy of the model that reports per-sound durations (Piper's own patch), made once in the cache folder.
        Loading a patched file uses far less memory than patching in memory."""
        if self.timings is False:
            return onnx_path, False
        try:
            import onnx  # optional: pip install "local-tts[timings]"
            from piper.patch_voice_with_alignment import add_alignment_output
        except ImportError:
            if self.timings is True:
                raise LocalTTSError("OPTION_INVALID", 'Exact word timings need the onnx package: pip install "local-tts[timings]".')
            return onnx_path, False
        stat = onnx_path.stat()
        target = cache_dir(self.root) / "aligned" / f"{onnx_path.stem}-{stat.st_size}.onnx"
        if target.is_file():
            return target, True
        try:
            model = onnx.load(str(onnx_path))
            add_alignment_output(model)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            tmp.write_bytes(model.SerializeToString())
            del model
            os.replace(tmp, target)
            return target, True
        except ValueError:  # already patched, or no duration tensor
            return onnx_path, False

    def close(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # ------------------------------------------------------------------ synthesis
    def _options(self, speed, pause, expressiveness, rhythm):
        if isinstance(pause, str):
            if pause not in PAUSES:
                raise LocalTTSError("OPTION_INVALID", "pause must be short, normal, long or a number of seconds.")
            pause_s = PAUSES[pause]
        else:
            pause_s = _bounded(pause, 0.35, 0.0, 3.0, "pause")
        return (_bounded(speed, 1.0, 0.5, 2.0, "speed"), pause_s, _bounded(expressiveness, 1.0, 0.5, 1.5, "expressiveness"),
                _bounded(rhythm, 1.0, 0.5, 1.5, "rhythm"))

    def _sentences(self, loaded: _Loaded, text: str, syn_config):
        """Audio for each sentence of one prepared segment (Piper's own loop, with collapsed word gaps restored), with
        each sound's start and end sample when the model reports durations."""
        voice = loaded.voice
        id_map = voice.config.phoneme_id_map
        with _PHONEMIZE:
            sentences = voice.phonemize(text)
        for sentence in sentences:
            phonemes = [p for i, p in enumerate(sentence) if p != " " or (i and sentence[i - 1] != " ")]
            phonemes = [p for p in phonemes if p in id_map]
            while phonemes and phonemes[0] == " ":
                phonemes.pop(0)
            while phonemes and phonemes[-1] == " ":
                phonemes.pop()
            if not phonemes:
                continue
            ids = voice.phonemes_to_ids(phonemes)
            result = voice.phoneme_ids_to_audio(ids, syn_config, include_alignments=loaded.aligned)
            audio, per_id = result if isinstance(result, tuple) else (result, None)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if not audio.size:
                continue
            sounds = None
            if per_id is not None and len(per_id) == len(ids):
                pad = len(id_map["_"])
                position = int(per_id[:len(id_map["^"]) + pad].sum())
                index = len(id_map["^"]) + pad
                sounds = []
                for phoneme in phonemes:
                    count = len(id_map[phoneme]) + pad
                    length = int(per_id[index:index + count].sum())
                    sounds.append((phoneme, position, position + length))
                    position += length
                    index += count
            yield audio, sounds

    def _generate(self, text, info, speed, pause, expressiveness, rhythm, prepare_text, cancel, deadline):
        """Yields (kind, payload) in order: ("audio", samples) pieces and ("segment", timing record) entries."""
        from piper import SynthesisConfig

        loaded = self._load(info)
        config = loaded.voice.config
        syn = SynthesisConfig(
            speaker_id=info.speaker, length_scale=config.length_scale / speed,
            noise_scale=(config.noise_scale * CALM if config.noise_scale > 0.5 else config.noise_scale) * expressiveness,
            noise_w_scale=(config.noise_w_scale * CALM if config.noise_w_scale > 0.5 else config.noise_w_scale) * rhythm,
            normalize_audio=False,  # the whole result is normalized once, so sentences keep one level
        )
        rate = loaded.sample_rate
        if prepare_text:
            segments = textprep.prepare(text, info.locale, loaded.us_words)
        else:
            segments = [textprep.Segment(text=text, start=0, end=len(text), pause="", tokens=())]
        gap = np.zeros(int(rate * pause), dtype=np.float32)
        position = 0
        for segment in segments:
            if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
                raise CancelledError()
            if deadline is not None and time.monotonic() > deadline:
                raise LocalTTSError("TIMEOUT", "Speech generation took longer than the timeout.")
            chunks = list(self._sentences(loaded, segment.text, syn))
            heard, stream, aligned = [], [], True
            for index, (chunk, sounds) in enumerate(chunks):
                if index:
                    yield "audio", gap
                    position += gap.size
                    stream.append((" ", position - gap.size, position))
                chunk = _faded(chunk, rate)
                spoken = [(a, b) for p, a, b in sounds or [] if p.strip()]
                first, last = (spoken[0][0], spoken[-1][1]) if spoken else _voiced(chunk)
                if sounds:
                    stream += [(p, position + a, position + b) for p, a, b in sounds]
                else:
                    aligned = False
                heard.append((position + first, position + last))
                yield "audio", chunk
                position += chunk.size
            if heard:
                yield "segment", (segment, heard[0][0], heard[-1][1], stream if aligned else None, loaded)
            after = _pause_after(segment.pause, pause) if chunks else 0.0
            if after:
                silence = np.zeros(int(rate * after), dtype=np.float32)
                yield "audio", silence
                position += silence.size
        yield "audio", np.zeros(int(rate * TAIL_SECONDS), dtype=np.float32)

    def synthesize(self, text: str, *, voice: Optional[str] = None, language: Optional[str] = None, speed: float = 1.0,
                   pause: Union[str, float] = "normal", expressiveness: float = 1.0, rhythm: float = 1.0,
                   prepare_text: bool = True, normalize: bool = True, target_lufs: float = -16.0,
                   true_peak_db: float = -1.5, cancel=None, timeout: Optional[float] = None) -> Speech:
        """Speak ``text`` and return a :class:`Speech` (samples, timings, subtitles, save helpers)."""
        started = time.monotonic()
        text = clean_text(text, self.max_characters)
        speed, pause_s, expressiveness, rhythm = self._options(speed, pause, expressiveness, rhythm)
        target_lufs = _bounded(target_lufs, -16.0, -40.0, -5.0, "target_lufs")
        true_peak_db = _bounded(true_peak_db, -1.5, -9.0, 0.0, "true_peak_db")
        info = self._resolve(voice, language)
        deadline = started + timeout if timeout else None
        parts, timed, rate, loaded = [], [], None, None
        for kind, payload in self._generate(text, info, speed, pause_s, expressiveness, rhythm, prepare_text, cancel, deadline):
            if kind == "audio":
                parts.append(payload)
            else:
                timed.append(payload[:4])
                loaded = payload[4]
        if not timed:
            raise LocalTTSError("NOTHING_TO_SPEAK", "There is nothing to speak in this text.")
        rate = loaded.sample_rate
        audio = np.concatenate(parts)
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-6:
            raise LocalTTSError("NOTHING_TO_SPEAK", "There is nothing to speak in this text.")
        audio = np.clip(audio / peak * 0.9, -1.0, 1.0).astype(np.float32)
        report = {"normalized": False, "limited": False}
        if normalize:
            audio, report = loudness.normalize(audio, rate, target_lufs, true_peak_db)
        timings = _timings(text, timed, loaded, rate, audio.size)
        spoken = " ".join(seg.text for seg, *_ in timed)
        elapsed = time.monotonic() - started
        metrics = {"seconds": round(elapsed, 3), "audio_seconds": round(audio.size / rate, 3),
                   "real_time_factor": round(elapsed / max(audio.size / rate, 1e-9), 4),
                   "timings": "exact" if loaded.aligned else "estimated", **report}
        return Speech(samples=audio, sample_rate=rate, text=text, voice=info, timings=timings, metrics=metrics, spoken=spoken)

    def stream(self, text: str, *, voice: Optional[str] = None, language: Optional[str] = None, speed: float = 1.0,
               pause: Union[str, float] = "normal", expressiveness: float = 1.0, rhythm: float = 1.0,
               prepare_text: bool = True, normalize: bool = True, target_lufs: float = -16.0, true_peak_db: float = -1.5,
               cancel=None, timeout: Optional[float] = None) -> Iterator[SpeechChunk]:
        """Speak ``text`` piece by piece (each spoken sentence with the pause after it) so playback can start at once.
        Loudness is set from the first sentence and peaks are limited per piece, so the level is close to, not
        exactly, ``target_lufs``."""
        started = time.monotonic()
        text = clean_text(text, self.max_characters)
        speed, pause_s, expressiveness, rhythm = self._options(speed, pause, expressiveness, rhythm)
        info = self._resolve(voice, language)
        deadline = started + timeout if timeout else None
        gain, pending, index, rate = None, [], 0, info.sample_rate
        for kind, payload in self._generate(text, info, speed, pause_s, expressiveness, rhythm, prepare_text, cancel, deadline):
            if kind == "audio":
                pending.append(payload)
                continue
            segment, rate = payload[0], payload[4].sample_rate
            piece = np.concatenate(pending) if pending else np.zeros(0, np.float32)
            pending = []
            if normalize and piece.size:
                if gain is None:
                    measured = loudness.integrated_loudness(piece, rate)
                    gain = 10 ** (max(-20.0, min(20.0, target_lufs - measured)) / 20) if measured is not None else 1.0
                piece = loudness.limit((piece * gain).astype(np.float32), rate, true_peak_db - 0.5)
            yield SpeechChunk(samples=piece, sample_rate=rate, index=index, start=segment.start, end=segment.end)
            index += 1
        if pending:
            tail = np.concatenate(pending)
            if normalize and gain is not None:
                tail = (tail * gain).astype(np.float32)
            yield SpeechChunk(samples=tail, sample_rate=rate, index=index, start=len(text), end=len(text), final=True)


# ---------------------------------------------------------------------- word timings
def _sound_count(voice, token: str, cache: dict) -> int:
    if token not in cache:
        if "[[" in token:
            sounds = token.replace("[", "").replace("]", "")
        else:
            with _PHONEMIZE:
                sounds = "".join("".join(sentence) for sentence in voice.phonemize(token))
        cache[token] = max(1, len(sounds.replace(" ", "")))
    return cache[token]


def _token_times(voice, segment, stream, start: int, end: int, cache: dict) -> list:
    counts = [_sound_count(voice, token, cache) for token, _, _ in segment.tokens]
    total = sum(counts)
    if not counts:
        return []
    if not stream:
        out, done = [], 0
        for count in counts:
            out.append((start + (end - start) * done / total, start + (end - start) * (done + count) / total))
            done += count
        return out
    sounds = [i for i, (p, _, _) in enumerate(stream) if p != " "]
    if not sounds:
        return [(start, end)] * len(counts)
    word_starts = {k for k, i in enumerate(sounds) if k == 0 or stream[i - 1][0] == " "}
    bounds = textprep.cut_points(counts, len(sounds), word_starts)
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b > a:
            out.append((stream[sounds[a]][1], stream[sounds[b - 1]][2]))
        else:
            edge = stream[sounds[min(a, len(sounds) - 1)]][1]
            out.append((edge, edge))
    return out


def _word_timings(voice, segment, text: str, stream, start: int, end: int, cache: dict) -> list:
    words = [(m.start() + segment.start, m.end() + segment.start) for m in re.finditer(r"\S+", text[segment.start:segment.end])]
    groups: list = []
    for (token, ts, te), (t0, t1) in zip(segment.tokens, _token_times(voice, segment, stream, start, end, cache)):
        hit = tuple(i for i, (a, b) in enumerate(words) if a < te and ts < b)
        if not hit:
            continue
        if groups and groups[-1][0] == hit:
            groups[-1][2] = t1
        else:
            groups.append([hit, t0, t1])
    spans: dict = {}
    for hit, t0, t1 in groups:
        lengths = [words[i][1] - words[i][0] for i in hit]
        done = 0
        for i, length in zip(hit, lengths):
            piece = (t0 + (t1 - t0) * done / sum(lengths), t0 + (t1 - t0) * (done + length) / sum(lengths))
            spans[i] = (min(spans[i][0], piece[0]), max(spans[i][1], piece[1])) if i in spans else piece
            done += length
    return [[words[i][0], words[i][1], spans[i][0], spans[i][1]] for i in sorted(spans)]


def _timings(text: str, timed: list, loaded: _Loaded, rate: int, total: int) -> dict:
    def ms(samples):
        return int(round(samples * 1000 / rate))
    cache: dict = {}
    words = []
    for segment, first, last, stream in timed:
        if segment.tokens:
            words += [[a, b, ms(t0), ms(t1)] for a, b, t0, t1 in
                      _word_timings(loaded.voice, segment, text, stream, first, last, cache)]
    return {"version": 1, "text": text, "duration": ms(total),
            "segments": [[s.start, s.end, ms(a), ms(b)] for s, a, b, _ in timed], "words": words}
