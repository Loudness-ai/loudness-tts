# SPDX-License-Identifier: GPL-3.0-or-later
"""The voice catalogue: every commercially usable Piper voice, ranked by naturalness and clarity.

Scores come from Loudness.ai's open voice study (LA-RB-2026-003,
https://loudness.ai/research/piper-tts-voices-ranked/). Compare scores within one language only.

    from loudness_tts import voices, find_voice
    voices(language="es")                      # Spanish voices, best first
    voices(language="en-US", sex="female")
    find_voice("auto", language="de-DE")       # the best German voice
"""
from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Optional

from .errors import TTSError

USE_CLASSES = ("none", "credit", "copyleft", "unclear")
# Voices picked automatically have a clear licence for commercial use (no conditions, or a credit line).
AUTO_USE = ("none", "credit")
DEFAULT_LANGUAGE = "en-US"


@dataclass(frozen=True)
class Voice:
    id: str
    model: str
    speaker: Optional[int]
    name: str
    language: str          # language variant as ranked in the study, e.g. "English (United States)"
    locale: str            # BCP 47 tag, e.g. "en-US"
    quality: str           # Piper quality tier: x_low, low, medium or high
    sex: Optional[str]     # "F" or "M" where the training data records the reader, else None
    overall: float         # 0-100: (natural + clear) / 2, +4 for 22 kHz voices
    natural: float         # 0-100: UTMOS22 predicted naturalness
    clear: float           # 0-100: 100 minus Whisper large-v3 character error rate
    rank: int              # rank within its language variant
    of: int
    licence: str
    use: str               # licence class: none, credit, copyleft or unclear
    dataset: str
    source: str            # "catalog" (official Piper voices) or "community"
    sample_rate: int
    note: Optional[str] = None

    @property
    def credit(self) -> Optional[str]:
        """A credit line to show where your users can see it, or None when the licence asks for none."""
        if self.use == "none":
            return None
        data = f", trained on {self.dataset}" if self.dataset else ""
        return f'Voice "{self.name}" (Piper voice {self.model}){data}, {self.licence}.'

    def to_dict(self) -> dict:
        d = asdict(self)
        d["credit"] = self.credit
        return d


@functools.lru_cache(maxsize=1)
def _data() -> dict:
    text = resources.files("loudness_tts").joinpath("data/voices.json").read_text(encoding="utf-8")
    return json.loads(text)


@functools.lru_cache(maxsize=1)
def _all() -> tuple:
    data = _data()
    fields = data["fields"]
    out = []
    for row in data["voices"]:
        v = dict(zip(fields, row))
        m = data["models"][v["model"]]
        out.append(Voice(id=v["id"], model=v["model"], speaker=v["speaker"], name=v["name"], language=v["language"],
                         locale=m["locale"], quality=m["quality"], sex=v["sex"], overall=v["overall"],
                         natural=v["natural"], clear=v["clear"], rank=v["rank"], of=v["of"], licence=m["licence"],
                         use=m["use"], dataset=m["dataset"], source=m["source"], sample_rate=m["sampleRate"],
                         note=m.get("note")))
    return tuple(out)


def model_info(model: str) -> dict:
    """Download files, checksums, licence and card of one model file."""
    try:
        return _data()["models"][model]
    except KeyError:
        raise TTSError("VOICE_NOT_FOUND", f"No voice model called {model!r} is in the catalogue.") from None


def _matches_language(v: Voice, language: str) -> bool:
    want = language.replace("_", "-").lower()
    have = v.locale.lower()
    return have == want or have.split("-")[0] == want or v.language.lower() == want


def voices(language: Optional[str] = None, *, sex: Optional[str] = None, use=None, quality=None, source: Optional[str] = None,
           min_score: Optional[float] = None, sort: str = "overall") -> list:
    """Voices matching every given filter, best first (``sort``: overall, natural, clear or name).

    language: a locale ("en-US", "es-MX"), a language code ("es") or a study language name.
    sex: "F" or "M" (voices whose reader is not recorded are left out when given).
    use: a licence class or a list of them: none, credit, copyleft, unclear.
    quality: a tier or a list of them: x_low, low, medium, high.
    """
    uses = (use,) if isinstance(use, str) else tuple(use) if use else None
    qualities = (quality,) if isinstance(quality, str) else tuple(quality) if quality else None
    out = [v for v in _all()
           if (language is None or _matches_language(v, language))
           and (sex is None or v.sex == sex.upper()[:1])
           and (uses is None or v.use in uses)
           and (qualities is None or v.quality in qualities)
           and (source is None or v.source == source)
           and (min_score is None or v.overall >= min_score)]
    if sort == "name":
        out.sort(key=lambda v: v.name.lower())
    elif sort in ("overall", "natural", "clear"):
        out.sort(key=lambda v: (-getattr(v, sort), v.id))
    else:
        raise TTSError("OPTION_INVALID", "sort must be overall, natural, clear or name.")
    return out


def languages() -> list:
    """Locales with at least one voice, with their best voice and how many voices they have."""
    best: dict = {}
    for v in _all():
        entry = best.setdefault(v.locale, {"locale": v.locale, "language": v.language, "voices": 0, "best": v})
        entry["voices"] += 1
        if v.overall > entry["best"].overall:
            entry["best"] = v
    return sorted(best.values(), key=lambda e: e["locale"])


# Voice names used by OpenAI's speech API, mapped to strong English (US) voices so existing clients work unchanged.
OPENAI_NAMES = {
    "female": ("nova", "shimmer", "coral", "sage", "marin"),
    "male": ("echo", "onyx", "ash", "ballad", "verse", "cedar"),
    "any": ("alloy", "fable"),
}


@functools.lru_cache(maxsize=1)
def openai_voice_map() -> dict:
    pool = voices(DEFAULT_LANGUAGE, use=AUTO_USE)
    picks: dict = {}
    for sex, names in (("F", OPENAI_NAMES["female"]), ("M", OPENAI_NAMES["male"])):
        ranked = [v for v in pool if v.sex == sex]
        for name, voice in zip(names, ranked):
            picks[name] = voice.id
    for name, voice in zip(OPENAI_NAMES["any"], pool):
        picks[name] = voice.id
    return picks


def find_voice(voice: Optional[str] = "auto", language: Optional[str] = None) -> Voice:
    """Resolve a voice id, a model name (its best speaker), an OpenAI voice name, "auto" or "best:<language>".

    "auto" picks the best voice with a clear commercial licence for ``language`` (default en-US).
    """
    if voice in (None, "", "auto", "best"):
        return _best(language or DEFAULT_LANGUAGE)
    if voice.startswith("best:"):
        return _best(voice.split(":", 1)[1])
    by_id = {v.id: v for v in _all()}
    if voice in by_id:
        return by_id[voice]
    speakers = [v for v in _all() if v.model == voice]
    if speakers:
        return min(speakers, key=lambda v: (v.rank, v.id))
    mapped = openai_voice_map().get(voice.lower())
    if mapped:
        return by_id[mapped]
    raise TTSError("VOICE_NOT_FOUND", f"No voice called {voice!r}. Run `loudness-tts voices` to see them.")


def _best(language: str) -> Voice:
    pool = voices(language, use=AUTO_USE) or voices(language)
    if not pool:
        raise TTSError("VOICE_NOT_FOUND", f"No voice for {language!r}. Run `loudness-tts languages` to see the list.")
    return pool[0]
