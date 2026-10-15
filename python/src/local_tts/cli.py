# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line: ``local-tts "Hello" -o hello.wav``. Run ``local-tts --help`` or ``local-tts COMMAND --help``.

Commands: say (default), voices, languages, download, serve, cache, doctor.
Exit codes: 0 success, 1 error (message on stderr; --json prints {"error": {code, message}}), 130 interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from . import __version__
from .catalog import find_voice, languages, voices
from .downloader import cache_dir, downloaded_models, ensure_model, remove_model
from .engine import TTS
from .errors import LocalTTSError
from .formats import encode, ffmpeg_path

COMMANDS = {"say", "voices", "languages", "download", "serve", "cache", "doctor"}


def _progress(info):
    total = info["total"] or 1
    sys.stderr.write(f"\rDownloading {info['file']}: {100 * info['bytes'] // total}%")
    if info["bytes"] >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _engine(args) -> TTS:
    return TTS(getattr(args, "voice", None) or "auto", language=getattr(args, "language", None), threads=args.threads,
               offline=args.offline, progress=None if args.quiet else _progress)


def cmd_say(args):
    text = sys.stdin.read() if args.text in (None, "-") else args.text
    fmt = args.format or (os.path.splitext(args.output)[1].lstrip(".").lower() if args.output else "wav") or "wav"
    if not args.output and sys.stdout.isatty():
        raise LocalTTSError("OPTION_INVALID", "Give an output file with -o speech.wav (or pipe the output).")
    speech = _engine(args).synthesize(text, speed=args.speed, pause=args.pause, expressiveness=args.expressiveness,
                                      rhythm=args.rhythm, prepare_text=not args.raw, normalize=not args.no_normalize,
                                      target_lufs=args.target_lufs)
    data = speech.wav(args.bit_depth) if fmt == "wav" else encode(speech.samples, speech.sample_rate, fmt)
    if args.output:
        with open(args.output, "wb") as f:
            f.write(data)
    else:
        sys.stdout.buffer.write(data)
    if args.srt:
        speech.save_srt(args.srt)
    if args.vtt:
        speech.save_vtt(args.vtt)
    if args.json:
        out = speech.to_dict()
        out["output"] = args.output
        print(json.dumps(out, ensure_ascii=False))
    elif not args.quiet and args.output:
        credit = f"\nCredit: {speech.voice.credit}" if speech.voice.credit else ""
        print(f"{args.output}: {speech.duration:.1f} s, voice {speech.voice.id}, made in {speech.metrics['seconds']} s{credit}",
              file=sys.stderr)


def cmd_voices(args):
    found = voices(args.language, sex=args.sex, use=args.use, quality=args.quality, min_score=args.min_score, sort=args.sort)
    shown = found[: args.limit] if args.limit else found
    if args.json:
        print(json.dumps([v.to_dict() for v in shown], ensure_ascii=False))
        return
    print(f"{'VOICE':<38} {'LANGUAGE':<26} {'SEX':<4} {'SCORE':>5} {'NAT':>5} {'CLEAR':>5}  LICENCE")
    for v in shown:
        print(f"{v.id:<38} {v.language[:26]:<26} {v.sex or '-':<4} {v.overall:>5.1f} {v.natural:>5.1f} {v.clear:>5.1f}  {v.licence}")
    if len(found) > len(shown):
        print(f"... {len(found) - len(shown)} more (use --limit 0 for all)")


def cmd_languages(args):
    rows = languages()
    if args.json:
        print(json.dumps([{**r, "best": r["best"].id} for r in rows], ensure_ascii=False))
        return
    for r in rows:
        print(f"{r['locale']:<8} {r['language']:<34} {r['voices']:>5} voices   best: {r['best'].id} ({r['best'].overall})")


def cmd_download(args):
    targets = args.voices or ["auto"]
    for name in targets:
        voice = find_voice(name, args.language)
        ensure_model(voice.model, progress=None if args.quiet else _progress)
        if not args.quiet:
            print(f"{voice.model}: ready ({voice.licence})", file=sys.stderr)


def cmd_serve(args):
    from .server import serve

    tts = TTS(args.voice or "auto", language=args.language, threads=args.threads, offline=args.offline,
              cache_size=args.cache_size, max_characters=args.max_characters, progress=None if args.quiet else _progress)
    serve(args.host, args.port, tts=tts, workers=args.workers, api_key=args.api_key or "", cors=args.cors or "",
          preload=not args.no_preload, quiet=args.quiet)


def cmd_cache(args):
    if args.action == "path":
        print(cache_dir())
    elif args.action == "list":
        for m in downloaded_models():
            print(m)
    elif args.action == "remove":
        for m in args.models:
            print(f"{m}: {'removed' if remove_model(m) else 'not in the cache'}")


def cmd_doctor(args):
    checks = {"local_tts": __version__, "python": platform.python_version(), "platform": platform.platform(),
              "cache_dir": str(cache_dir()), "ffmpeg": ffmpeg_path() or "not found (wav and pcm still work)"}
    try:
        import onnxruntime
        import piper  # noqa: F401
        checks["onnxruntime"] = onnxruntime.__version__
        checks["piper"] = "ok"
    except ImportError as error:
        checks["piper"] = f"missing: {error}"
    try:
        import onnx  # noqa: F401
        checks["word_timings"] = "exact (onnx installed)"
    except ImportError:
        checks["word_timings"] = 'estimated (pip install "local-tts[timings]" for exact)'
    checks["downloaded_voices"] = downloaded_models()
    if args.synthesize:
        started = time.monotonic()
        speech = _engine(args).synthesize("Local text to speech is ready.")
        checks["synthesis"] = {"voice": speech.voice.id, "seconds": round(time.monotonic() - started, 2),
                               "audio_seconds": round(speech.duration, 2)}
    if args.json:
        print(json.dumps(checks, ensure_ascii=False))
    else:
        for k, v in checks.items():
            print(f"{k:<18} {v}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--threads", type=int, help="CPU threads per synthesis (default: up to 4)")
    common.add_argument("--offline", action="store_true", help="never download; use voices already in the cache")
    common.add_argument("--quiet", "-q", action="store_true")
    common.add_argument("--json", action="store_true", help="machine-readable output")

    p = argparse.ArgumentParser(prog="local-tts", description="Natural text to speech on your own computer or server.")
    p.add_argument("--version", action="version", version=f"local-tts {__version__}")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("say", parents=[common], help="speak text into a file (default command)")
    s.add_argument("text", nargs="?", help='text to speak, or "-" to read standard input')
    s.add_argument("-o", "--output", help="output file (.wav; .mp3/.opus/.flac/.aac need FFmpeg)")
    s.add_argument("-v", "--voice", help="voice id, model name or OpenAI voice name (default: best for the language)")
    s.add_argument("-l", "--language", help="language, e.g. en-US, es-ES, de (default en-US)")
    s.add_argument("--speed", type=float, default=1.0, help="0.5 to 2.0")
    s.add_argument("--pause", default="normal", help="short, normal, long, or seconds between sentences")
    s.add_argument("--expressiveness", type=float, default=1.0, help="0.5 to 1.5")
    s.add_argument("--rhythm", type=float, default=1.0, help="0.5 to 1.5")
    s.add_argument("--raw", action="store_true", help="do not prepare the text (numbers, dates, abbreviations)")
    s.add_argument("--no-normalize", action="store_true", help="skip loudness normalization")
    s.add_argument("--target-lufs", type=float, default=-16.0)
    s.add_argument("--bit-depth", type=int, default=16, choices=(16, 24))
    s.add_argument("--format", help="wav, pcm, mp3, opus, aac or flac (default from the file name)")
    s.add_argument("--srt", help="also write SRT subtitles")
    s.add_argument("--vtt", help="also write VTT subtitles")
    s.set_defaults(func=cmd_say)

    v = sub.add_parser("voices", parents=[common], help="list voices, best first")
    v.add_argument("-l", "--language")
    v.add_argument("--sex", choices=("F", "M", "female", "male"))
    v.add_argument("--use", nargs="*", choices=("none", "credit", "copyleft", "unclear"), help="licence classes")
    v.add_argument("--quality", nargs="*", choices=("x_low", "low", "medium", "high"))
    v.add_argument("--min-score", type=float)
    v.add_argument("--sort", default="overall", choices=("overall", "natural", "clear", "name"))
    v.add_argument("--limit", type=int, default=20, help="rows to show (0 for all)")
    v.set_defaults(func=cmd_voices)

    lg = sub.add_parser("languages", parents=[common], help="languages with voices")
    lg.set_defaults(func=cmd_languages)

    d = sub.add_parser("download", parents=[common], help="download voices now (for offline use or Docker images)")
    d.add_argument("voices", nargs="*", help="voice ids or model names (default: best voice for --language)")
    d.add_argument("-l", "--language")
    d.set_defaults(func=cmd_download)

    sv = sub.add_parser("serve", parents=[common], help="run an OpenAI-compatible speech server")
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept other machines (set an --api-key)")
    sv.add_argument("--port", type=int, default=8880)
    sv.add_argument("-v", "--voice", help="default voice")
    sv.add_argument("-l", "--language", help="default language")
    sv.add_argument("--workers", type=int, default=1, help="syntheses at the same time")
    sv.add_argument("--cache-size", type=int, default=4, help="voices kept loaded")
    sv.add_argument("--max-characters", type=int, default=20000)
    sv.add_argument("--api-key", help="require this bearer token (or set LOCAL_TTS_API_KEY)")
    sv.add_argument("--cors", help="allow this browser origin, e.g. http://localhost:3000 or *")
    sv.add_argument("--no-preload", action="store_true", help="load the default voice on first request")
    sv.set_defaults(func=cmd_serve)

    c = sub.add_parser("cache", parents=[common], help="show or clean the voice cache")
    c.add_argument("action", choices=("path", "list", "remove"))
    c.add_argument("models", nargs="*")
    c.set_defaults(func=cmd_cache)

    dr = sub.add_parser("doctor", parents=[common], help="check the installation")
    dr.add_argument("--synthesize", action="store_true", help="also make a short test sentence")
    dr.add_argument("-v", "--voice")
    dr.add_argument("-l", "--language")
    dr.set_defaults(func=cmd_doctor)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv.insert(0, "say")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        args.func(args)
        return 0
    except LocalTTSError as error:
        if getattr(args, "json", False):
            print(json.dumps({"error": error.to_dict()}))
        else:
            print(f"local-tts: {error.message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
