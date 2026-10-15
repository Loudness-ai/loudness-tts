"""Builds the voice catalogue shipped with loudness-tts from the published voice study (Loudness.ai LA-RB-2026-003).

  python3 scripts/build_catalog.py

Writes python/src/loudness_tts/data/voices.json (and javascript/src/data/voices.json when that folder exists): every
commercially usable Piper voice with its scores, licence, credit line and download files pinned to the tested revision,
each file with its size and checksum. Nothing here says which voices the Loudness.ai website offers.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RESEARCH = HERE.parent.parent / 'research'
STUDY = RESEARCH / 'tts-voice-study-2026-10/public'
CATALOG = json.loads((RESEARCH / 'tts-voice-benchmark-2026-10/data/piper-catalog-voices.json').read_text())
BRYCE = RESEARCH / 'tts-voice-sets-2026-10/models/bryce-beattie'
SHA = dict(line.split()[::-1] for line in (BRYCE / 'SHA256SUMS').read_text().splitlines() if line.strip())
REVISION = 'c10ece1aade47bb51c153c893d14e5bf8e5b7117'
HF = f'https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}/'
RENAME = {'bb-clean100': 'en_US-clean100-medium', 'bb-manyvoice': 'en_US-manyvoice-medium', 'bb-jenny': 'en_IE-jenny-medium'}
RATE = {'x_low': 16000, 'low': 16000, 'medium': 22050, 'high': 22050}

study_voices = json.loads((STUDY / 'voices.json').read_text())['voices']
study_models = json.loads((STUDY / 'models.json').read_text())
models = {}
for key, m in study_models.items():
    public = RENAME.get(key, key)
    if m['source'] == 'catalog':
        entry = CATALOG[key]
        onnx = next(p for p in entry['files'] if p.endswith('.onnx'))
        cfg = next(p for p in entry['files'] if p.endswith('.onnx.json'))
        files = {
            'model': {'url': HF + onnx, 'size': entry['files'][onnx]['size_bytes'], 'md5': entry['files'][onnx]['md5_digest']},
            'config': {'url': HF + cfg, 'size': entry['files'][cfg]['size_bytes'], 'md5': entry['files'][cfg]['md5_digest']},
        }
    else:
        stem = m['model'].rsplit('/', 1)[1].removesuffix('.onnx')
        files = {
            'model': {'url': m['model'], 'size': (BRYCE / f'{stem}.onnx').stat().st_size, 'sha256': SHA[f'{stem}.onnx']},
            'config': {'url': m['config'], 'size': (BRYCE / f'{stem}.onnx.json').stat().st_size, 'sha256': SHA[f'{stem}.onnx.json']},
        }
    models[public] = {
        'locale': m['locale'].replace('_', '-'), 'quality': m['quality'], 'sampleRate': RATE[m['quality']],
        'speakers': m['speakers'], 'licence': m['licence'], 'use': m['use'], 'dataset': m.get('dataset') or '',
        'source': m['source'], 'card': m['card'], 'files': files, **({'note': m['note']} if m.get('note') else {}),
    }

voices = []
for v in sorted(study_voices, key=lambda v: (v['language'], v['rank'])):
    model = RENAME.get(v['model'], v['model'])
    vid = model if v['speaker'] is None or models[model]['speakers'] == 1 else f"{model}-s{v['speaker']}"
    voices.append([vid, model, v['speaker'] if models[model]['speakers'] > 1 else None, v['name'], v['language'], v['sex'],
                   v['overall'], v['natural'], v['clear'], v['rank'], v['of']])
out = {
    'version': 1,
    'source': 'Loudness.ai technical report LA-RB-2026-003, Open Piper TTS Voices Ranked (2026-10-15), '
              'https://loudness.ai/research/piper-tts-voices-ranked/',
    'pinnedRevision': REVISION,
    'fields': ['id', 'model', 'speaker', 'name', 'language', 'sex', 'overall', 'natural', 'clear', 'rank', 'of'],
    'models': models,
    'voices': voices,
}
text = json.dumps(out, ensure_ascii=False, separators=(',', ':'))
for target in (HERE / 'python/src/loudness_tts/data/voices.json', HERE / 'javascript/src/data/voices.json'):
    if target.parent.parent.exists() or target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(target.relative_to(HERE), len(voices), 'voices', len(models), 'models', round(len(text) / 1024), 'KB')
