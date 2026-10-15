"""Development helper: copy voice models that are already on disk into a loudness-tts cache, verifying each against the
catalogue's checksum exactly as a download would, so tests run without network access.
  python3 scripts/seed_cache.py <cache dir> <folder with .onnx files> [model ...]"""
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python/src'))
from loudness_tts.catalog import model_info  # noqa: E402
from loudness_tts.downloader import model_paths  # noqa: E402

root, folder, wanted = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
names = {'en_US-clean100-medium': 'clean100', 'en_US-manyvoice-medium': 'mv2', 'en_IE-jenny-medium': 'jenny'}
for model in wanted:
    info = model_info(model)
    stem = names.get(model, model)
    for key, suffix in (('model', '.onnx'), ('config', '.onnx.json')):
        src = folder / f'{stem}{suffix}'
        spec = info['files'][key]
        algo = 'sha256' if 'sha256' in spec else 'md5'
        h = hashlib.new(algo, src.read_bytes()).hexdigest()
        assert src.stat().st_size == spec['size'] and h == spec[algo], (src, 'checksum mismatch')
        dst = model_paths(model, root)[0 if key == 'model' else 1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        dst.with_name(dst.name + '.ok').write_text(f'{algo}:{spec[algo]}')
    print('seeded', model)
