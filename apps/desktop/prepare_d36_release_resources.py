import hashlib
import json
import shutil
import zipfile
from pathlib import Path

source = Path('D:/AI地学/apps/d36_desktop_predictor/assets')
repository = Path('D:/AI地学/repos/basalt-geochemistry')
destination = repository / 'apps/desktop/resources'
destination.mkdir(parents=True, exist_ok=True)
files = sorted(p for p in source.rglob('*') if p.is_file())
model_files = [p for p in files if p.name.startswith('d36_full_seed_')]
manifest = {'version': '1.0.2', 'model': 'GeoSinglePrior-D36-v1',
            'model_training_rows': 16076, 'retrieval_library_rows': 16346,
            'archives': [], 'files': []}
for name, members in [(p.stem + '.zip', [p]) for p in model_files] + [('support.zip', [p for p in files if p not in model_files])]:
    archive_path = destination / name
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_LZMA) as archive:
        for path in members:
            relative = path.relative_to(source).as_posix()
            content = path.read_bytes()
            if relative == 'retrieval_library_v1.json':
                meta = json.loads(content)
                meta['source_data'] = 'experiment_v5_clean_identity_expansion06_2026-10-22/D36_v9_clean_expansion06.parquet'
                content = json.dumps(meta, ensure_ascii=False, indent=2).encode('utf-8')
            archive.writestr(relative, content)
            manifest['files'].append({'path': relative, 'sha256': hashlib.sha256(content).hexdigest()})
    manifest['archives'].append({'file': name, 'bytes': archive_path.stat().st_size,
                                 'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()})
(destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
# Assets in the source folder remain ignored and are used for source-level replay.
target = repository / 'apps/desktop/assets'
shutil.copytree(source, target, dirs_exist_ok=True)
print(json.dumps(manifest['archives'], indent=2))
