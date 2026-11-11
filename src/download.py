"""Download the official version-4 Kaggle archive; retain its provenance."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

URL = 'https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database?datasetVersionNumber=4'


def main():
    root = Path('data'); root.mkdir(exist_ok=True)
    archive = root / 'covid19-radiography-v4.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.zip.partial')
        with urlopen(URL, timeout=60) as response, temporary.open('wb') as out:
            total = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk: break
                out.write(chunk); total += len(chunk)
                print(f'Downloaded {total / 1e6:.1f} MB', flush=True)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError('The official endpoint did not return a ZIP archive')
        temporary.rename(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): digest.update(chunk)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe archive path')
            if member.is_dir() or 'masks' in path.parts: continue
            z.extract(member, root)
    (root / 'source.json').write_text(json.dumps({
        'source_url': URL, 'requested_version': 4,
        'archive_sha256': digest.hexdigest(), 'archive_bytes': archive.stat().st_size,
        'retrieved_for': 'new reconstruction run; not the unavailable original experiment',
        'excluded_from_extraction': 'lung masks; not used in the classification experiment'
    }, indent=2) + '\n')
    print('Official data extracted; provenance recorded.', flush=True)


if __name__ == '__main__': main()
