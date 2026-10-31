"""Import the eight frozen inputs from an existing local author analysis package."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing baseline/honest package directory or ZIP')
    args = parser.parse_args()
    expected = json.loads((ROOT/'analysis/input_manifest.json').read_text(encoding='utf-8'))['files']
    contents = {}
    if args.source.is_dir():
        folder = args.source/'inputs' if (args.source/'inputs').is_dir() else args.source
        for name in expected:
            contents[name] = (folder/name).read_bytes()
    else:
        with ZipFile(args.source) as archive:
            for name in expected:
                matches = [key for key in archive.namelist() if key.endswith('/inputs/'+name) or key == 'inputs/'+name]
                if len(matches) != 1:
                    raise RuntimeError('No unique expected input in archive: '+name)
                contents[name] = archive.read(matches[0])
    destination = ROOT/'analysis/inputs'
    for name, content in contents.items():
        if hashlib.sha256(content).hexdigest() != expected[name]:
            raise RuntimeError('Input checksum mismatch: '+name)
        target = destination/name
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError('Refusing to overwrite a different existing input: '+name)
    for name, content in contents.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
    print('IMPORTED AND SHA256-VERIFIED', len(contents), 'inputs into', destination)
    print('These local inputs are excluded by .gitignore; public deposition is a separate step.')

if __name__ == '__main__':
    main()
