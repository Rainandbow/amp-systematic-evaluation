"""Public entry point for checks, numerical self-tests and figure reconstruction."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files'].items():
        path = ROOT/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Repository file mismatch: '+rel)
    print('REPOSITORY HASHES PASS', len(manifest['files']), flush=True)

def child(script, *args):
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    subprocess.run([sys.executable, '-B', str(ROOT/script), *args], cwd=ROOT, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check','test','figures','analysis-check','baseline','transfer','all-analyses'])
    args = parser.parse_args()
    verify()
    if args.action == 'check':
        child('figures/run.py', 'check')
        child('analysis/run.py', 'check')
    elif args.action == 'test':
        sys.path.insert(0, str(ROOT/'analysis/code'))
        import selftest
        checks = selftest.run()
        for item in checks:
            print('TEST PASS', item)
        print('NUMERICAL SELF-TESTS PASS', len(checks))
        child('analysis/run.py', 'check')
    elif args.action == 'figures':
        child('figures/run.py', 'all')
    else:
        needed = ROOT/'analysis/inputs'
        if not needed.is_dir():
            raise SystemExit('Bundled public inputs are missing. Download the complete updated repository or source ZIP; see docs/DATA_ACCESS.md.')
        action = {'analysis-check':'check','baseline':'baseline','transfer':'honest','all-analyses':'all'}[args.action]
        child('analysis/run.py', action)

if __name__ == '__main__':
    main()
