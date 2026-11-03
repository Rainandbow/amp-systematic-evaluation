"""Portable entry points; all paths resolve relative to this folder."""
from pathlib import Path
import os,sys,json,argparse,subprocess,hashlib,importlib,platform

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python_deps'))
sys.path.insert(0,str(ROOT/'code'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('PYTHONIOENCODING','utf-8')
EXPECTED_PROTOCOL='1fe201f7eeb1248eadc52b420ad67940206466121af2521dad72a2b11ccc1c46'

def check():
    for name,item in json.loads((ROOT/'input_manifest.json').read_text(encoding='utf8')).items():
        p=ROOT/'inputs'/name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Input changed: '+name
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==EXPECTED_PROTOCOL,'Protocol changed'
    for name in ['numpy','pandas','matplotlib','rapidfuzz','pymupdf']:
        m=importlib.import_module(name);print(name,getattr(m,'__version__','available'),flush=True)
    for folder in ['results','qa','figures','cache']:(ROOT/folder).mkdir(exist_ok=True)
    import analyze
    analyze.selftest()
    print('INPUT AND STATISTICAL SELF-CHECK PASS',flush=True)

def call(script,*args):subprocess.run([sys.executable,str(ROOT/'code'/script),*args],cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','distance','statistics','statistics-resume','plots','audit','all']);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();check()
    if a.action in ['distance','all']:call('distance_features.py','all','--workers',str(a.workers))
    if a.action in ['statistics','all']:call('analyze.py','all')
    if a.action=='statistics-resume':
        call('analyze.py','historical','--resume');call('analyze.py','experimental')
    if a.action in ['plots','all']:call('plot_results.py')
    if a.action in ['audit','all']:call('finalize.py')

if __name__=='__main__':main()
