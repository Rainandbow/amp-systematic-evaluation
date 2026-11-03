"""Verified manuscript release entry points. Does not run AMP inference."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
 m=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf8'))['files']
 for rel,h in m.items():
  p=ROOT/rel
  if not p.is_file() or digest(p)!=h:raise RuntimeError('Release checksum mismatch: '+rel)
 print('RELEASE HASHES PASS',len(m),flush=True)
def child(script,*args,cwd=None,env=None):
 e=os.environ.copy();e.update({'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 if env:e.update(env)
 subprocess.run([sys.executable,'-B',str(script),*map(str,args)],cwd=cwd or ROOT,env=e,check=True)
def workspace(out,results=False):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 for module in ['reliability','refit']:
  src=ROOT/module;dst=out/module;dst.mkdir()
  for p in src.iterdir():
   if p.name in ['results','figures','qa','__pycache__']:continue
   if p.is_dir():shutil.copytree(p,dst/p.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   else:shutil.copy2(p,dst/p.name)
  for d in ['results','figures','qa']:(dst/d).mkdir()
  if results:
   shutil.copytree(src/'results',dst/'results',dirs_exist_ok=True)
 if not results:
  p=ROOT/'reliability/results/inventory_nearest_reference.tsv.gz';shutil.copy2(p,out/'reliability/results'/p.name)
 return out
def verify_raw(source):
 old=json.loads((ROOT/'docs/RELIABILITY_ORIGINAL_INPUT_FINGERPRINTS.json').read_text(encoding='utf8'))
 for n in ['historical_members.tsv.gz','historical_scores.tsv.gz','references.tsv.gz','reference_membership.tsv.gz']:
  p=Path(source)/n
  if not p.is_file() or digest(p)!=old[n]['sha256']:raise RuntimeError('Exact source snapshot unavailable/mismatched: '+n+'; see docs/RELIABILITY_DATA_ACCESS.md')
 print('EXTERNAL RAW SNAPSHOT HASHES PASS',flush=True)
 return old
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('action',choices=['check','selftest','main-points','main-figures','reliability-statistics','reliability-figures','refit-smoke','refit-statistics','refit-figure','distance-check','distance'])
 ap.add_argument('--output');ap.add_argument('--source-dir');ap.add_argument('--workers',type=int,default=3)
 a=ap.parse_args();check()
 if a.action=='check':return
 if a.action=='selftest':
  child(ROOT/'reliability/run.py','check',cwd=ROOT/'reliability')
  child(ROOT/'inference_reference/verify_publication.py')
  return
 if a.action.startswith('main-'):
  if not a.output:ap.error('--output must name a new directory')
  child(ROOT/'six_main_figures_v1_1_20261101/run.py','points' if a.action=='main-points' else 'figures','--output',Path(a.output).resolve());return
 if a.action=='distance-check':
  if not a.source_dir:ap.error('--source-dir required')
  verify_raw(a.source_dir);return
 if not a.output:ap.error('--output must name a new directory')
 if a.action=='distance':
  if not a.source_dir:ap.error('--source-dir required')
  old=verify_raw(a.source_dir);out=workspace(a.output)
  m=json.loads((out/'reliability/input_manifest.json').read_text(encoding='utf8'))
  for n in ['historical_members.tsv.gz','historical_scores.tsv.gz','references.tsv.gz','reference_membership.tsv.gz']:
   shutil.copy2(Path(a.source_dir)/n,out/'reliability/inputs'/n);m[n]=old[n]
  (out/'reliability/input_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8')
  child(out/'reliability/run.py','distance','--workers',a.workers,cwd=out/'reliability');return
 out=workspace(a.output,results=a.action.endswith('figures') or a.action=='refit-figure')
 if a.action=='reliability-statistics':child(out/'reliability/run.py','statistics',cwd=out/'reliability')
 elif a.action=='reliability-figures':child(out/'reliability/run.py','plots',cwd=out/'reliability')
 elif a.action in ['refit-smoke','refit-statistics']:
  args=['--smoke'] if a.action=='refit-smoke' else ['--workers',str(a.workers)]
  child(out/'refit/code/refit_uncertainty_v2.py',*args,cwd=out/'refit')
 elif a.action=='refit-figure':child(out/'refit/code/plot_refit_s29.py',cwd=out/'refit')
 print('COMPLETE',a.action,out,flush=True)
if __name__=='__main__':main()
