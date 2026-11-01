"""Portable saved-output reconstruction for AMP evaluation v9; no model training/inference."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def verify():
 m=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
 bad=[]
 for name,h in m['files'].items():
  p=ROOT/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(name)
 if bad:raise RuntimeError('Input checksum mismatch: '+str(bad))
 print('INPUT HASHES PASS',len(m['files']),flush=True)

def run(script,*args):
 env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
 subprocess.run([sys.executable,'-B',str(ROOT/'code'/script),*map(str,args)],cwd=ROOT,env=env,check=True)

def analyses():
 run('analyses/run_reanalysis.py','--scope','151','--bootstrap','10000')
 run('analyses/analyze_threshold_task.py')
 run('analyses/decompose_sources.py')

def check_results():
 pairs=[('membership/active_design_151','membership_analysis/active_design_151'),('task_threshold','task_threshold'),('source_decomposition','source_decomposition')]
 records=[]
 def read(p):
  with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
 for out,frozen in pairs:
  for p in sorted((ROOT/'rebuild_analyses'/out).glob('*.tsv')):
   q=ROOT/'frozen'/frozen/p.name
   if not q.exists() or any(x in p.name for x in ['来源文件','指纹']):continue
   aa,bb=read(p),read(q);issues=[];cells=0
   if len(aa)!=len(bb):issues.append('row_count')
   else:
    for i,(a,b) in enumerate(zip(aa,bb)):
     if set(a)!=set(b):issues.append('column_names');break
     for k,v in a.items():
      cells+=1
      if v==b[k] or k in ['path','input_path','output_path']:continue
      try:
       x,y=float(v),float(b[k])
       if (math.isnan(x) and math.isnan(y)) or math.isclose(x,y,rel_tol=1e-11,abs_tol=1e-12):continue
      except (ValueError,TypeError):pass
      issues.append([i,k,v,b[k]])
   records.append(dict(file=str(p.relative_to(ROOT)),rows=len(aa),cells=cells,passed=not issues,issues=issues[:10]))
 if not records:raise RuntimeError('No reconstructed tables found; run analyses first')
 (ROOT/'reconstruction_checks.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 if not all(x['passed'] for x in records):raise RuntimeError('Reconstruction mismatch; inspect reconstruction_checks.json')
 print('NUMERIC RECONSTRUCTION PASS',len(records),'tables;',sum(x['cells'] for x in records),'cells',flush=True)

def figures():run('plot_v9.py','--output',ROOT/'rebuild_figures');audit_figures()

def audit_figures():
 # Load optional desktop dependency paths only after NumPy: preserve the pinned statistical version.
 import numpy
 for p in os.environ.get('AMP_EXTRA_PYTHONPATH','').split(os.pathsep):
  if p:sys.path.insert(0,p)
 env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
 if env.get('AMP_EXTRA_PYTHONPATH'):env['PYTHONPATH']=env['AMP_EXTRA_PYTHONPATH']
 receipt=[]
 for p in sorted((ROOT/'rebuild_figures').glob('*/PDF/*.pdf')):
  q=p.parent.parent/'QA';q.mkdir(exist_ok=True)
  for name in ['audit_pdf_text.py','audit_figure_collisions.py']:
   out=q/(p.stem+'.'+name+'.json')
   cmd=[sys.executable,'-B',str(ROOT/'code'/name),str(p)]
   if name=='audit_pdf_text.py':cmd+=['--min-pt','5','--json']
   else:cmd+=['--json-out',str(out)]
   r=subprocess.run(cmd,env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
   (q/(p.stem+'.'+name+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
   receipt.append(dict(figure=p.stem,audit=name,exit=r.returncode))
 (ROOT/'figure_audit_summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 if len(receipt)!=50 or any(r['exit'] for r in receipt):raise RuntimeError('Figure audit incomplete/failed; see QA logs')
 print('FIGURE AUDITS PASS 50/50',flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['verify','analyses','check-results','figures','audit-figures','documents','all']);a=ap.parse_args()
 if a.action in ['verify','all','analyses','figures']:verify()
 if a.action in ['analyses','all']:analyses()
 if a.action in ['check-results','all']:check_results()
 if a.action in ['figures','all']:figures()
 if a.action=='audit-figures':audit_figures()
 if a.action=='documents':
  verify()
  import shutil
  destination=ROOT/'document_build';destination.mkdir(exist_ok=True)
  for source in (ROOT/'manuscript').glob('*_v11_整合清洁版.docx'):
   shutil.copy2(source,destination/source.name)
  print('COPIED CURRENT V11 MANUSCRIPTS; numerical reconstruction uses the preserved v9 code',flush=True)
if __name__=='__main__':main()

