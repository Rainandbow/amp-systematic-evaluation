"""One entry point for all six current main figures; frozen predictions, no inference."""
from pathlib import Path
import argparse,csv,json,hashlib,subprocess,sys,os,shutil,importlib.util
ROOT=Path(__file__).resolve().parent
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def verify():
 m=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
 for rel,h in m['files'].items():
  p=ROOT/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('Checksum mismatch: '+rel)
 print('PACKAGE HASHES PASS',len(m['files']),flush=True)
def child(script,*args):
 env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
 subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,args)],cwd=ROOT,env=env,check=True)
def load(name):
 sys.path.insert(0,str(ROOT/'tools'))
 spec=importlib.util.spec_from_file_location(name,ROOT/'code'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def render(out,figures):
 out.mkdir(parents=True,exist_ok=False)
 for n in ['PDF','SVG','PNG_600dpi','TIFF_600dpi','preview','QA','source_data']:(out/n).mkdir()
 core=json.loads((ROOT/'inputs/historical/frozen_core.json').read_text(encoding='utf8'))
 if any(x in figures for x in [1,2,3]):
  m=load('plot_historical')
  data=dict(core=core,patterns=read(ROOT/'inputs/historical/exclusive_membership_patterns.tsv'),length=read(ROOT/'inputs/historical/length_mcc_intervals.tsv'),new_summary=read(ROOT/'inputs/historical/new_panel_summary.tsv'))
  for r in data['new_summary']:
   if r['cohort']=='HydrAMP_no_auxiliary_preselection':r['generated_candidates']='7';r['generated_positive']='2'
  for i in [1,2,3]:
   if i in figures:getattr(m,'fig'+str(i))(data,argparse.Namespace(output=out))
 if 4 in figures:
  local=out/'_figure4';local.mkdir()
  for n in ['inputs','tables','tools']:shutil.copytree(ROOT/'figure4'/n,local/n)
  child('code/plot_paired.py',local)
  src=local/'figures/Figure_4_updated_paired_gain'
  for ext,dst in [('pdf','PDF'),('svg','SVG'),('png','PNG_600dpi'),('tiff','TIFF_600dpi')]:shutil.copy2(src.with_suffix('.'+ext),out/dst/('Fig04_budget_paired_and_family.'+ext))
  shutil.copy2(src.with_suffix('.png'),out/'preview/Fig04_budget_paired_and_family.png')
  shutil.copy2(src.with_suffix('.alignment.json'),out/'QA/Fig04.alignment.json')
  shutil.copy2(ROOT/'figure4/tables/Table_S12_all_48_paired_hit_differences.tsv',out/'source_data/Fig04_all_48_paired_differences.tsv')
 if 5 in figures:
  m=load('plot_transfer');stage=out/'_figure5'
  m.render(stage,read(ROOT/'formal_analysis/figures/inputs/honest_transfer/selected_rules.tsv'),read(ROOT/'formal_analysis/figures/inputs/honest_transfer/evaluation_points.tsv'))
  for ext,dst in [('pdf','PDF'),('svg','SVG'),('png','PNG_600dpi'),('tiff','TIFF_600dpi')]:
   # Archived exporter uses .tif for TIFF.
   src=stage/({'pdf':'PDF','svg':'SVG','png':'PNG','tiff':'TIFF'}[ext])/('Fig05_honest_source_constraints.'+ext)
   if not src.exists() and ext=='tiff':src=stage/'TIFF/Fig05_honest_source_constraints.tif'
   shutil.copy2(src,out/dst/('Fig05_honest_source_constraints.'+ext))
  for p in (stage/'source_data').glob('*.tsv'):shutil.copy2(p,out/'source_data'/p.name)
  for p in (stage/'QA').glob('*.json'):shutil.copy2(p,out/'QA'/p.name)
 if 6 in figures:
  # The current Figure6 uses the archived table definitions and corrected display terms.
  m=load('plot_experimental');load('plot_transfer').style();m.main6(out,core)
 audits=[]
 for pdf in sorted((out/'PDF').glob('*.pdf')):
  for script,extra in [('audit_pdf_text.py',['--min-pt','5','--json']),('audit_figure_collisions.py',['--json-out',str(out/'QA'/(pdf.stem+'.collision.json'))])]:
   r=subprocess.run([sys.executable,'-B',str(ROOT/'tools'/script),str(pdf),*extra],capture_output=True,text=True,encoding='utf8',errors='replace')
   (out/'QA'/(pdf.stem+'.'+script+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
   audits.append({'figure':pdf.stem,'audit':script,'exit_code':r.returncode})
 (out/'QA/audits.json').write_text(json.dumps(audits,indent=2))
 assert len(list((out/'PDF').glob('*.pdf')))==len(figures)
 assert not any(r['exit_code'] for r in audits),'Figure QA failed: see QA logs'
 (out/'COMPLETE.json').write_text(json.dumps({'figures':figures,'new_inference':False,'bootstrap_rerun':False,'intervals':'saved formal intervals','audit_count':len(audits)},indent=2))
 print('SIX-FIGURE RECONSTRUCTION PASS',out,flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','points','figures','uncertainty','baseline','transfer','all']);p.add_argument('--output',default=str(ROOT/'rebuilt'));p.add_argument('--figure',choices=['all','1','2','3','4','5','6'],default='all');a=p.parse_args();verify()
 out=Path(a.output).resolve()
 if a.action in ['points','all']:child('code/points.py',out/'recalculated')
 if a.action in ['figures','all']:render(out/'figures',list(range(1,7)) if a.figure=='all' else [int(a.figure)])
 if a.action in ['baseline','transfer']:child('formal_analysis/run.py',a.action)
 if a.action=='uncertainty':child('code/uncertainty.py','--output',out/'independent_uncertainty','--bootstrap','10000')
if __name__=='__main__':main()
