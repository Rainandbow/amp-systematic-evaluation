"""New panel inference with the unchanged, checksum-verified released adapter."""
import argparse,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor_adapter'))
import core as c

def run(args):
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 plan=c.read_json(ROOT/'server_plan.json');plan['work']=str(out)
 rows=c.read_tsv(ROOT/'inputs/new_panel_queries.tsv')
 assert len(rows)==169 and len({r['sequence_id'] for r in rows})==169
 c.package_checks()
 for r in rows:
  assert r['label']=='-1' and r['sequence_id']=='s_'+hashlib.sha256(r['sequence'].encode('ascii')).hexdigest()
 settings={'version':'panel_extension_inference_v1','query_sha256':c.sha(ROOT/'inputs/new_panel_queries.tsv'),'plan_template_sha256':c.sha(ROOT/'server_plan.json'),'adapter_manifest_sha256':c.sha(ROOT/'vendor_adapter/SHA256SUMS.txt')}
 if (out/'prepared.json').exists():
  previous=c.read_json(out/'prepared.json');c.require(previous['inputs']==settings,'Resume inputs changed; use a new work directory');c.verify_files(out,previous['files']);plan=c.read_json(out/'plan.json');jobs=c.read_json(out/'jobs.json')
 else:
  c.require(not any(out.iterdir()),'Use an empty inference work directory')
  c.dump(out/'plan.json',plan);jobs=[]
  for comp in c.COMPONENTS:
   tool='c_AMPs' if comp.startswith('camps_') else comp
   selected=[r for r in rows if c.support(tool,r)];union=out/'02_union'/comp/'queries.tsv';c.write_tsv(union,selected,c.FIELDS)
   if selected:jobs.append(dict(key=comp+'__000000',component=comp,query=str(union),query_sha256=c.sha(union),n=len(selected)))
  c.dump(out/'jobs.json',jobs);c.dump(out/'protocol.json',dict(**settings,model_fit=False,threshold_tuning=False,labels_to_predictors='UNKNOWN -1',support='unchanged verified adapter rules',gpu='one explicitly allocated GPU, sequential components'))
  files={p.relative_to(out).as_posix():c.sha(p) for p in out.rglob('*') if p.is_file()}
  c.dump(out/'prepared.json',dict(inputs=settings,files=files))
 cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1));cpu_text=','.join(map(str,cpus[:min(2,len(cpus))]))
 # A scheduler-provided single device can be used without widening its mask.
 inherited=os.environ.get('CUDA_VISIBLE_DEVICES','');mask=[x.strip() for x in inherited.split(',') if x.strip()]
 gpu=os.environ.get('AMP_GPU_ID','')
 if not gpu and len(mask)==1:gpu=mask[0]
 if gpu and mask:c.require(gpu in mask,'AMP_GPU_ID must be inside inherited CUDA_VISIBLE_DEVICES; use physical index or GPU UUID, not a remapped ordinal')
 if args.action=='check':
  check={'adapter_checksums':'passed','queries':len(rows),'components':{j['component']:j['n'] for j in jobs},'gpu':gpu or 'not configured','missing_runtime_paths':[],'model_hashes_checked':False}
  for t in c.COMPONENTS:
   exe=plan['tools'][t]['executable']
   if not Path(exe).is_file():check['missing_runtime_paths'].append(exe)
  if not args.skip_model_hashes:
   for comp in c.COMPONENTS:
    identity=c.read_json(ROOT/'expected_identities'/(comp+'.json'))
    for path,h in identity.get('files',{}).items():c.require(Path(path).is_file() and c.sha(path)==h,'Missing/changed released model file '+path)
   check['model_hashes_checked']=True
  print(json.dumps(check,indent=2));
  if not args.skip_model_hashes:
   c.require(not check['missing_runtime_paths'],'Missing existing model runtimes; inspect server_plan.json')
   c.require(bool(gpu),'No explicitly allocated GPU configured for panel inference')
  return
 c.require(gpu,'Set AMP_GPU_ID to an allocated GPU or provide one CUDA_VISIBLE_DEVICES device; no allocation widened')
 for job in jobs:
  if c.completed_job(out,job):print('REUSE VERIFIED',job['key'],flush=True);continue
  comp=job['component'];expected=c.read_json(ROOT/'expected_identities'/(comp+'.json'))
  for path,h in expected.get('files',{}).items():c.require(Path(path).is_file() and c.sha(path)==h,'Missing/changed model '+path)
  attempt=out/'03_jobs'/job['key']/('attempt_'+uuid.uuid4().hex[:12]);attempt.mkdir(parents=True)
  exe=plan['audit_python'] if comp=='ampir' else plan['tools'][comp]['executable'];device=gpu if comp in ('tsAMP_I','HMD_AMP','camps_bert') else ''
  cmd=[exe,'-B','-u',str(ROOT/'vendor_adapter/worker.py'),'--plan',str(out/'plan.json'),'--key',job['key'],'--attempt',str(attempt),'--cpus',cpu_text]
  if device:cmd+=['--gpu',device]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=device or '-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  print('START',comp,job['n'],'log',attempt/'console.log',flush=True)
  with (attempt/'console.log').open('x',encoding='utf-8') as log:rc=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  c.require(rc==0,'Component failed; inspect '+str(attempt/'console.log'))
  c.require(c.read_json(attempt/'identity.json')==expected,'Runtime/weight identity differs from prior verified experiment')
  c.align(c.read_tsv(attempt/'predictions.tsv'),c.read_tsv(job['query']),comp);c.commit_job(out,job,attempt)
  print('COMMITTED',comp,flush=True)
 outputs={comp:c.collect_component(out,comp,jobs) for comp in c.COMPONENTS};result=[]
 for r in rows:
  for tool in c.TOOLS:
   d=dict(sequence_id=r['sequence_id'],sequence=r['sequence'],model=tool,supported=int(c.support(tool,r)),score='',prediction='',lstm='',attention='',bert='')
   if d['supported']:
    if tool=='c_AMPs':
     a=outputs['camps_keras'][r['sequence_id']];vals=[float(a['lstm']),float(a['attention']),float(outputs['camps_bert'][r['sequence_id']]['probability'])];d.update(lstm=vals[0],attention=vals[1],bert=vals[2],prediction=int(all(x>.5 for x in vals)))
    else:
     a=outputs[tool][r['sequence_id']];d.update(score=float(a['probability']),prediction=int(a['prediction']))
   result.append(d)
 c.write_tsv(out/'predictions_long.tsv',result,list(result[0]));c.dump(out/'complete.json',dict(settings=settings,predictions_sha256=c.sha(out/'predictions_long.tsv'),rows=len(result),queries=len(rows),unsupported=sum(not r['supported'] for r in result),new_panel_inference_completed=True))
 print('PANEL INFERENCE COMPLETE',len(result),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','run']);ap.add_argument('--output',default=str(ROOT/'results/panel_inference'));ap.add_argument('--skip-model-hashes',action='store_true',help='check mode only for local package smoke; not a server readiness pass');run(ap.parse_args())
