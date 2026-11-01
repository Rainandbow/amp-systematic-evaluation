"""Retrospective fixed-budget evaluation of frozen AMP outputs. No model fitting."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math, sys, os
import numpy as np

MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']
BUDGETS=[.10,.20,.30,.50]
COHORTS=['ApexGO_template_based','ApexGO_template_free','HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection']
SEED=2026102401
def read(p):
 op=gzip.open if str(p).endswith('.gz') else open
 with op(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def yes(x):return str(x).lower() in ['true','1']
def div(a,b):
 a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
 return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)
def scalar(v):return float(np.asarray(v).ravel()[0])
def interval(v,B,G):
 v=np.asarray(v);ok=v[np.isfinite(v)];frac=len(ok)/B
 if G<5:return np.nan,np.nan,'fewer_than_five_blocks',frac
 if frac<.95:return np.nan,np.nan,'insufficient_valid_replicates',frac
 if not len(ok):return np.nan,np.nan,'undefined',frac
 if np.ptp(ok)<1e-12:return np.nan,np.nan,'degenerate',frac
 lo,hi=np.quantile(ok,[.025,.975]);return float(lo),float(hi),'pointwise_percentile_exploratory',frac

def selected(W,score,budget,eligible=None):
 """Weighted top-K, including expected random selection within a tied score tier."""
 W=np.atleast_2d(W).astype(float);score=np.asarray(score,float)
 eligible=np.ones(len(score),bool) if eligible is None else np.asarray(eligible,bool)
 capacity=np.ceil(budget*W.sum(axis=1)-1e-10)
 out=np.zeros_like(W)
 if not eligible.any():return out,capacity
 ix=np.flatnonzero(eligible);order=ix[np.argsort(-score[ix],kind='stable')]
 starts=np.r_[0,np.flatnonzero(np.diff(score[order])!=0)+1]
 mass=np.add.reduceat(W[:,order],starts,axis=1)
 before=np.cumsum(mass,axis=1)-mass
 fraction=np.clip(np.divide(capacity[:,None]-before,mass,out=np.zeros_like(mass),where=mass>0),0,1)
 tiers=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(order)]))
 out[:,order]=W[:,order]*fraction[:,tiers]
 assert np.all(out.sum(axis=1)<=capacity+1e-7)
 return out,capacity

def budget_metrics(W,S,y,K):
 y=np.asarray(y,int);N=W.sum(axis=1);P=W[:,y==1].sum(axis=1);C=S.sum(axis=1);H=S[:,y==1].sum(axis=1)
 prevalence=div(P,N);random_H=C*prevalence
 return {'budget_slots':K,'selected_n':C,'unfilled_slots':K-C,'positive_hits':H,
         'precision_at_budget':div(H,C),'positive_recovery':div(H,P),
         'enrichment':div(div(H,C),prevalence),'random_expected_hits':random_H,
         'excess_hits_vs_random':H-random_H,'precision_gain_vs_random':div(H,C)-prevalence,
         'recovery_gain_vs_random':div(H,P)-div(C,N),'selected_fraction':div(C,N)}

def diagnostics(W,y,pred,score=None):
 W=np.atleast_2d(W);y=np.asarray(y,int);pred=np.asarray(pred,int)
 P=W[:,y==1].sum(axis=1);N=W[:,y==0].sum(axis=1);n=P+N
 TP=W[:,(y==1)&(pred==1)].sum(axis=1);FP=W[:,(y==0)&(pred==1)].sum(axis=1)
 FN=P-TP;TN=N-FP;den=np.sqrt((TP+FP)*P*N*(TN+FN))
 mcc=div(TP*TN-FP*FN,den);both=(P>0)&(N>0);mcc[both&(den==0)]=0;mcc[~both]=np.nan
 d={'n':n,'positive':P,'negative':N,'TP':TP,'FP':FP,'TN':TN,'FN':FN,
    'native_positive_call_rate':div(TP+FP,n),'native_recall':div(TP,P),
    'assay_negative_positive_call_rate':div(FP,N),'native_MCC':mcc,
    'native_balanced_accuracy':.5*(div(TP,P)+div(TN,N)),'prevalence':div(P,n)}
 if score is not None:
  score=np.asarray(score,float);order=np.argsort(-score,kind='stable')
  starts=np.r_[0,np.flatnonzero(np.diff(score[order])!=0)+1]
  pm=np.add.reduceat(W[:,order]*(y[order]==1)[None,:],starts,axis=1)
  nm=np.add.reduceat(W[:,order]*(y[order]==0)[None,:],starts,axis=1)
  cumP=np.cumsum(pm,axis=1);cumN=np.cumsum(nm,axis=1)
  # Zero-mass tiers can produce 0/0; their AP contribution is exactly zero.
  term=div(cumP,cumP+cumN)*pm;term=np.where(pm==0,0,term)
  ap=div(term.sum(axis=1),P)
  auc=div((nm*(cumP-.5*pm)).sum(axis=1),P*N)
  d.update(AP=ap,AUROC=auc,AP_minus_prevalence=ap-d['prevalence'])
 return d

def score_for(row,model):
 if not yes(row['supported']):return None
 if model=='c_AMPs':
  vals=[float(row[k]) for k in ['lstm','attention','bert']]
  assert all(np.isfinite(x) and 0<=x<=1 for x in vals)
  assert int(row['prediction'])==int(all(x>.5 for x in vals))
  return min(vals)
 v=float(row['score']);assert np.isfinite(v) and 0<=v<=1
 return v

def load(root):
 paths={k:root/v for k,v in {
  'candidates':'inputs/new_panel_candidates.tsv','assays':'inputs/new_panel_assays.tsv',
  'groups':'results/panel_screen/sequence_groups.tsv','screen':'results/panel_screen/inventory_screen.tsv',
  'screen_summary':'results/panel_screen/summary.json',
  'predictions':'results/panel_inference/predictions_long.tsv','complete':'results/panel_inference/complete.json'}.items()}
 missing=[str(p) for p in paths.values() if not p.is_file()]
 if missing:raise RuntimeError('Missing server inputs:\n'+'\n'.join(missing))
 stamp=json.loads(paths['complete'].read_text(encoding='utf8'))
 if stamp.get('software_test_fixture'):
  assert os.environ.get('AMP_ALLOW_TEST_FIXTURE')=='1','Synthetic fixture is allowed only in an explicit software test'
 assert stamp['new_panel_inference_completed'] is True,'Inference completion not verified'
 assert sha(paths['predictions'])==stamp['predictions_sha256'],'Prediction hash mismatch'
 summary=json.loads(paths['screen_summary'].read_text(encoding='utf8'))
 assert summary['five_inventory_pass']==35 and summary['resampling_blocks']==36,'Unexpected screen version; reconcile source first'
 candidates=read(paths['candidates']);assays=read(paths['assays']);groups=read(paths['groups']);screen=read(paths['screen']);pr=read(paths['predictions'])
 assert len(candidates)==169 and len(pr)==1014
 assert len({x['sequence_id']for x in candidates})==169
 gids={x['sequence_id']:x['resampling_block'] for x in groups};sm={x['sequence_id']:yes(x['five_inventory_pass']) for x in screen}
 pred={(x['sequence_id'],x['model']):x for x in pr};assert len(pred)==1014
 for c in candidates:
  sid=c['sequence_id'];assert sid=='s_'+hashlib.sha256(c['sequence'].encode('ascii')).hexdigest()
  assert sid in gids and sid in sm
  for m in MODELS:
   r=pred[sid,m];assert r['sequence']==c['sequence'] and int(r['prediction']) in (0,1)
   assert yes(r['supported']),'New panel must retain verified full six-model support'
   score_for(r,m)
 assert sum(yes(c['primary_candidate']) for c in candidates)==153
 return candidates,assays,gids,sm,pred,paths

def make_tasks(candidates,assays):
 tasks=[];byrec={c['record_id']:c for c in candidates}
 for cohort in COHORTS:
  rr=[c for c in candidates if c['cohort']==cohort and yes(c['primary_candidate'])]
  assert all(c['panel_label_any_tested_strain'] in ('0','1')for c in rr)
  tasks.append({'cohort':cohort,'task':'any_eligible_measured_strain','role':'primary_source_native',
                'rows':[(c,int(c['panel_label_any_tested_strain']))for c in rr]})
  # Source-native strain-specific endpoints; missing or ambiguous assays are not negatives.
  aa=[a for a in assays if a['cohort']==cohort and yes(a['primary_assay']) and a['assay_label'] in ('0','1') and yes(byrec[a['record_id']]['primary_candidate'])]
  for key in sorted({(a['strain'],a['unit'],a['threshold'])for a in aa}):
   lab={}
   for a in aa:
    if (a['strain'],a['unit'],a['threshold'])!=key:continue
    rid=a['record_id'];v=int(a['assay_label'])
    if rid in lab:assert lab[rid]==v,'Conflicting repeated assay labels; resolve before analysis'
    lab[rid]=v
   tasks.append({'cohort':cohort,'task':'strain|'+'|'.join(key),'role':'source_native_strain_sensitivity',
                 'rows':[(byrec[rid],lab[rid])for rid in sorted(lab)]})
 return tasks

def evaluate(args):
 root=Path(args.root).resolve();out=Path(args.output).resolve()
 if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be a new or empty directory; original results are preserved')
 out.mkdir(parents=True,exist_ok=True)
 candidates,assays,gids,sm,pred,paths=load(root)
 tasks=make_tasks(candidates,assays);points=[];cis=[];diag=[];dcis=[];roster=[];avail=[];loo=[];strat=[];controls=[];paired=[]
 B=args.bootstrap
 for ti,task in enumerate(tasks):
  for sub in ['all_generated','five_inventory_pass']:
   rr=[(c,y) for c,y in task['rows'] if sub=='all_generated' or sm[c['sequence_id']]]
   key={'cohort':task['cohort'],'task':task['task'],'endpoint_role':task['role'],'subset':sub}
   if not rr:avail.append(dict(**key,n=0,positive=0,negative=0,blocks=0,status='empty'));continue
   yy=np.array([y for c,y in rr]);cc=[c for c,y in rr];N=len(rr);P=int(yy.sum())
   gg,gi=np.unique([gids[c['sequence_id']]for c in cc],return_inverse=True);G=len(gg)
   avail.append(dict(**key,n=N,positive=P,negative=N-P,blocks=G,status='no_positive' if P==0 else ('positive_only' if P==N else 'both_classes')))
   for c,y in rr:
    roster.append(dict(**key,sequence_id=c['sequence_id'],record_id=c['record_id'],label=y,block=gids[c['sequence_id']],length=c['length'],family=c['family']))
   # One shared draw matrix per endpoint/subset gives paired comparisons across all models and policies.
   rng=np.random.default_rng(SEED+ti*100+int(sub!='all_generated'))
   boot_W=None if G<5 else rng.multinomial(G,np.ones(G)/G,size=B)[:,gi].astype(float)
   contrast_draws={};contrast_points={}
   for model in MODELS:
    sc=np.array([score_for(pred[c['sequence_id'],model],model)for c in cc]);de=np.array([int(pred[c['sequence_id'],model]['prediction'])for c in cc])
    status='exploratory_min_branch_extension' if model=='c_AMPs' else 'saved_AMP_class_score'
    mk=dict(**key,model=model,score_definition=status,n=N,positive=P,negative=N-P,blocks=G)
    d=diagnostics(np.ones((1,N)),yy,de,sc)
    # n/P/N already in key, do not silently overwrite their meanings.
    diag.append(dict(**mk,**{k:scalar(v)for k,v in d.items()if k not in ['n','positive','negative']},
                     score_unique=len(np.unique(sc)),score_min=float(sc.min()),score_max=float(sc.max()),
                     positive_score_median=float(np.median(sc[yy==1]))if P else np.nan,
                     negative_score_median=float(np.median(sc[yy==0]))if N>P else np.nan))
    if boot_W is not None:
     bd=diagnostics(boot_W,yy,de,sc)
     for met in ['native_recall','assay_negative_positive_call_rate','native_MCC','native_balanced_accuracy','AP','AUROC','AP_minus_prevalence']:
      lo,hi,st,v=interval(bd[met],B,G);dcis.append(dict(**key,model=model,metric=met,ci_low=lo,ci_high=hi,status=st,valid_fraction=v,bootstrap=B))
    else:
     for met in ['native_recall','assay_negative_positive_call_rate','native_MCC','native_balanced_accuracy','AP','AUROC','AP_minus_prevalence']:
      dcis.append(dict(**key,model=model,metric=met,ci_low=np.nan,ci_high=np.nan,status='fewer_than_five_blocks',valid_fraction=np.nan,bootstrap=0))
    for budget in BUDGETS:
     for policy in ['score_topK','native_positive_then_score']:
      elig=None if policy=='score_topK' else de==1
      W=np.ones((1,N));S,K=selected(W,sc,budget,elig);d=budget_metrics(W,S,yy,K)
      pk=dict(**mk,budget=budget,policy=policy,primary_budget=budget==.2)
      points.append(dict(**pk,**{k:scalar(v)for k,v in d.items()}))
      if budget==.2 and policy=='score_topK':contrast_points[model]=d
      if boot_W is not None:
       S,K=selected(boot_W,sc,budget,elig);bd=budget_metrics(boot_W,S,yy,K)
       if budget==.2 and policy=='score_topK':contrast_draws[model]=bd
       for met,vv in bd.items():
        lo,hi,st,v=interval(vv,B,G);cis.append(dict(**key,model=model,budget=budget,policy=policy,metric=met,ci_low=lo,ci_high=hi,status=st,valid_fraction=v,bootstrap=B))
      else:
       for met in d:cis.append(dict(**key,model=model,budget=budget,policy=policy,metric=met,ci_low=np.nan,ci_high=np.nan,status='fewer_than_five_blocks',valid_fraction=np.nan,bootstrap=0))
    if task['role']=='primary_source_native' and sub=='all_generated':
     for c,y,s,dd in zip(cc,yy,sc,de):
      controls.append(dict(**key,model=model,sequence_id=c['sequence_id'],label=int(y),score=float(s),prediction=int(dd),length=int(c['length']),block=gids[c['sequence_id']],inventory_pass=sm[c['sequence_id']],score_definition=status))
     # Leave-one-block-out changes membership; it is a sensitivity check, not an independent validation fold.
     for group in gg:
      ix=np.array([gids[c['sequence_id']]!=group for c in cc]);w=np.ones((1,int(ix.sum())))
      if not ix.any():continue
      S,K=selected(w,sc[ix],.2);bb=budget_metrics(w,S,yy[ix],K);dd=diagnostics(w,yy[ix],de[ix],sc[ix])
      loo.append(dict(**mk,omitted_block=group,remaining_n=int(ix.sum()),remaining_positive=int(yy[ix].sum()),
                      precision_gain_vs_random=scalar(bb['precision_gain_vs_random']),AUROC=scalar(dd['AUROC']),native_MCC=scalar(dd['native_MCC'])))
     # Prespecified descriptive strata only; no multivariable causal or mechanism claims.
     lengths=np.array([int(c['length'])for c in cc]);passes=np.array([sm[c['sequence_id']]for c in cc])
     strata=[('length','10-19',lengths<20),('length','20-29',(lengths>=20)&(lengths<30)),('length','30+',lengths>=30),
             ('inventory','pass',passes),('inventory','not_pass',~passes)]
     for axis,level,ix in strata:
      if not ix.any():continue
      d=diagnostics(np.ones((1,int(ix.sum()))),yy[ix],de[ix],sc[ix])
      strat.append(dict(**key,model=model,stratum_axis=axis,stratum=level,blocks=len({gids[c['sequence_id']]for c,take in zip(cc,ix)if take}),**{k:scalar(v)for k,v in d.items()},interpretation='descriptive membership-specific association; not a causal mechanism'))
   if task['role']=='primary_source_native':
    from itertools import combinations
    for m1,m2 in combinations(MODELS,2):
     for met in ['precision_at_budget','positive_recovery','excess_hits_vs_random']:
      delta=scalar(contrast_points[m1][met]-contrast_points[m2][met])
      lo,hi,st,v=(np.nan,np.nan,'fewer_than_five_blocks',np.nan) if boot_W is None else interval(contrast_draws[m1][met]-contrast_draws[m2][met],B,G)
      paired.append(dict(**key,model_1=m1,model_2=m2,budget=.2,policy='score_topK',metric=met,difference=delta,ci_low=lo,ci_high=hi,status=st,valid_fraction=v,exploratory_camps_extension='c_AMPs' in [m1,m2]))
   print('ANALYSED',task['cohort'],task['task'],sub,'n',N,'P',P,'blocks',G,flush=True)
 write(out/'budget_points.tsv',points);write(out/'budget_intervals.tsv',cis);write(out/'task_diagnostics.tsv',diag);write(out/'diagnostic_intervals.tsv',dcis)
 write(out/'task_membership.tsv',roster);write(out/'task_availability.tsv',avail);write(out/'sequence_score_diagnostics.tsv',controls)
 write(out/'leave_one_block_out.tsv',loo);write(out/'descriptive_strata.tsv',strat)
 write(out/'paired_primary_budget_differences.tsv',paired)
 historical_paths=[root/'inputs/historical_scores.tsv.gz',root/'inputs/frozen_members_and_groups.tsv.gz']
 if all(p.is_file()for p in historical_paths):
  raw=read(historical_paths[0]);members=read(historical_paths[1]);ix={(r['dataset'],r['sequence_id']):r for r in raw};hp=[]
  assert len(members)==29104 and len(ix)==41956
  for ds in dict.fromkeys(r['dataset']for r in members):
   rr=[]
   for member in members:
    if member['dataset']!=ds:continue
    r=ix[ds,member['sequence_id']];assert r['sequence']==member['sequence'] and r['label']==member['label']
    for model in MODELS:assert r[model+'__prediction']==member[model+'__prediction']
    rr.append(r)
   lens=np.array([int(r['length'])for r in rr]);y=np.array([int(r['label'])for r in rr])
   for model in MODELS:
    field='c_AMPs__extension_min_score' if model=='c_AMPs' else model+'__score'
    scores=np.array([float(r[field])for r in rr]);dec=np.array([int(r[model+'__prediction'])for r in rr])
    for label,mask in [('all',np.ones(len(rr),bool)),('10-29',(lens>=10)&(lens<30)),('30-59',(lens>=30)&(lens<60)),('60-99',(lens>=60)&(lens<100)),('100-126',lens>=100)]:
     if not mask.any():continue
     dd=diagnostics(np.ones((1,int(mask.sum()))),y[mask],dec[mask],scores[mask])
     hp.append(dict(dataset=ds,model=model,length_stratum=label,**{k:scalar(v)for k,v in dd.items()},endpoint='historical_source_AMP_label',interval_status='point_only_diagnostic; see original formal intervals',score_definition='supplementary_min_branch'if model=='c_AMPs'else 'saved_AMP_score'))
  write(out/'historical_task_diagnostics.tsv',hp)
  paths.update({p.name:p for p in historical_paths})
 manifest={k:{'path':str(p),'sha256':sha(p)}for k,p in paths.items()}
 settings={'version':'budget_task_v1','bootstrap':B,'smoke_only':B<10000,'synthetic_fixture':json.loads(paths['complete'].read_text(encoding='utf8')).get('software_test_fixture',False),'seed':SEED,'budgets':BUDGETS,'primary_budget':.2,
  'selection':'descending p, ceiling(b*n), label-blind fractional weights for tied score tiers; c_AMPs minimum-branch extension reported separately',
  'random_reference':'exact expectation conditional on the same endpoint, realized selected size and members; not a randomization-test p value',
  'bootstrap':'uniform block resampling with replacement, record multiplicities retained, budgets reselected, paired model and policy draws',
  'task_boundary':'source-native AMP classification differs from source-condition measured activity; no universal inactivity labels',
  'prior_inspection':'retrospective exploratory analysis specified after seeing existing results, not preregistered or a new blinded test',
  'no_new_inference':True,'no_model_fitting':True,'no_wet_lab':True,'no_unit_conversion':True,'no_pooled_cross_study_MCC':True,
  'intervals':'pointwise percentile; minimum five blocks and >=95% valid replicates; degenerate intervals omitted; no multiplicity adjustment',
  'cautions':['all-positive endpoints cannot establish activity discrimination','inventory pass does not establish full training independence','small measured negative class limits inference','no automatic proof of cause or mechanism'],
  'python_version':sys.version,'numpy_version':np.__version__,
  'input_manifest':manifest,'output_files':{p.name:sha(p) for p in out.glob('*.tsv')}}
 (out/'run_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf8')
 primary=[r for r in points if r['endpoint_role']=='primary_source_native' and r['subset']=='all_generated' and r['budget']==.2 and r['policy']=='score_topK']
 write(out/'primary_budget20_summary.tsv',primary)
 (out/'INTERPRETATION_CN.txt').write_text('固定预算结果只适用于已发表且实际测试的候选池。原生AMP分类和特定实验条件下的MIC阈值标签不是同一任务。优先查看primary_budget20_summary.tsv及其配对区间；再看task_diagnostics.tsv和逐菌株敏感性。不能根据点估计或区间重叠自动宣称原因或机制。全阳性组只支持阳性找回描述，富集倍数为1不表示模型已经得到有效验证。c_AMPs最低分支分数属于单列的探索性扩展。',encoding='utf8')
 print('ANALYSIS COMPLETE',out,flush=True)

def check(args):
 import importlib
 for mod in ['numpy','matplotlib','pymupdf']:
  m=importlib.import_module(mod);print(mod,getattr(m,'__version__','available'))
 _,_,_,_,_,paths=load(Path(args.root).resolve())
 print('REAL INPUTS PASS',len(paths),'files; all 169 sequences, 1014 predictions and screen version checked')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','run']);ap.add_argument('--root',required=True);ap.add_argument('--output');ap.add_argument('--bootstrap',type=int,default=10000)
 a=ap.parse_args();assert a.bootstrap>0
 if a.action=='check':check(a)
 else:
  assert a.output,'--output is required';evaluate(a)
