"""Frozen-output selective prediction analysis. No AMP fitting or score calibration."""
import argparse,csv,gzip,json,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TOOLS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP']
DATASETS=['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']
SEED=2026102301

def read(path):
 op=gzip.open if str(path).endswith('.gz') else open
 with op(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def load():
 meta=read(ROOT/'inputs/frozen_members_and_groups.tsv.gz');raw=read(ROOT/'inputs/historical_scores.tsv.gz')
 ix={(r['dataset'],r['sequence_id']):r for r in raw}
 assert len(ix)==len(raw)==41956 and len(meta)==29104
 rows=[]
 for r in meta:
  d=ix[r['dataset'],r['sequence_id']]
  assert d['sequence']==r['sequence'] and d['label']==r['label'] and len(r['sequence'])==int(r['length'])
  for t in TOOLS+['c_AMPs']:
   assert int(d[t+'__prediction'])==int(r[t+'__prediction'])
  d=dict(d);d['global_group']=r['global_group'];d['local_group']=r['local_group'];rows.append(d)
 return rows

def div(a,b):
 a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
 return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b!=0)

def selected_weights(W,q,policy):
 W=np.atleast_2d(W).astype(float,copy=False)
 if policy=='q095':return W*(q>=.95)[None,:]
 if policy!='top80':raise ValueError(policy)
 _,inv=np.unique(-q,return_inverse=True)
 order=np.argsort(-q,kind='stable');qs=q[order]
 starts=np.r_[0,np.where(qs[1:]!=qs[:-1])[0]+1]
 cnt=np.add.reduceat(W[:,order],starts,axis=1)
 before=np.cumsum(cnt,axis=1)-cnt
 target=np.ceil(.8*W.sum(axis=1))[:,None]
 frac=np.clip(np.divide(target-before,cnt,out=np.zeros_like(cnt),where=cnt>0),0,1)
 return W*frac[:,inv]

METRICS=['coverage','positive_retention','negative_retention','retained_error','error_change','conditional_recall','source_label_precision','end_to_end_detection','loss_previously_detected','positive_rejection','retained_positive_fraction','balanced_error','random_expected_detection','random_expected_error']
def metrics(W,S,y,pred):
 W=np.atleast_2d(W);S=np.atleast_2d(S)
 pos=y==1;neg=~pos;tp=pos&(pred==1);fp=neg&(pred==1);fn=pos&(pred==0);tn=neg&(pred==0);err=pred!=y
 n=W.sum(axis=1);ret=S.sum(axis=1);P=W[:,pos].sum(axis=1);N=W[:,neg].sum(axis=1)
 p=S[:,pos].sum(axis=1);nneg=S[:,neg].sum(axis=1);TP=S[:,tp].sum(axis=1);FP=S[:,fp].sum(axis=1);FN=S[:,fn].sum(axis=1)
 full_error=div(W[:,err].sum(axis=1),n);re=div(S[:,err].sum(axis=1),ret);cov=div(ret,n)
 d=dict(coverage=cov,positive_retention=div(p,P),negative_retention=div(nneg,N),retained_error=re,error_change=re-full_error,conditional_recall=div(TP,p),source_label_precision=div(TP,TP+FP),end_to_end_detection=div(TP,P),loss_previously_detected=div(W[:,tp].sum(axis=1)-TP,P),positive_rejection=1-div(p,P),retained_positive_fraction=div(p,ret),balanced_error=.5*(div(FN,p)+div(FP,nneg)),random_expected_detection=cov*div(W[:,tp].sum(axis=1),P),random_expected_error=full_error)
 return d

def interval(x,B):
 v=x[np.isfinite(x)];ratio=len(v)/B
 if ratio<.95:return float('nan'),float('nan'),'insufficient_valid',ratio
 if len(v)==0:return float('nan'),float('nan'),'undefined',ratio
 if np.ptp(v)<1e-12:return float('nan'),float('nan'),'degenerate',ratio
 lo,hi=np.quantile(v,[.025,.975]);return float(lo),float(hi),'pointwise_percentile_exploratory',ratio

def loss(args):
 rows=load();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);points=[];intervals=[];curves=[];groups=[]
 for di,ds in enumerate(DATASETS):
  rr=[r for r in rows if r['dataset']==ds];y=np.array([int(r['label']) for r in rr]);gs,gi=np.unique([r['global_group'] for r in rr],return_inverse=True);G=len(gs);n=len(rr)
  groups.append(dict(dataset=ds,n=n,groups=G,max_group_n=int(np.bincount(gi).max()),positive=int(y.sum()),negative=int(n-y.sum())))
  pars={t:(np.array([float(r[t+'__score']) for r in rr]),np.array([int(r[t+'__prediction']) for r in rr])) for t in TOOLS}
  boot={(t,pol,m):[] for t in TOOLS for pol in ['top80','q095'] for m in METRICS}
  for t,(score,pred) in pars.items():
   assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
   q=np.maximum(score,1-score);W=np.ones((1,n))
   for pol in ['top80','q095']:
    S=selected_weights(W,q,pol);d=metrics(W,S,y,pred);pr=dict(dataset=ds,model=t,policy=pol,n=n,positive=int(y.sum()),negative=int(n-y.sum()),groups=G,full_error=float(np.mean(pred!=y)),full_recall=float(np.mean(pred[y==1]==1)))
    pr.update({m:float(d[m][0]) for m in METRICS})
    pr.update(retained_n=float(S.sum()),retained_positive_n=float(S[:,y==1].sum()),retained_negative_n=float(S[:,y==0].sum()),retained_TP=float(S[:,(y==1)&(pred==1)].sum()),retained_FP=float(S[:,(y==0)&(pred==1)].sum()),retained_FN=float(S[:,(y==1)&(pred==0)].sum()),retained_TN=float(S[:,(y==0)&(pred==0)].sum()))
    points.append(pr)
   # Coverage curve uses the same tie rule for a prespecified grid, no labels in selection.
   order=np.argsort(-q,kind='stable');levels,inv=np.unique(-q,return_inverse=True);starts=np.r_[0,np.where(q[order][1:]!=q[order][:-1])[0]+1];cnt=np.diff(np.r_[starts,n]);before=np.r_[0,np.cumsum(cnt)[:-1]]
   for c in np.arange(.05,1.001,.05):
    frac=np.clip((np.ceil(c*n)-before)/cnt,0,1);S=frac[inv][None,:];d=metrics(W,S,y,pred)
    curves.append(dict(dataset=ds,model=t,target_coverage=float(c),**{m:float(d[m][0]) for m in METRICS}))
  rng=np.random.default_rng(SEED+di)
  for start in range(0,args.bootstrap,args.batch):
   b=min(args.batch,args.bootstrap-start);C=rng.multinomial(G,np.ones(G)/G,size=b);W=C[:,gi].astype(float)
   for t,(score,pred) in pars.items():
    q=np.maximum(score,1-score)
    for pol in ['top80','q095']:
     d=metrics(W,selected_weights(W,q,pol),y,pred)
     for m,v in d.items():boot[t,pol,m].append(v)
  for t in TOOLS:
   for pol in ['top80','q095']:
    for m in METRICS:
     lo,hi,status,valid=interval(np.concatenate(boot[t,pol,m]),args.bootstrap)
     intervals.append(dict(dataset=ds,model=t,policy=pol,metric=m,ci_low=lo,ci_high=hi,status=status,valid_fraction=valid,bootstrap=args.bootstrap))
  print('LOSS',ds,n,G,'bootstrap',args.bootstrap,flush=True)
 write(out/'selection_points.tsv',points);write(out/'selection_intervals.tsv',intervals);write(out/'selection_curves.tsv',curves);write(out/'group_summary.tsv',groups)
 replay=[r for r in points if r['policy']=='top80'];reduced=sum(r['error_change'] < -1e-12 for r in replay)
 check={'top80_reduced_combinations':reduced,'combinations':len(replay),'expected':29,'tsAMP_I_on_AMPlify_delta_pp':next(r['error_change']*100 for r in replay if r['model']=='tsAMP_I' and r['dataset']=='AMPlify'),'bootstrap':args.bootstrap,'seed':SEED,'smoke_only':args.bootstrap<10000}
 assert reduced==29 and abs(check['tsAMP_I_on_AMPlify_delta_pp']+9.11)<.02,check
 (out/'baseline_replay.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
 print(json.dumps(check),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--batch',type=int,default=32);ap.add_argument('--output',default=str(ROOT/'results/loss'));args=ap.parse_args();loss(args)
