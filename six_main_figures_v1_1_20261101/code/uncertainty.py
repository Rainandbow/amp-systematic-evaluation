"""Independent grouped-bootstrap verification from frozen predictions.
Fresh fixed seed; outputs are separate and NEVER replace archived figure intervals.
"""
from pathlib import Path
import sys,json,math,argparse
import numpy as np
from points import read,write,MODELS,SOURCES,ROOT
sys.path.insert(0,str(ROOT/'formal_analysis/analysis/code'))
import common as C

def interval(values,B,G,need5=False):
 v=np.array(values,float);v=v[np.isfinite(v)];f=len(v)/B
 if need5 and G<5:return np.nan,np.nan,'fewer_than_five_contributing_blocks',f
 if f<.95:return np.nan,np.nan,'fewer_than_95_percent_valid',f
 if not len(v) or np.ptp(v)==0:return np.nan,np.nan,'degenerate',f
 lo,hi=np.percentile(v,[2.5,97.5]);return float(lo),float(hi),'independent_pointwise_percentile',f
def vector_mcc(cm):
 tp,fn,fp,tn=cm[...,0],cm[...,1],cm[...,2],cm[...,3]
 d=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
 with np.errstate(divide='ignore',invalid='ignore'):v=np.where(d>0,(tp*tn-fp*fn)/np.sqrt(d),0.)
 return np.where((tp+fn>0)&(tn+fp>0),v,np.nan)
def run(out,B):
 if B<10000:raise ValueError('Use at least 10,000 draws; this command has no publication-ready smoke mode')
 out.mkdir(parents=True,exist_ok=False);members=read(ROOT/'formal_analysis/analysis/inputs/historical_members.tsv.gz')
 rows=[];seed=2026110101
 bins=[('all',0,100000),('10_29',10,29),('30_59',30,59),('60_99',60,99),('100_126',100,126)]
 for si,source in enumerate(SOURCES):
  rr=[r for r in members if r['dataset']==source];groups,gi=np.unique([r['global_group'] for r in rr],return_inverse=True);G=len(groups)
  A=np.zeros((G,5,6,4))
  for g,r in zip(gi,rr):
   y=int(r['label']);length=int(r['length'])
   for j,(_,lo,hi) in enumerate(bins):
    if not lo<=length<=hi:continue
    for m,model in enumerate(MODELS):
     p=int(r[model+'__prediction']);idx={(1,1):0,(1,0):1,(0,1):2,(0,0):3}[y,p];A[g,j,m,idx]+=1
  flat=A.reshape(G,-1);rng=np.random.default_rng(seed+si*100)
  values=[]
  for start in range(0,B,250):
   D=rng.multinomial(G,np.ones(G)/G,size=min(250,B-start));cm=(D@flat).reshape(-1,5,6,4);values.append(vector_mcc(cm))
  draws=np.concatenate(values);point=vector_mcc(A.sum(axis=0))
  for j,(name,_,_) in enumerate(bins):
   for m,model in enumerate(MODELS):
    lo,hi,status,frac=interval(draws[:,j,m],B,G)
    rows.append(dict(figure='2' if j==0 else '3',dataset=source,stratum=name,model=model,estimate=float(point[j,m]),ci_low=lo,ci_high=hi,status=status,valid_fraction=frac,bootstrap=B,global_groups=G,seed=seed+si*100))
  print('HISTORICAL UNCERTAINTY',source,G,B,flush=True)
 write(out/'Figures2_3_independent_intervals.tsv',rows)
 panel,scores,tasks=C.load_panel();cc=[r for r in panel if r['cohort']=='ApexGO_template_based']
 groups,gi=np.unique([r['block'] for r in cc],return_inverse=True);G=len(groups)
 rng=np.random.default_rng(seed+1000);D=rng.multinomial(G,np.ones(G)/G,size=B);W=D[:,gi].astype(float);P=np.ones((1,len(cc)))
 rows=[]
 for task in ['any_eligible_measured_strain','source_reference_Ecoli']:
  y=np.array([int(tasks[r['cohort'],task,r['record_id']]['label']) for r in cc]);pos=W@y;neg=W@(1-y)
  for model in MODELS:
   sc=np.array([float(scores[r['sequence_id'],model]['score']) for r in cc]);pred=np.array([int(scores[r['sequence_id'],model]['prediction']) for r in cc])
   S,K=C.selected(W,sc,.2);bm=C.budget_metrics(W,S,y);PS,PK=C.selected(P,sc,.2);pm=C.budget_metrics(P,PS,y)
   bm['AUROC']=C.auc(W,y,sc);pm['AUROC']=C.auc(P,y,sc)
   with np.errstate(divide='ignore',invalid='ignore'):
    bm['native_recall']=(W@(y*pred))/pos;bm['threshold_negative_call_fraction']=(W@((1-y)*pred))/neg
   pm['native_recall']=np.array([(y*pred).sum()/y.sum()]);pm['threshold_negative_call_fraction']=np.array([((1-y)*pred).sum()/(1-y).sum()])
   for metric in ['precision','AUROC','native_recall','threshold_negative_call_fraction']:
    lo,hi,status,frac=interval(bm[metric],B,G,True)
    rows.append(dict(figure='6',cohort='ApexGO_template_based',task=task,model=model,metric=metric,estimate=float(pm[metric][0]),ci_low=lo,ci_high=hi,status=status,valid_fraction=frac,bootstrap=B,blocks=G,seed=seed+1000))
 write(out/'Figure6_independent_intervals.tsv',rows)
 (out/'COMPLETE.json').write_text(json.dumps(dict(bootstrap=B,base_seed=seed,scope='Figures2/3 MCC and Figure6 ApexGO discrimination/precision intervals; independent verification only',new_inference=False,archived_intervals_replaced=False,source_of_uncertainty='Frozen label and output block resampling; no assay, chemistry, training or grouping uncertainty'),indent=2))
 print('INDEPENDENT UNCERTAINTY PASS',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);ap.add_argument('--bootstrap',type=int,default=10000);a=ap.parse_args();run(Path(a.output).resolve(),a.bootstrap)
