"""Recalculate every main-figure point layer from frozen records and decisions.
Historical interval endpoints are verified as archived inputs, not recomputed here.
"""
from pathlib import Path
import csv,gzip,json,math,sys
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']
SOURCES=['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']

def read(p):
 with (gzip.open(p,'rt',encoding='utf-8-sig',newline='') if str(p).endswith('.gz') else p.open(encoding='utf-8-sig',newline='')) as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def mcc(tp,fn,fp,tn):
 if tp+fn==0 or tn+fp==0:return float('nan')
 d=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
 return (tp*tn-fp*fn)/math.sqrt(d) if d else 0.
def close(a,b):
 a=float(a) if str(a) else float('nan');b=float(b) if str(b) else float('nan')
 assert math.isnan(a) and math.isnan(b) or math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),(a,b)
def auc(y,s):
 pos=np.array([x for a,x in zip(y,s) if a==1]);neg=np.sort([x for a,x in zip(y,s) if a==0])
 if not len(pos) or not len(neg):return float('nan')
 return float((np.searchsorted(neg,pos,side='left')+.5*(np.searchsorted(neg,pos,side='right')-np.searchsorted(neg,pos,side='left'))).sum()/(len(pos)*len(neg)))
def run(out):
 core=json.loads((ROOT/'inputs/historical/frozen_core.json').read_text(encoding='utf8'))
 members=read(ROOT/'formal_analysis/analysis/inputs/historical_members.tsv.gz')
 assert len(members)==29104 and len({r['sequence_sha256'] for r in members})==28481
 membership=defaultdict(set)
 for r in members:membership[r['sequence_sha256']].add(r['dataset'])
 patterns=Counter(tuple(int(s in v) for s in SOURCES) for v in membership.values())
 saved=read(ROOT/'inputs/historical/exclusive_membership_patterns.tsv')
 assert len(patterns)==len(saved)==28
 for r in saved:assert patterns[tuple(int(r[s]) for s in SOURCES)]==int(r['exclusive_n'])
 assert sum(v for k,v in patterns.items() if sum(k)>1)==605
 write(out/'Figure1_exact_membership.tsv',[dict(zip(SOURCES,k),exclusive_n=v) for k,v in sorted(patterns.items())])
 overall=[];strata=[]
 intervals=read(ROOT/'inputs/historical/length_mcc_intervals.tsv')
 for source in SOURCES:
  rr=[r for r in members if r['dataset']==source]
  c=next(c for c in core['cohorts'] if c['dataset']==source)
  assert len(rr)==int(c['common_n']) and sum(int(r['label']) for r in rr)==int(c['common_positive'])
  for binname,low,high in [('all',0,100000),('10_29',10,29),('30_59',30,59),('60_99',60,99),('100_126',100,126)]:
   rs=[r for r in rr if low<=int(r['length'])<=high]
   for model in MODELS:
    cm=Counter((int(r['label']),int(r[model+'__prediction'])) for r in rs)
    tp,fn,fp,tn=[cm[k] for k in [(1,1),(1,0),(0,1),(0,0)]]
    v=mcc(tp,fn,fp,tn)
    row={'dataset':source,'model':model,'stratum':binname,'n':len(rs),'positive':tp+fn,'negative':fp+tn,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'mcc':v}
    if binname=='all':
     ref=next(r for r in core['mcc'] if r['dataset']==source and r['tool']==model);close(v,ref['estimate']);overall.append(row)
    else:
     ref=next(r for r in intervals if r['scope']=='global_primary' and r['dataset']==source and r['model']==model and r['length_bin']==binname)
     if len(rs):close(v,ref['mcc'])
     strata.append(row)
 write(out/'Figure2_recomputed_MCC.tsv',overall);write(out/'Figure3_recomputed_length_MCC.tsv',strata)
 # Original evidence table retains unsupported records and source cell locations.
 original=read(ROOT/'inputs/original_panel/evidence_prediction_joined.tsv')
 counts=[]
 for model in MODELS:
  pos=[r for r in original if r['tool']==model and r['paper_rule_positive']=='1']
  assert len(pos)==84
  sup=[r for r in pos if r['supported']=='1'];det=sum(r['prediction']=='1' for r in sup)
  ref=[r for r in core['external'] if r['tool']==model]
  assert sum(int(r.get('supported_positive_n',r.get('positive_supported_n','0'))) for r in ref)==len(sup)
  assert sum(int(r.get('detected_positive_n',r.get('predicted_positive_n','0'))) for r in ref)==det
  counts.append({'model':model,'eligible':84,'supported':len(sup),'detected':det,'supported_fraction':len(sup)/84,'end_to_end_detection':det/84})
 write(out/'Figure6_original84_recomputed.tsv',counts)
 # Recalculate experimental budget and family points using the archived implementation.
 sys.path.insert(0,str(ROOT/'formal_analysis/analysis/code'))
 import common as C
 import baseline_family as B
 panel,si,ti=C.load_panel();savedpoints=read(ROOT/'figure4/inputs/baseline_and_model_points.tsv')
 fresh=[]
 for cohort in C.COHORTS:
  cc=[r for r in panel if r['cohort']==cohort];families=sorted({r['family'] for r in cc})
  W=np.ones((1,len(cc)))
  for task in ['any_eligible_measured_strain','source_reference_Ecoli']:
   y=np.array([int(ti[cohort,task,r['record_id']]['label']) for r in cc])
   for method in MODELS+C.BASELINES:
    sc=np.array([float(si[r['sequence_id'],method]['score']) if method in MODELS else C.descriptors(r['sequence'])[method] for r in cc])
    S,K=C.selected(W,sc,.2);vals=C.budget_metrics(W,S,y);vals['pooled_AUROC']=C.auc(W,y,sc)
    fa=[]
    for family in families:
     ix=np.array([r['family']==family for r in cc])
     if len(np.unique(y[ix]))==2:fa.append(float(C.auc(np.ones((1,int(ix.sum()))),y[ix],sc[ix])[0]))
    vals['macro_within_family_AUROC']=np.array([np.mean(fa) if fa else np.nan])
    ref=next(r for r in savedpoints if r['cohort']==cohort and r['task']==task and r['method']==method)
    row={'cohort':cohort,'task':task,'method':method,'n':len(cc),'positive':int(y.sum()),'budget_slots':float(K[0])}
    for metric,arr in vals.items():row[metric]=float(arr[0]);close(row[metric],ref[metric])
    fresh.append(row)
 assert len(fresh)==96
 write(out/'Figure4_recomputed_all_96_method_task_points.tsv',fresh)
 pd=[]
 for r in fresh:
  if r['method'] not in MODELS:continue
  b=next(b for b in fresh if b['cohort']==r['cohort'] and b['task']==r['task'] and b['method']=='length_low')
  pd.append({'cohort':r['cohort'],'task':r['task'],'model':r['method'],'comparator':'length_low','hit_difference':r['positive_hits']-b['positive_hits']})
 write(out/'Figure4_recomputed_paired_48_points.tsv',pd)
 score=read(ROOT/'formal_analysis/analysis/inputs/scores151.tsv')
 labels=read(ROOT/'formal_analysis/analysis/inputs/task_labels.tsv')
 labelmap={(r['sequence_id'],r['task']):int(r['label']) for r in labels if r['cohort']=='ApexGO_template_based' and r['threshold']=='64'}
 dg=read(ROOT/'inputs/fig6/task_diagnostics.tsv');tb=read(ROOT/'inputs/fig6/task_budget.tsv')
 diag=[];budgets=[]
 for model in MODELS:
  rs=[r for r in score if r['cohort']=='ApexGO_template_based' and r['model']==model]
  assert len(rs)==100
  y=[int(r['label']) for r in rs];pred=[int(r['prediction']) for r in rs];s=[float(r['score']) for r in rs]
  tp=sum(a==b==1 for a,b in zip(y,pred));fp=sum(a==0 and b==1 for a,b in zip(y,pred))
  ref=next(r for r in dg if r['cohort']=='ApexGO_template_based' and r['subset']=='all_generated' and r['task']=='any_eligible_measured_strain' and r['model']==model)
  close(tp/86,ref['native_recall']);close(fp/14,ref['assay_negative_positive_call_rate']);close(auc(y,s),ref['AUROC'])
  diag.append({'model':model,'n':100,'P':86,'N':14,'TP':tp,'FP':fp,'recall':tp/86,'threshold_negative_call_fraction':fp/14,'AUROC':auc(y,s)})
  cutoff=sorted(s,reverse=True)[19];above=sum(x>cutoff for x in s);ties=sum(x==cutoff for x in s)
  weights=[1. if x>cutoff else (20-above)/ties if x==cutoff else 0. for x in s]
  for task in ['any_eligible_measured_strain','source_reference_Ecoli']:
   yy=[labelmap[r['sequence_id'],task] for r in rs];hits=sum(w*a for w,a in zip(weights,yy))
   ref=next(r for r in tb if r['cohort']=='ApexGO_template_based' and r['model']==model and r['subset']=='all_active_design' and r['task']==task and r['threshold']=='64' and r['policy']=='score_topK' and r['budget']=='0.2')
   close(hits/20,ref['precision_at_budget'])
   budgets.append({'model':model,'task':task,'K':20,'hits':hits,'precision':hits/20,'positive':sum(yy)})
 write(out/'Figure6_recomputed_diagnostics.tsv',diag);write(out/'Figure6_recomputed_budget.tsv',budgets)
 # Figure5 is re-aggregated from 1,800 frozen selected rules and 5,400 evaluations.
 rules=read(ROOT/'formal_analysis/figures/inputs/honest_transfer/selected_rules.tsv')
 ev=read(ROOT/'formal_analysis/figures/inputs/honest_transfer/evaluation_points.tsv')
 assert len(rules)==1800 and len(ev)==5400
 availability=[]
 for policy,wanted in [('mixture_original',458),('mixture_source_support',312),('per_source_supported',201)]:
  assert sum(r['policy']==policy and r['policy_status']=='feasible' for r in rules)==wanted
  for model in MODELS[:5]:availability.append({'policy':policy,'model':model,'feasible':sum(r['policy']==policy and r['model']==model and r['policy_status']=='feasible' for r in rules),'planned':120})
 write(out/'Figure5_recomputed_availability.tsv',availability)
 lu={(r['held_source'],r['model'],r['repeat'],r['policy']):r for r in ev if r['phase']=='held_source'}
 pairs=[]
 for (h,m,k,p),b in lu.items():
  if p!='per_source_supported' or b['policy_status']!='feasible':continue
  a=lu[h,m,k,'mixture_source_support'];assert a['policy_status']=='feasible'
  r={'held_source':h,'model':m,'repeat':k}
  for met in ['conditional_error_balanced','coverage_balanced','positive_discarded_fraction_balanced']:r[met+'_difference_pp']=100*(float(b[met])-float(a[met]))
  pairs.append(r)
 assert len(pairs)==201
 med=[float(np.median([r[x+'_difference_pp'] for r in pairs])) for x in ['conditional_error_balanced','coverage_balanced','positive_discarded_fraction_balanced']]
 assert [round(x,2) for x in med]==[-2.90,-25.22,28.28]
 write(out/'Figure5_recomputed_paired_201.tsv',pairs)
 receipt={'status':'PASS','historical_records':len(members),'unique_sequences':len(membership),'Figure2_points':len(overall),'Figure3_planned_points':len(strata),'Figure5_rules':len(rules),'Figure5_pairs':len(pairs),'Figure6_original_positive_denominator':84,'new_inference':False,'bootstrap_rerun':False,'historical_intervals':'Archived interval endpoints reused with hashes; this point audit does not rederive their sampling distributions.'}
 (out/'POINTS_COMPLETE.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
 print(json.dumps(receipt),flush=True)

if __name__=='__main__':run(Path(sys.argv[1]).resolve())
