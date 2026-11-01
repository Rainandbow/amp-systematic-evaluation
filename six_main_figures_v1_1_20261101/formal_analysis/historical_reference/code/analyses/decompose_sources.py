"""Decompose frozen exact-v3 leave-one-source-out policies, no new inference.
Finite-source observed quantities; no new bootstrap or uncertainty substitution.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,hashlib,importlib.util,json,math,sys
import numpy as np
ROOT=Path(r'E:\北鲳\AMP论文投稿资料汇总_20261017')
OUT=Path(__file__).resolve().parents[2]/'rebuild_analyses/source_decomposition'
OUT.mkdir(parents=True,exist_ok=True)
EXT=ROOT/'独立面板与筛选迁移_v1_20261023'
FORMAL=Path(r'E:\北鲳\output_supplement_v2_20261024_115912/source_data')
PATHS={'members':EXT/'inputs/frozen_members_and_groups.tsv.gz','scores':EXT/'inputs/historical_scores.tsv.gz','formal_policies':FORMAL/'FigS13_complete_policy_results.tsv','formal_settings':FORMAL/'FigS13_run_settings.json','formal_threshold_stability':FORMAL/'FigS13_threshold_stability.tsv','original_plot_points':ROOT/'figures_main_transfer_v3_20261024_091019/Fig5_main_display_data.tsv'}
# Bundled frozen inputs make the delivery reproducible after relocation.
if (OUT/'inputs/formal_settings.json').is_file():
 PATHS={k:OUT/'inputs'/v for k,v in {'members':'frozen_members_and_groups.tsv.gz','scores':'historical_scores.tsv.gz','formal_policies':'formal_policies.tsv','formal_settings':'formal_settings.json','formal_threshold_stability':'formal_threshold_stability.tsv','original_plot_points':'original_Fig5_points.tsv'}.items()}
PCK=Path(__file__).resolve().parents[2]
FI=PCK/'frozen/source_decomposition/inputs'
PATHS={k:FI/v for k,v in {'members':'frozen_members_and_groups.tsv.gz','scores':'historical_scores.tsv.gz','formal_policies':'formal_policies.tsv','formal_settings':'formal_settings.json','formal_threshold_stability':'formal_threshold_stability.tsv','original_plot_points':'original_Fig5_points.tsv'}.items()}
SOURCES=['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']
MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']

def read(path):
 op=gzip.open if str(path).endswith('.gz') else open
 with op(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 for ext,sep in [('tsv','\t'),('csv',',')]:
  with (OUT/(name+'.'+ext)).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields,delimiter=sep);w.writeheader();w.writerows(rows)
def ratio(a,b):return a/b if b else np.nan
def eq(a,b):
 try:
  a,b=float(a),float(b)
  return (math.isnan(a) and math.isnan(b)) or math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
 except (ValueError,TypeError):return a==b
spec=importlib.util.spec_from_file_location('exact',Path(__file__).with_name('original_transfer_exact_v3.py'));exact=importlib.util.module_from_spec(spec);spec.loader.exec_module(exact)
# Load/read routines are copied as reference only; ROOT stays explicit here.
spec=importlib.util.spec_from_file_location('selective',Path(__file__).with_name('original_selective.py'));selective=importlib.util.module_from_spec(spec);spec.loader.exec_module(selective)
settings=json.loads(PATHS['formal_settings'].read_text(encoding='utf-8'));assert not settings['smoke_only'] and settings['bootstrap']==10000
for k,n in [('members','frozen_members_and_groups.tsv.gz'),('scores','historical_scores.tsv.gz')]:assert sha(PATHS[k])==settings['input_sha256'][n]
before={str(p):sha(p) for p in PATHS.values()}
meta=read(PATHS['members']);raw=read(PATHS['scores']);ix={(r['dataset'],r['sequence_id']):r for r in raw};assert len(meta)==29104 and len(ix)==len(raw)==41956
rows=[]
for d in meta:
 r=ix[d['dataset'],d['sequence_id']];assert r['sequence']==d['sequence'] and r['label']==d['label']
 assert len(r['sequence'])==int(d['length']) and hashlib.sha256(r['sequence'].encode('ascii')).hexdigest()==r['sequence_id'][2:]
 for m in MODELS:assert int(r[m+'__prediction'])==int(d[m+'__prediction'])
 x=dict(r);x['global_group']=d['global_group'];x['local_group']=d['local_group'];rows.append(x)
policies={(r['held_source'],r['model']):r for r in read(PATHS['formal_policies'])};assert len(policies)==36
stability={(r['held_source'],r['model']):r for r in read(PATHS['formal_threshold_stability'])}
frozen_points={(r['held_source'],r['model']):r for r in read(PATHS['original_plot_points'])};assert len(frozen_points)==23
source_rows={s:[r for r in rows if r['dataset']==s] for s in SOURCES}
decomposition=[];summaries=[];checks=[];grids=[];membership=[]

def check(held,model,metric,old,new):
 checks.append(dict(held_source=held,model=model,metric=metric,frozen_value=old,recomputed_value=new,pass_check=eq(old,new)))

def describe(rr,model,t):
 n=len(rr);y=np.array([int(r['label']) for r in rr]);pred=np.array([int(r[model+'__prediction']) for r in rr]);P=int(y.sum());N=n-P;G=len({r['global_group'] for r in rr});err=pred!=y
 base=dict(n=n,positive=P,negative=N,groups=G,full_errors=int(err.sum()),full_error_rate=float(err.mean()))
 if not np.isfinite(t):
  base.update({k:np.nan for k in ['retained_n','retained_positive','retained_negative','retained_groups','retained_positive_groups','retained_negative_groups','retained_error_groups','retained_errors','retained_TP','retained_FP','retained_TN','retained_FN','coverage','conditional_error','positive_rejection','retained_positive_fraction','retained_positive_error_rate','retained_negative_error_rate']});return base
 p=np.array([float(r[model+'__score']) for r in rr]);q=np.maximum(p,1-p);keep=q>=t;K=int(keep.sum());E=int((keep&err).sum());TP=int((keep&(y==1)&(pred==1)).sum());FP=int((keep&(y==0)&(pred==1)).sum());TN=int((keep&(y==0)&(pred==0)).sum());FN=int((keep&(y==1)&(pred==0)).sum());kp=TP+FN;kn=TN+FP
 base.update(retained_n=K,retained_positive=kp,retained_negative=kn,retained_groups=len({r['global_group'] for r,b in zip(rr,keep) if b}),retained_positive_groups=len({r['global_group'] for r,b,l in zip(rr,keep,y) if b and l==1}),retained_negative_groups=len({r['global_group'] for r,b,l in zip(rr,keep,y) if b and l==0}),retained_error_groups=len({r['global_group'] for r,b,e in zip(rr,keep,err) if b and e}),retained_errors=E,retained_TP=TP,retained_FP=FP,retained_TN=TN,retained_FN=FN,coverage=K/n,conditional_error=ratio(E,K),positive_rejection=1-ratio(kp,P),retained_positive_fraction=ratio(kp,K),retained_positive_error_rate=ratio(FN,kp),retained_negative_error_rate=ratio(FP,kn))
 return base
for fi,held in enumerate(SOURCES,1):
 target=source_rows[held];forbidden={r['global_group'] for r in target};before_dev=[r for r in rows if r['dataset']!=held];dev=[r for r in before_dev if r['global_group'] not in forbidden]
 assert not forbidden & {r['global_group'] for r in dev}
 assert not {r['sequence_id'] for r in target} & {r['sequence_id'] for r in dev}
 assert set(r['dataset'] for r in dev)==set(SOURCES)-{held}
 for m in MODELS:
  formal=policies[held,m];status=formal['policy_status'];eligible=m!='c_AMPs';t=float(formal['threshold']) if status=='feasible' else np.nan
  if eligible:
   prep=exact.prepare(dev,m);selected=exact.select_exact(np.ones((1,prep['G'])),prep,curve=True)
   newstatus='feasible' if bool(selected['feasible'][0]) else 'no_feasible_policy';check(held,m,'policy_status',status,newstatus)
   check(held,m,'threshold',formal['threshold'],selected['threshold'][0] if newstatus=='feasible' else np.nan)
   for metric,nm in [('development_coverage','coverage'),('development_error','error')]:check(held,m,metric,formal[metric],selected[nm][0])
   for metric,val in [('development_records',len(dev)),('development_groups',len({r['global_group'] for r in dev})),('purged_development_records',len(before_dev)-len(dev)),('test_records',len(target)),('test_groups',len(forbidden))]:check(held,m,metric,formal[metric],val)
   curve=selected['curve'];grids.append(dict(fold=fi,held_source=held,model=m,observed_q_thresholds=len(curve['threshold']),feasible_candidate_count=int(sum((curve['error']<=.05+1e-12)&(curve['coverage']>=.2-1e-12)&(curve['retained_records']>=100)&(curve['contributing_groups']>=30))),selection_reproduced=eq(t,selected['threshold'][0] if newstatus=='feasible' else np.nan)))
  else:assert status!='feasible'
  devdes=[]
  for source in SOURCES:
   role='held_out' if source==held else 'development';full=source_rows[source];rr=full if role=='held_out' else [r for r in dev if r['dataset']==source];desc=describe(rr,m,t)
   d=dict(fold=fi,held_source=held,model=m,source=source,role=role,policy_status=status,threshold=t,records_before_group_purge=len(full),purged_records=len(full)-len(rr),**desc,initial_source_mass=.2 if role=='development' else np.nan,initial_per_record_weight=1/(5*len(rr)) if role=='development' else np.nan)
   if role=='development':devdes.append(d)
   decomposition.append(d)
  if status=='feasible':
   Cdev=sum(d['coverage'] for d in devdes)/5;num=sum(d['retained_errors']/d['n'] for d in devdes)/5;Rdev=num/Cdev;Kdev=sum(d['retained_n'] for d in devdes)
   keepgroups={r['global_group'] for r in dev if max(float(r[m+'__score']),1-float(r[m+'__score']))>=t}
   check(held,m,'decomposed_development_coverage',formal['development_coverage'],Cdev);check(held,m,'decomposed_development_error',formal['development_error'],Rdev)
   assert Rdev<=.05+1e-12 and Cdev>=.2-1e-12 and Kdev>=100 and len(keepgroups)>=30
   for d in devdes:
    d.update(retained_source_mass=d['coverage']/5,retained_source_weight=d['coverage']/(5*Cdev),risk_contribution=(d['retained_errors']/d['n'])/(5*Cdev),FP_risk_contribution=(d['retained_FP']/d['n'])/(5*Cdev),FN_risk_contribution=(d['retained_FN']/d['n'])/(5*Cdev))
   assert abs(sum(d['retained_source_weight'] for d in devdes)-1)<1e-12
   assert abs(sum(d['risk_contribution'] for d in devdes)-Rdev)<1e-12
   assert abs(sum(d['FP_risk_contribution']+d['FN_risk_contribution'] for d in devdes)-Rdev)<1e-12
   y=np.array([int(r['label']) for r in target]);pred=np.array([int(r[m+'__prediction']) for r in target]);p=np.array([float(r[m+'__score']) for r in target]);W=np.ones((1,len(target)));S=W*(np.maximum(p,1-p)>=t)[None,:];metrics=selective.metrics(W,S,y,pred)
   for metric,v in metrics.items():check(held,m,'target_'+metric,formal[metric],float(v[0]))
   for metric,v in [('coverage',metrics['coverage'][0]),('retained_error',metrics['retained_error'][0]),('positive_rejection',metrics['positive_rejection'][0])]:check(held,m,'original_Fig5_'+metric,frozen_points[held,m][metric],float(v))
   td=next(d for d in decomposition[-6:] if d['role']=='held_out');risks=[d['conditional_error'] for d in devdes if np.isfinite(d['conditional_error'])]
   summ=dict(fold=fi,held_source=held,model=m,policy_status=status,threshold=t,development_records=len(dev),development_groups=len({r['global_group'] for r in dev}),purged_development_records=len(before_dev)-len(dev),development_coverage=Cdev,development_error=Rdev,development_retained_records=Kdev,development_retained_groups=len(keepgroups),development_source_error_min=min(risks),development_source_error_max=max(risks),development_sources_with_retained_records=len(risks),development_sources_above5=sum(r>.05+1e-12 for r in risks),development_highest_risk_source=' | '.join(d['source'] for d in devdes if eq(d['conditional_error'],max(risks))),development_retained_source_weight_min=min(d['retained_source_weight'] for d in devdes),development_retained_source_weight_max=max(d['retained_source_weight'] for d in devdes),development_pooled_unweighted_error=sum(d['retained_errors'] for d in devdes)/Kdev,development_unweighted_mean_source_conditional_error=float(np.mean(risks)),development_retained_positive_fraction=sum(d['retained_positive']/d['n'] for d in devdes)/(5*Cdev),held_n=td['n'],held_positive=td['positive'],held_negative=td['negative'],held_groups=td['groups'],held_retained_n=td['retained_n'],held_retained_positive=td['retained_positive'],held_retained_negative=td['retained_negative'],held_retained_errors=td['retained_errors'],held_retained_error=td['conditional_error'],held_coverage=td['coverage'],held_positive_rejection=td['positive_rejection'],held_error_minus_development_mix=td['conditional_error']-Rdev,held_error_above5=td['conditional_error']>.05+1e-12,bootstrap_policy_feasible_fraction=formal['bootstrap_policy_feasible_fraction'])
  else:
   for d in devdes:d.update(retained_source_mass=np.nan,retained_source_weight=np.nan,risk_contribution=np.nan,FP_risk_contribution=np.nan,FN_risk_contribution=np.nan)
   summ=dict(fold=fi,held_source=held,model=m,policy_status=status,threshold=np.nan,development_records=len(dev),development_groups=len({r['global_group'] for r in dev}),purged_development_records=len(before_dev)-len(dev),held_n=len(target),held_positive=sum(int(r['label']) for r in target),held_negative=sum(int(r['label'])==0 for r in target),held_groups=len(forbidden),development_error=np.nan,development_coverage=np.nan,held_retained_error=np.nan,held_coverage=np.nan,held_positive_rejection=np.nan,reason=formal.get('reason',''),bootstrap_policy_feasible_fraction=formal.get('bootstrap_policy_feasible_fraction',''))
  summaries.append(summ)
 # No per-model duplicate records needed: one fold/source membership table.
 for source in SOURCES:
  full=source_rows[source];rr=full if source==held else [r for r in dev if r['dataset']==source]
  membership.append(dict(fold=fi,held_source=held,source=source,role='held_out' if source==held else 'development',records_before=len(full),records_after=len(rr),purged_records=len(full)-len(rr),positive_after=sum(int(r['label']) for r in rr),negative_after=sum(int(r['label'])==0 for r in rr),groups_after=len({r['global_group'] for r in rr}),group_overlap_with_held=0 if source!=held else 'not_applicable',exact_overlap_with_held=0 if source!=held else 'not_applicable'))
 print('FOLD',fi,held,'dev',len(dev),'purged',len(before_dev)-len(dev),flush=True)
assert all(d['pass_check'] for d in checks)
feasible=[r for r in summaries if r['policy_status']=='feasible'];assert len(feasible)==23 and sum(r['held_error_above5'] for r in feasible)==11
assert Counter(r['policy_status'] for r in summaries if r['model']!='c_AMPs')=={'feasible':23,'no_feasible_policy':7}
write('01_逐折来源分解表_EN',decomposition);write('02_逐折模型汇总',summaries);write('03_逐折成员与去重核查',membership);write('04_阈值与原结果复现核验',checks);write('05_完整阈值搜索复现摘要',grids);write('06_图5来源数据',feasible)
# Compact Chinese table for direct inspection in Excel.
cnmap={'fold':'折次','held_source':'留出来源','model':'模型','source':'当前来源','role':'用途','policy_status':'规则状态','threshold':'冻结阈值_q','records_before_group_purge':'排除前记录数','purged_records':'排除的重叠分组记录数','n':'评估记录数','positive':'来源阳性数','negative':'来源阴性数','groups':'依赖块数','retained_n':'保留记录数','retained_positive':'保留阳性数','retained_negative':'保留阴性数','retained_groups':'保留依赖块数','retained_errors':'保留错误数','retained_FP':'保留假阳性数','retained_FN':'保留假阴性数','coverage':'保留覆盖率','conditional_error':'保留集条件错误率','positive_rejection':'来源阳性丢弃率','retained_source_weight':'开发保留集来源权重','risk_contribution':'对开发混合条件错误的贡献','FP_risk_contribution':'假阳性风险贡献','FN_risk_contribution':'假阴性风险贡献'}
cn=[]
for d in decomposition:
 row={cnmap[k]:d.get(k,np.nan) for k in cnmap};row['用途']='开发' if d['role']=='development' else '留出评估';cn.append(row)
write('07_逐折来源分解表_中文',cn)
write('08_来源文件与校验值',[dict(name=k,path=str(p),sha256=before[str(p)],unchanged=before[str(p)]==sha(p)) for k,p in PATHS.items()])
result=dict(folds=6,models=6,source_decomposition_rows=len(decomposition),development_source_rows=sum(r['role']=='development' for r in decomposition),held_source_rows=sum(r['role']=='held_out' for r in decomposition),feasible=23,infeasible=7,score_ineligible=6,feasible_folds_with_any_development_source_error_above5=sum(r['development_sources_above5']>0 for r in feasible),feasible_folds_with_all_development_sources_at_most5=sum(r['development_sources_above5']==0 for r in feasible),development_source_cells_above5=sum(r['development_sources_above5'] for r in feasible),held_above5=11,max_development_source_error=max(r['development_source_error_max'] for r in feasible),max_held_error=max(r['held_retained_error'] for r in feasible),reproduction_checks=len(checks),all_reproduction_checks_pass=all(x['pass_check'] for x in checks),new_model_inference=False,new_bootstrap=False,uncertainty='new source decomposition is a finite-source point summary and source range; original selector-refit bootstrap output preserved separately; range is not a confidence interval',versions={'python':sys.version,'numpy':np.__version__},original_inputs_unchanged=all(before[str(p)]==sha(p) for p in PATHS.values()))
(OUT/'分解结论.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
