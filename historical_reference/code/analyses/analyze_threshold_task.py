"""Activity-threshold and assay-task sensitivity on 151 adjudicated candidates.
Frozen sequence/model outputs; no fitting, unit conversion or universal-negative inference.
Read-only inputs; separate output files. Interval-censored unknowns remain unknown.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,importlib.util,json,math,re,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'rebuild_analyses/task_threshold'
OUT.mkdir(parents=True,exist_ok=True)
AUDIT=ROOT/'frozen/evidence_adjudication'
BUDGET=ROOT/'frozen/membership_analysis'
SPEC=importlib.util.spec_from_file_location('budget_core',Path(__file__).with_name('original_budget_functions.py'))
C=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(C)
B=10000;SEED=2026102502
PATHS={'members':AUDIT/'08_新增151活性设计序列_排除定向阴性对照.tsv','assays':AUDIT/'02_MIC逐条证据与标签.tsv','scores':BUDGET/'active_design_151/sequence_score_diagnostics.tsv','previous_diagnostics':BUDGET/'active_design_151/task_diagnostics.tsv','previous_budget_points':BUDGET/'active_design_151/budget_points.tsv','audit_delivery_manifest':AUDIT/'交付文件清单.json','budget_delivery_manifest':BUDGET/'交付文件SHA256.json'}

def write(name,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    for ext,sep in [('tsv','\t'),('csv',',')]:
        with (OUT/(name+'.'+ext)).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,delimiter=sep);w.writeheader();w.writerows(rows)
def n(x):return C.scalar(x)
def same(a,b):
    if a==b:return True
    try:
        a,b=float(a),float(b)
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    except (ValueError,TypeError):return False

def parse(raw,t):
    s=str(raw).strip();m=re.fullmatch(r'(<=|>=|<|>|≤|≥)?\s*(\d+(?:\.\d+)?)',s)
    if not m:return -1
    op={'≤':'<=','≥':'>='}.get(m[1],m[1]) or '=';v=float(m[2]);t=float(t)
    if op=='=':return int(v<=t)
    if op in ['<','<='] and v<=t:return 1
    if (op=='>' and v>=t) or (op=='>=' and v>t):return 0
    return -1

def combine(labels):
    if 1 in labels:return 1
    if labels and all(v==0 for v in labels):return 0
    return -1

def partial_bounds(y,weights):
    """Sharp finite-pool bounds under free binary completions of unresolved labels.
    These are identification bounds, not confidence intervals. Recovery/enrichment
    extrema enumerate unknown-positive count and sort fixed selection weights.
    """
    y=np.asarray(y);w=np.asarray(weights,float);N=len(y);P=int((y==1).sum());U=int((y==-1).sum());sel=float(w.sum())
    h=float(w[y==1].sum());wu=np.sort(w[y==-1]);lo=np.r_[0,np.cumsum(wu)];hi=np.r_[0,np.cumsum(wu[::-1])]
    den=P+np.arange(U+1);valid=den>0
    rlo=float(np.min((h+lo[valid])/den[valid])) if valid.any() else np.nan
    rhi=float(np.max((h+hi[valid])/den[valid])) if valid.any() else np.nan
    coeff=wu-sel/N
    excess_base=h-sel*P/N
    return dict(known_positive=P,unresolved=U,selected_n=sel,positive_hits_lower=h,positive_hits_upper=h+wu.sum(),precision_lower=h/sel if sel else np.nan,precision_upper=(h+wu.sum())/sel if sel else np.nan,positive_recovery_lower=rlo,positive_recovery_upper=rhi,enrichment_lower=rlo*N/sel if sel else np.nan,enrichment_upper=rhi*N/sel if sel else np.nan,random_expected_hits_lower=sel*P/N,random_expected_hits_upper=sel*(P+U)/N,excess_hits_lower=excess_base+coeff[coeff<0].sum(),excess_hits_upper=excess_base+coeff[coeff>0].sum(),zero_positive_completion_possible=P==0,bounds_type='finite_pool_identification_not_confidence_interval')

# Boundary and partial-identification checks, including a bound exactly at T.
for raw,t,exp in [('>64',64,0),('>64',128,-1),('>=64',64,-1),('<=64',64,1),('N',64,-1),('64',64,1),('>512',128,0)]:assert parse(raw,t)==exp
from itertools import product
for y,w in [([1,-1,-1,0],[1,.5,0,.5]),([0,-1,-1],[0,1,0]),([1,0],[.5,.5])]:
    z=partial_bounds(y,w);ui=np.flatnonzero(np.array(y)==-1);vals=[]
    for fill in product([0,1],repeat=len(ui)):
        yy=np.array(y);yy[ui]=fill;P=int(yy.sum());H=float(np.asarray(w)[yy==1].sum());S=sum(w)
        vals.append((H,H/P if P else np.nan,H-S*P/len(yy)))
    assert same(z['positive_hits_lower'],min(x[0] for x in vals)) and same(z['positive_hits_upper'],max(x[0] for x in vals))
    good=[x[1] for x in vals if np.isfinite(x[1])]
    if good:assert same(z['positive_recovery_lower'],min(good)) and same(z['positive_recovery_upper'],max(good))
    assert same(z['excess_hits_lower'],min(x[2] for x in vals)) and same(z['excess_hits_upper'],max(x[2] for x in vals))

# Verify against existing delivery manifests before deriving endpoint labels.
for prefix,path,needed in [('audit',PATHS['audit_delivery_manifest'],['08_新增151活性设计序列_排除定向阴性对照.tsv','02_MIC逐条证据与标签.tsv']),('budget',PATHS['budget_delivery_manifest'],['active_design_151/sequence_score_diagnostics.tsv','active_design_151/task_diagnostics.tsv','active_design_151/budget_points.tsv'])]:
    manifest=json.loads(path.read_text(encoding='utf-8'));ix={x['path'].replace('\\','/'):x['sha256'] for x in manifest}
    root=AUDIT if prefix=='audit' else BUDGET
    for name in needed:assert C.sha(root/name)==ix[name],name
before={str(p):C.sha(p) for p in PATHS.values()}
members=C.read(PATHS['members']);assays=C.read(PATHS['assays']);scores=C.read(PATHS['scores'])
assert len(members)==151 and len(scores)==906
assert not {'HydrAMP__Pex-P1-4','HydrAMP__Temp-P1-5'} & {p['record_id'] for p in members}
mid={p['record_id']:p for p in members};six={(s['sequence_id'],s['model']):s for s in scores};assert len(six)==906
byid=defaultdict(list)
for a in assays:
    if a['record_id'] in mid and a['primary_eligible']=='1':byid[a['record_id']].append(a)
assert sum(len(v) for v in byid.values())==1447 # 120*11 + 24*5 + 7*1
for p in members:
    assert hashlib.sha256(p['sequence'].encode('ascii')).hexdigest()==p['sequence_sha256']
    for model in C.MODELS:
        s=six[p['sequence_id'],model];assert int(s['label'])==int(p['activity_label_binary']) and int(s['length'])==int(p['length'])

labels=[];assay_labels=[];summary=[];availability=[];diags=[];intervals=[];budgets=[];bounds=[];paired=[];conflicts=[];transitions=[];topweights=[];baseline_checks=[]
LABELS={};TASKS={}
# Main tasks are each source's specific reference E. coli strain, never a fake cross-study common strain.
for cohort in C.COHORTS:
    pp=[p for p in members if p['cohort']==cohort];is_apex=cohort.startswith('ApexGO');unit='uM' if is_apex else 'ug/mL';thresholds=[16,32,64,128] if is_apex else [32,64,128];base=64 if is_apex else 32
    reference='E. coli ATCC 11775' if is_apex else 'E. coli ATCC 25922'
    TASKS[cohort]=[]
    for task,rawstrain in [('any_eligible_measured_strain',''),('source_reference_Ecoli',reference)]:
        for t in thresholds:
            tid=(task,t);TASKS[cohort].append(dict(task=task,threshold=t,unit=unit,strain=rawstrain,baseline_threshold=base,threshold_role='censoring_stress_nonidentifiable_when_gt64' if is_apex and t==128 else ('source_primary_threshold' if t==base else 'activity_threshold_sensitivity')))
            yy=[]
            for p in pp:
                aa=[a for a in byid[p['record_id']] if not rawstrain or a['strain_raw']==rawstrain]
                assert aa and all(a['unit']==unit for a in aa)
                labs=[parse(a['MIC_raw'],t) for a in aa];y=combine(labs);yy.append(y)
                alt=[parse(a.get('alternate_source_MIC') or a['MIC_raw'],t) for a in aa];ya=combine(alt)
                if any(a.get('alternate_source_MIC') for a in aa):
                    conflicts.append(dict(cohort=cohort,task=task,reference_strain=rawstrain,threshold=t,unit=unit,record_id=p['record_id'],workbook_label=y,article_alternative_label=ya,label_changed=int(y!=ya),source_locations=' | '.join(a['source_location'] for a in aa if a.get('alternate_source_MIC')),raw_MIC=' | '.join(a['MIC_raw'] for a in aa if a.get('alternate_source_MIC')),alternative_MIC=' | '.join(a['alternate_source_MIC'] for a in aa if a.get('alternate_source_MIC'))))
                labels.append(dict(cohort=cohort,task=task,reference_strain=rawstrain,threshold=t,unit=unit,record_id=p['record_id'],sequence_id=p['sequence_id'],label=y,label_name={1:'positive',0:'threshold_negative',-1:'unresolved'}[y],eligible_assay_n=len(aa),known_positive_assays=sum(x==1 for x in labs),known_negative_assays=sum(x==0 for x in labs),unresolved_assays=sum(x==-1 for x in labs),qualifying_source_locations=' | '.join(a['source_location']+'='+a['MIC_raw'] for a,l in zip(aa,labs) if l==1),unit_conversion=False))
                for a,l in zip(aa,labs):assay_labels.append(dict(cohort=cohort,task=task,record_id=p['record_id'],threshold=t,unit=unit,strain_raw=a['strain_raw'],MIC_raw=a['MIC_raw'],derived_assay_label=l,source_file=a['source_file'],source_location=a['source_location'],alternate_source_MIC=a.get('alternate_source_MIC','')))
            LABELS[cohort,task,t]={p['record_id']:y for p,y in zip(pp,yy)}
    if is_apex:
        availability.append(dict(cohort=cohort,requested_task='E. coli ATCC 25922',status='not_measured_in_this_source',available_reference=reference,action='do_not_substitute_strain_ID; source-specific reference task reported separately',same_strain_crossstudy_comparison_possible=False))
    else:availability.append(dict(cohort=cohort,requested_task='E. coli ATCC 25922',status='measured_and_eligible',available_reference=reference,action='compare within HydrAMP; not a common strain shared with ApexGO',same_strain_crossstudy_comparison_possible=False))

for ci,cohort in enumerate(C.COHORTS):
    for subset in ['all_active_design','five_inventory_pass']:
        pp=[p for p in members if p['cohort']==cohort and (subset=='all_active_design' or C.yes(six[p['sequence_id'],'Macrel']['inventory_pass']))];N=len(pp)
        if not N:
            for task in TASKS[cohort]:summary.append(dict(cohort=cohort,subset=subset,**task,n=0,positive=0,threshold_negative=0,unresolved=0,blocks=0,positive_blocks=0,negative_blocks=0,prevalence_lower=np.nan,prevalence_upper=np.nan,status='empty'))
            continue
        groups=[six[p['sequence_id'],'Macrel']['block'] for p in pp];gg,gi=np.unique(groups,return_inverse=True);G=len(gg);W=np.ones((1,N))
        rng=np.random.default_rng(SEED+ci*100+int(subset!='all_active_design'));BW=None if G<5 else rng.multinomial(G,np.ones(G)/G,size=B)[:,gi].astype(float)
        tasks=TASKS[cohort];Y={};cache={};draws={}
        for task in tasks:
            tid=(task['task'],task['threshold']);y=np.array([LABELS[cohort,*tid][p['record_id']] for p in pp]);Y[tid]=y
            P=int((y==1).sum());U=int((y==-1).sum());nn=int((y==0).sum());status='unresolved_activity_membership' if U else ('positive_only' if P==N else ('negative_only' if P==0 else 'both_classes'))
            summary.append(dict(cohort=cohort,subset=subset,**task,n=N,positive=P,threshold_negative=nn,unresolved=U,blocks=G,positive_blocks=len({g for g,v in zip(groups,y) if v==1}),negative_blocks=len({g for g,v in zip(groups,y) if v==0}),prevalence_lower=P/N,prevalence_upper=(P+U)/N,status=status))
        for model in C.MODELS:
            sc=np.array([float(six[p['sequence_id'],model]['score']) for p in pp]);de=np.array([int(six[p['sequence_id'],model]['prediction']) for p in pp]);scoredef='exploratory_min_branch_extension' if model=='c_AMPs' else 'saved_AMP_class_score'
            for task in tasks:
                tid=(task['task'],task['threshold']);y=Y[tid];U=int((y==-1).sum());P=int((y==1).sum())
                mk=dict(cohort=cohort,subset=subset,**task,model=model,score_definition=scoredef,n=N,positive=P,threshold_negative=int((y==0).sum()),unresolved=U,blocks=G)
                if U:
                    diags.append(dict(**mk,metric_status='not_point_identified_do_not_drop_unknowns',native_positive_call_rate=float(de.mean()),TP=np.nan,FP=np.nan,TN=np.nan,FN=np.nan,native_recall=np.nan,assay_negative_positive_call_rate=np.nan,native_MCC=np.nan,native_balanced_accuracy=np.nan,prevalence=np.nan,AP=np.nan,AUROC=np.nan,AP_minus_prevalence=np.nan))
                    continue
                d=C.diagnostics(W,y,de,sc);cache[tid,model]={k:n(v) for k,v in d.items()}
                diags.append(dict(**mk,metric_status='binary_discrimination_estimable' if 0<P<N else 'single_class_descriptive_only',**{k:n(v) for k,v in d.items() if k not in ['n','positive','negative']}))
                if BW is not None:
                    bd=C.diagnostics(BW,y,de,sc)
                    # For discrimination intervals, a bootstrap resample requires both true classes.
                    both=(BW[:,y==1].sum(axis=1)>0)&(BW[:,y==0].sum(axis=1)>0)
                    for met in ['AP','AUROC','native_MCC','native_balanced_accuracy','AP_minus_prevalence']:bd[met]=np.where(both,bd[met],np.nan)
                for met in ['native_recall','assay_negative_positive_call_rate','native_MCC','native_balanced_accuracy','AP','AUROC','AP_minus_prevalence']:
                    if BW is None:lo,hi,st,v=np.nan,np.nan,'fewer_than_five_blocks',np.nan
                    else:lo,hi,st,v=C.interval(bd[met],B,G)
                    intervals.append(dict(**mk,analysis='diagnostic',metric=met,budget='',policy='',ci_low=lo,ci_high=hi,interval_status=st,valid_fraction=v,bootstrap=B if BW is not None else 0))
                    if BW is not None:draws[tid,model,met]=bd[met]
            for budget in C.BUDGETS:
                for policy in ['score_topK','native_positive_then_score']:
                    S,K=C.selected(W,sc,budget,None if policy=='score_topK' else de==1)
                    BS,BK=(None,None) if BW is None else C.selected(BW,sc,budget,None if policy=='score_topK' else de==1)
                    if subset=='all_active_design' and budget==.2:
                        for p,s,yy,ww in zip(pp,sc,de,S[0]):topweights.append(dict(cohort=cohort,model=model,policy=policy,budget=.2,record_id=p['record_id'],sequence_id=p['sequence_id'],score=float(s),prediction=int(yy),selection_weight=float(ww),budget_slots=float(K[0]),score_definition=scoredef))
                    for task in tasks:
                        tid=(task['task'],task['threshold']);y=Y[tid];U=int((y==-1).sum());P=int((y==1).sum())
                        mk=dict(cohort=cohort,subset=subset,**task,model=model,score_definition=scoredef,n=N,positive=P,threshold_negative=int((y==0).sum()),unresolved=U,blocks=G,budget=budget,policy=policy)
                        if U:
                            bounds.append(dict(**mk,**{k:v for k,v in partial_bounds(y,S[0]).items() if k not in ['known_positive','unresolved']},budget_slots=float(K[0])))
                            continue
                        d=C.budget_metrics(W,S,y,K);budgets.append(dict(**mk,**{k:n(v) for k,v in d.items()}))
                        if budget==.2 and policy=='score_topK':cache[tid,model].update({'B20_'+k:n(v) for k,v in d.items()})
                        if BW is not None:bd=C.budget_metrics(BW,BS,y,BK)
                        for met in ['positive_hits','precision_at_budget','positive_recovery','enrichment','excess_hits_vs_random','precision_gain_vs_random','recovery_gain_vs_random']:
                            if BW is None:lo,hi,st,v=np.nan,np.nan,'fewer_than_five_blocks',np.nan
                            else:lo,hi,st,v=C.interval(bd[met],B,G)
                            intervals.append(dict(**mk,analysis='budget',metric=met,ci_low=lo,ci_high=hi,interval_status=st,valid_fraction=v,bootstrap=B if BW is not None else 0))
                            if BW is not None and budget==.2 and policy=='score_topK':draws[tid,model,'B20_'+met]=bd[met]
        # Paired changes across thresholds and tasks share the exact same block draws.
        contrast_tasks=[]
        base=tasks[0]['baseline_threshold']
        for name in ['any_eligible_measured_strain','source_reference_Ecoli']:
            for t in sorted({t['threshold'] for t in tasks}):
                if t!=base:contrast_tasks.append(('threshold_change',(name,base),(name,t)))
        for t in sorted({t['threshold'] for t in tasks}):contrast_tasks.append(('task_change',('any_eligible_measured_strain',t),('source_reference_Ecoli',t)))
        for contrast,oldtid,newtid in contrast_tasks:
            ya,yb=Y[oldtid],Y[newtid];U=int((ya==-1).sum()+(yb==-1).sum())
            for model in C.MODELS:
                pred=np.array([int(six[p['sequence_id'],model]['prediction']) for p in pp])
                for p,a,b,dd in zip(pp,ya,yb,pred):
                    if subset=='all_active_design':transitions.append(dict(cohort=cohort,contrast=contrast,model=model,record_id=p['record_id'],from_task=oldtid[0],to_task=newtid[0],from_threshold=oldtid[1],to_threshold=newtid[1],unit=p['unit'],from_label=int(a),to_label=int(b),label_changed=int(a!=b),frozen_native_prediction=int(dd),from_error=int(dd!=a) if a!=-1 else '',to_error=int(dd!=b) if b!=-1 else '',note='label_definition_change_not_model_retraining'))
                for met in ['native_MCC','AUROC','AP','AP_minus_prevalence','native_recall','assay_negative_positive_call_rate','B20_positive_hits','B20_precision_at_budget','B20_positive_recovery','B20_enrichment','B20_excess_hits_vs_random']:
                    mk=dict(cohort=cohort,subset=subset,contrast=contrast,model=model,from_task=oldtid[0],to_task=newtid[0],from_threshold=oldtid[1],to_threshold=newtid[1],unit=tasks[0]['unit'],n=N,blocks=G,metric=met)
                    if U:paired.append(dict(**mk,from_estimate=np.nan,to_estimate=np.nan,difference=np.nan,ci_low=np.nan,ci_high=np.nan,interval_status='unresolved_membership_no_complete_case_substitution',valid_fraction=np.nan,bootstrap=0));continue
                    a,b=cache[oldtid,model][met],cache[newtid,model][met]
                    if BW is None:lo,hi,st,v=np.nan,np.nan,'fewer_than_five_blocks',np.nan
                    else:lo,hi,st,v=C.interval(draws[newtid,model,met]-draws[oldtid,model,met],B,G)
                    paired.append(dict(**mk,from_estimate=a,to_estimate=b,difference=b-a,ci_low=lo,ci_high=hi,interval_status=st,valid_fraction=v,bootstrap=B if BW is not None else 0))
        print('ANALYSED',cohort,subset,'n',N,'blocks',G,'endpoint_thresholds',len(tasks),flush=True)

# Reproduce the prior adjudicated 151 native-threshold results before interpreting sensitivities.
previous_diag={(x['cohort'],x['subset'],x['task'],x['model']):x for x in C.read(PATHS['previous_diagnostics'])}
previous_budget={(x['cohort'],x['subset'],x['task'],x['model'],float(x['budget']),x['policy']):x for x in C.read(PATHS['previous_budget_points'])}
for x in diags:
    if x['threshold']!=x['baseline_threshold']:continue
    oldsubset='all_generated' if x['subset']=='all_active_design' else x['subset'];oldtask='any_eligible_measured_strain' if x['task']=='any_eligible_measured_strain' else 'strain|'+x['strain']+'|'+x['unit']+'|'+str(x['threshold'])
    old=previous_diag[x['cohort'],oldsubset,oldtask,x['model']]
    for met in ['TP','FP','TN','FN','native_MCC','native_recall','assay_negative_positive_call_rate','AP','AUROC','AP_minus_prevalence']:
        baseline_checks.append(dict(cohort=x['cohort'],subset=x['subset'],task=x['task'],model=x['model'],metric=met,old_value=old[met],new_value=x[met],pass_check=same(old[met],x[met])))
for x in budgets:
    if x['threshold']!=x['baseline_threshold']:continue
    oldsubset='all_generated' if x['subset']=='all_active_design' else x['subset'];oldtask='any_eligible_measured_strain' if x['task']=='any_eligible_measured_strain' else 'strain|'+x['strain']+'|'+x['unit']+'|'+str(x['threshold'])
    old=previous_budget[x['cohort'],oldsubset,oldtask,x['model'],x['budget'],x['policy']]
    for met in ['positive_hits','precision_at_budget','positive_recovery','enrichment','random_expected_hits','excess_hits_vs_random']:
        baseline_checks.append(dict(cohort=x['cohort'],subset=x['subset'],task=x['task'],model=x['model'],budget=x['budget'],policy=x['policy'],metric=met,old_value=old[met],new_value=x[met],pass_check=same(old[met],x[met])))
assert all(x['pass_check'] for x in baseline_checks)
assert not any(x['label_changed'] for x in conflicts),'Source conflict changes main endpoints: requires identification treatment'

write('01_成员与阈值任务标签',labels);write('02_逐项MIC阈值判定',assay_labels);write('03_任务阈值与可评估性',summary);write('04_菌株任务对应关系',availability)
write('05_六模型原生分类与排序',diags);write('06_固定预算结果',budgets);write('07_探索性分组Bootstrap区间',intervals);write('08_阈值与任务的配对变化',paired);write('09_删失数据的部分识别界限',bounds)
write('10_原文补表冲突对标签的影响',conflicts);write('11_标签改变与原生错误转换',transitions);write('12_20pct预算选择权重',topweights);write('13_原151条结果复现核验',baseline_checks)
write('14_来源文件与指纹',[dict(name=k,path=str(p),sha256=before[str(p)],unchanged=C.sha(p)==before[str(p)]) for k,p in PATHS.items()])
write('15_主任务20pct结果',[r for r in budgets if r['subset']=='all_active_design' and r['budget']==.2 and r['policy']=='score_topK'])
settings=dict(version='activity_threshold_task_sensitivity_v1',date='2026-10-25',n_active_design=151,model_count=6,seed=SEED,bootstrap_replicates=B,thresholds={'ApexGO_uM':[16,32,64],'HydrAMP_ug_per_mL':[32,64,128],'ApexGO_128uM':'censoring stress; unresolved labels retained; partial identification where needed'},task_definition='any eligible initial measured strain vs source-specific E.coli reference (ApexGO11775; HydrAMP25922)',not_same_strain_across_sources=True,thresholds_fixed_for_sensitivity_not_optimized_to_model_performance=True,subsets=['all_active_design','five_inventory_pass'],policies=['score_topK','native_positive_then_score'],budgets=C.BUDGETS,primary_budget=.2,bootstrap='uniform original-block resampling; record multiplicities retained; budgets reselected; shared within cohort/subset across thresholds,tasks,models,policies',interval_rule='pointwise exploratory percentile95%; >=5 blocks; >=95% valid; degenerate omitted; paired changes computed directly; no multiplicity adjustment',diagnostic_bootstrap_rule='MCC/AP/AUROC/BA/APminusprevalence replicates require both classes; all-positive AP point descriptive only',unknown_policy='retain full candidate pool; no complete-case performance substitution; finite-pool identification bounds are not confidence intervals',source_conflict='both raw MIC versions audited; all primary endpoint labels invariant; exact strain MIC conflict retained',no_inference_or_fitting=True,no_wetlab=True,no_cross_unit_conversion=True,no_pooled_crossstudy_metrics=True,retrospective_exploratory=True,python=sys.version,numpy=np.__version__,input_hashes=before,original_inputs_unchanged=all(C.sha(p)==before[str(p)] for p in PATHS.values()),counts={k:len(v) for k,v in [('member_labels',labels),('assay_labels',assay_labels),('endpoint_summaries',summary),('diagnostics',diags),('budget_rows',budgets),('intervals',intervals),('paired_changes',paired),('identification_bounds',bounds),('native_threshold_checks',baseline_checks)]})
(OUT/'run_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(settings['counts']),flush=True)
