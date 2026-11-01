"""Group-separated internal validation and matched-support source-risk controls.

No model fitting; only selection labels choose a threshold. Repeated-split
quantiles are sensitivity summaries, not confidence intervals.
"""
from collections import defaultdict
import csv
import gzip
import math
import numpy as np
from common import *

POLICIES=['mixture_original','mixture_source_support','per_source_supported']

def curve(rows,model,sources):
    if not rows or any(not any(r['dataset']==s for r in rows) for s in sources):return None
    q=np.array([max(float(r[model+'__score']),1-float(r[model+'__score'])) for r in rows])
    assert np.isfinite(q).all() and ((q>=.5)&(q<=1)).all()
    order=np.argsort(-q,kind='stable');q=q[order]
    error=np.array([int(r[model+'__prediction'])!=int(r['label']) for r in rows])[order]
    sid=np.array([sources.index(r['dataset']) for r in rows],int)[order]
    groups=np.array([r['global_group'] for r in rows])[order]
    ends=np.r_[np.flatnonzero(q[1:]!=q[:-1]),len(q)-1]
    first=np.zeros(len(q),bool);first[np.unique(groups,return_index=True)[1]]=True
    k=[];e=[];g=[];n=[]
    for s in range(len(sources)):
        mask=sid==s;n.append(mask.sum());k.append(np.cumsum(mask)[ends]);e.append(np.cumsum(mask&error)[ends])
        positions=np.flatnonzero(mask);sf=np.zeros(len(q),bool)
        sf[positions[np.unique(groups[positions],return_index=True)[1]]]=True
        g.append(np.cumsum(sf)[ends])
    k=np.array(k).T;e=np.array(e).T;g=np.array(g).T;n=np.array(n)
    source_cov=k/n;coverage=source_cov.mean(axis=1)
    risk=ratio((e/n).sum(axis=1),source_cov.sum(axis=1))
    source_risk=ratio(e,k)
    return dict(threshold=q[ends],coverage=coverage,risk=risk,retained=k.sum(axis=1),
        groups=np.cumsum(first)[ends],source_cov=source_cov,source_risk=source_risk,
        source_records=k,source_groups=g,selection_source_n=n,sources=sources)

def choose(c,policy):
    if c is None:return dict(policy_status='no_feasible_policy',threshold=np.nan,reason='missing_development_source_in_selection',candidate_thresholds=0)
    shared=(c['coverage']>=.2-1e-12)&(c['retained']>=100)&(c['groups']>=30)
    if policy=='mixture_original':ok=shared&(c['risk']<=.05+1e-12)
    else:
        support=(c['source_cov']>=.2-1e-12).all(axis=1)&(c['source_records']>=20).all(axis=1)&(c['source_groups']>=5).all(axis=1)
        ok=shared&support
        if policy=='mixture_source_support':ok&=c['risk']<=.05+1e-12
        elif policy=='per_source_supported':ok&=(c['source_risk']<=.05+1e-12).all(axis=1)
        else:raise ValueError(policy)
    good=np.flatnonzero(ok)
    if not len(good):return dict(policy_status='no_feasible_policy',threshold=np.nan,reason='no_observed_threshold_met_all_prespecified_constraints',candidate_thresholds=len(ok))
    # As the threshold falls, source-balanced coverage increases. Last feasible
    # tier gives maximum coverage and the lowest threshold at any coverage tie.
    j=int(good[-1])
    result=dict(policy_status='feasible',threshold=float(c['threshold'][j]),reason='',candidate_thresholds=len(ok),feasible_thresholds=len(good),selection_coverage=float(c['coverage'][j]),selection_error=float(c['risk'][j]),selection_retained_records=int(c['retained'][j]),selection_retained_groups=int(c['groups'][j]),selection_worst_source_error=float(np.nanmax(c['source_risk'][j])),selection_min_source_coverage=float(c['source_cov'][j].min()))
    if policy=='per_source_supported':assert result['selection_worst_source_error']<=.05+1e-12
    return result

def evaluate(rows,model,threshold,sources):
    available=np.isfinite(threshold);detail=[]
    for source in sources:
        rr=[r for r in rows if r['dataset']==source];n=len(rr)
        y=np.array([int(r['label']) for r in rr],int)
        pred=np.array([int(r[model+'__prediction']) for r in rr],int)
        q=np.array([max(float(r[model+'__score']),1-float(r[model+'__score'])) for r in rr])
        err=pred!=y;P=int(y.sum());N=n-P
        row=dict(source=source,n=n,positive=P,threshold_negative=N,groups=len({r['global_group'] for r in rr}),full_errors=int(err.sum()),full_error=scalar(ratio(err.sum(),n)),policy_available=bool(available))
        if available and n:
            keep=q>=threshold;K=int(keep.sum());E=int((keep&err).sum());kp=int((keep&(y==1)).sum())
            kn=K-kp;tp=int((keep&(y==1)&(pred==1)).sum())
            row.update(retained_n=K,retained_errors=E,retained_positive=kp,retained_negative=kn,
                retained_positive_detected=tp,coverage=K/n,conditional_error=scalar(ratio(E,K)),
                positive_discarded=P-kp,positive_discarded_fraction=scalar(ratio(P-kp,P)),
                retained_groups=len({r['global_group'] for r,k in zip(rr,keep) if k}),
                outcome_status='estimable' if K else 'zero_retained_error_undefined')
        else:
            row.update({k:np.nan for k in ['retained_n','retained_errors','retained_positive','retained_negative','retained_positive_detected','coverage','conditional_error','positive_discarded','positive_discarded_fraction','retained_groups']})
            row['outcome_status']='policy_unavailable' if not available else 'empty_evaluation_source'
        detail.append(row)
    n=sum(r['n'] for r in detail);p=sum(r['positive'] for r in detail)
    result=dict(n=n,positive=p,threshold_negative=n-p,groups=len({r['global_group'] for r in rows}),full_error_pooled=scalar(ratio(sum(r['full_errors'] for r in detail),n)),evaluation_sources=len(sources))
    valid=available and all(r['n']>0 for r in detail)
    if valid:
        ks=np.array([r['retained_n'] for r in detail],float);es=np.array([r['retained_errors'] for r in detail],float)
        ns=np.array([r['n'] for r in detail],float);ps=np.array([r['positive'] for r in detail],float)
        kp=np.array([r['retained_positive'] for r in detail],float)
        source_cov=ks/ns;total=source_cov.sum();K=int(ks.sum());E=int(es.sum())
        result.update(retained_n=K,retained_errors=E,coverage_balanced=float(source_cov.mean()),coverage_pooled=scalar(ratio(K,n)),
            conditional_error_balanced=scalar(ratio((es/ns).sum(),total)),conditional_error_pooled=scalar(ratio(E,K)),
            positive_discarded=int((ps-kp).sum()),positive_discarded_fraction_balanced=scalar(ratio(((ps-kp)/ns).sum(),(ps/ns).sum())),
            positive_discarded_fraction_pooled=scalar(ratio((ps-kp).sum(),p)),
            worst_source_error=max((r['conditional_error'] for r in detail if np.isfinite(r['conditional_error'])),default=np.nan),
            zero_retained_sources=sum(r['retained_n']==0 for r in detail),
            outcome_status='estimable' if K else 'zero_retained_error_undefined')
        for row,cov in zip(detail,source_cov):row['retained_source_weight']=scalar(ratio(cov,total))
    else:
        result.update({k:np.nan for k in ['retained_n','retained_errors','coverage_balanced','coverage_pooled','conditional_error_balanced','conditional_error_pooled','positive_discarded','positive_discarded_fraction_balanced','positive_discarded_fraction_pooled','worst_source_error','zero_retained_sources']})
        result['outcome_status']='policy_unavailable' if not available else 'empty_evaluation_source'
        for row in detail:row['retained_source_weight']=np.nan
    return result,detail

def reproduce_original(rows,out):
    original={(r['held_source'],r['model']):r for r in read(ROOT/'inputs/original_policies.tsv')}
    checks=[]
    for held in SOURCES:
        target=[r for r in rows if r['dataset']==held];forbidden={r['global_group'] for r in target}
        dev=[r for r in rows if r['dataset']!=held and r['global_group'] not in forbidden]
        sources=[s for s in SOURCES if s!=held]
        for model in MODELS[:-1]:
            got=choose(curve(dev,model,sources),'mixture_original');old=original[held,model]
            same_status=got['policy_status']==old['policy_status']
            same_threshold=(not np.isfinite(got['threshold']) and not np.isfinite(float(old['threshold']))) or math.isclose(got['threshold'],float(old['threshold']),rel_tol=1e-11,abs_tol=1e-12)
            good=same_status and same_threshold
            for metric,key in [('selection_coverage','development_coverage'),('selection_error','development_error')]:
                if got['policy_status']=='feasible':good&=math.isclose(got[metric],float(old[key]),rel_tol=1e-10,abs_tol=1e-12)
            checks.append(dict(held_source=held,model=model,old_status=old['policy_status'],new_status=got['policy_status'],old_threshold=old['threshold'],new_threshold=got['threshold'],passed=bool(good)))
    write(out/'original_policy_reproduction.tsv',checks)
    if not all(r['passed'] for r in checks):raise RuntimeError('Original exact policy reproduction failed; stop before interpreting new analyses')
    assert sum(r['new_status']=='feasible' for r in checks)==23
    print('ORIGINAL EXACT POLICIES PASS 30; feasible 23',flush=True)

def run(out,repeats=20,smoke=False):
    if not smoke and repeats!=20:raise ValueError('Formal protocol fixes 20 repeated group partitions')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=load_historical();reproduce_original(rows,out)
    selected_rows=[];eval_rows=[];source_rows=[];audits=[];paired=[];unavailable=[]
    with gzip.open(out/'group_partitions.tsv.gz','wt',encoding='utf-8',newline='') as f:
        group_writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        group_writer.writerow(['held_source','repeat','global_group','partition'])
        for fi,held in enumerate(SOURCES):
            held_rows=[r for r in rows if r['dataset']==held];forbidden={r['global_group'] for r in held_rows}
            before=[r for r in rows if r['dataset']!=held]
            dev=[r for r in before if r['global_group'] not in forbidden]
            sources=[s for s in SOURCES if s!=held]
            groups=sorted({r['global_group'] for r in dev})
            assert not forbidden&set(groups)
            assert not {r['sequence_id'] for r in dev}&{r['sequence_id'] for r in held_rows}
            for rep in range(repeats):
                rng=np.random.default_rng(2026102602+fi*100000+rep)
                assignment=rng.random(len(groups))<.7
                fit_groups={g for g,a in zip(groups,assignment) if a};val_groups=set(groups)-fit_groups
                fit=[r for r in dev if r['global_group'] in fit_groups];val=[r for r in dev if r['global_group'] in val_groups]
                assert not fit_groups&val_groups
                exact_overlap=len({r['sequence_id'] for r in fit}&{r['sequence_id'] for r in val})
                assert exact_overlap==0
                for g,a in zip(groups,assignment):group_writer.writerow([held,rep+1,g,'selection' if a else 'same_source_validation'])
                audits.append(dict(held_source=held,repeat=rep+1,development_before_purge=len(before),purged_records=len(before)-len(dev),selection_n=len(fit),validation_n=len(val),held_n=len(held_rows),selection_groups=len(fit_groups),validation_groups=len(val_groups),held_groups=len(forbidden),group_overlap_selection_validation=0,exact_overlap_selection_validation=exact_overlap,group_overlap_development_held=0,exact_overlap_development_held=0))
                for model in MODELS[:-1]:
                    c=curve(fit,model,sources);this={}
                    for policy in POLICIES:
                        selection=choose(c,policy)
                        key=dict(held_source=held,repeat=rep+1,model=model,policy=policy,smoke_only=smoke)
                        selected_rows.append(dict(**key,**selection))
                        for phase,rr,ss in [('selection',fit,sources),('same_source_validation',val,sources),('held_source',held_rows,[held])]:
                            outcome,detail=evaluate(rr,model,selection['threshold'],ss)
                            result=dict(**key,phase=phase,threshold=selection['threshold'],policy_status=selection['policy_status'],**outcome)
                            eval_rows.append(result);this[policy,phase]=result
                            for d in detail:source_rows.append(dict(**key,phase=phase,threshold=selection['threshold'],policy_status=selection['policy_status'],**d))
                    for phase in ['same_source_validation','held_source']:
                        for a,b in [('mixture_source_support','mixture_original'),('per_source_supported','mixture_source_support'),('per_source_supported','mixture_original')]:
                            x,y=this[a,phase],this[b,phase]
                            both=x['policy_status']=='feasible' and y['policy_status']=='feasible'
                            for met in ['coverage_balanced','conditional_error_balanced','positive_discarded_fraction_balanced','worst_source_error']:
                                paired.append(dict(held_source=held,repeat=rep+1,model=model,phase=phase,policy_a=a,policy_b=b,metric=met,difference=x[met]-y[met] if both else np.nan,status='paired_feasible' if both else 'at_least_one_policy_unavailable',smoke_only=smoke))
                for policy in POLICIES:
                    unavailable.append(dict(held_source=held,repeat=rep+1,model='c_AMPs',policy=policy,policy_status='score_ineligible',reason='no_verified_native_continuous_ensemble_probability; minimum-branch_extension_not_used_for_q',smoke_only=smoke))
                print('HONEST',held,'split',rep+1,'/',repeats,'fit',len(fit),'validation',len(val),flush=True)
    selection_ix={(r['held_source'],r['repeat'],r['model'],r['policy']):r for r in selected_rows}
    summaries=[]
    for held in SOURCES:
        for model in MODELS[:-1]:
            for policy in POLICIES:
                ss=[r for r in selected_rows if r['held_source']==held and r['model']==model and r['policy']==policy]
                for phase in ['same_source_validation','held_source']:
                    rr=[r for r in eval_rows if r['held_source']==held and r['model']==model and r['policy']==policy and r['phase']==phase]
                    summary=dict(held_source=held,model=model,policy=policy,phase=phase,repeats=repeats,feasible_splits=sum(r['policy_status']=='feasible' for r in ss),feasible_fraction=sum(r['policy_status']=='feasible' for r in ss)/repeats,summary_type='repeated_partition_sensitivity_NOT_confidence_interval',smoke_only=smoke)
                    for name,values in [('threshold',[r['threshold'] for r in ss])]+[(met,[r[met] for r in rr]) for met in ['coverage_balanced','conditional_error_balanced','positive_discarded_fraction_balanced','worst_source_error']]:
                        finite=np.array(values,float);finite=finite[np.isfinite(finite)]
                        summary[name+'_valid_splits']=len(finite)
                        for suffix,q in [('min',0),('q025',.025),('median',.5),('q975',.975),('max',1)]:summary[name+'_'+suffix]=float(np.quantile(finite,q)) if len(finite) else np.nan
                    summaries.append(summary)
    write(out/'selected_rules.tsv',selected_rows)
    write(out/'evaluation_points.tsv',eval_rows)
    write(out/'evaluation_by_source.tsv',source_rows)
    write(out/'group_and_exact_overlap_audit.tsv',audits)
    write(out/'paired_strategy_differences.tsv',paired)
    write(out/'repeated_split_sensitivity_NOT_CI.tsv',summaries)
    write(out/'score_ineligible.tsv',unavailable)
    dump(out/'run_settings.json',dict(analysis='honest_transfer_v1',repeats=repeats,smoke_only=smoke,protocol_sha256=sha(ROOT/'protocol_v1.json'),input_manifest_sha256=sha(ROOT/'input_manifest.json'),new_model_inference=False,original_rule_reproduction='30/30 match, 23 feasible',leakage='group and exact-string overlap checked; validation/held labels not passed into curve/choose',statistical_scope='finite observed conditional errors; repeated-split quantiles are sensitivity, not CIs or future-risk guarantees',constraints=protocol()['policies']))
    return dict(selected_rules=len(selected_rows),evaluation_rows=len(eval_rows),repeats=repeats,smoke_only=smoke)
