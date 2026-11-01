"""Unfitted sequence controls and within-family ranking, frozen 151 members."""
from collections import defaultdict
import itertools
import math
import numpy as np
from common import *

def run(out,bootstrap=10000,smoke=False):
    if not smoke and bootstrap<10000:raise ValueError('Formal baseline analysis requires >=10000 resamples')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    members,si,task_index=load_panel()
    points=[];intervals=[];pairs=[];family_rows=[];roster=[];description=[];availability=[]
    for r in members:
        description.append(dict(record_id=r['record_id'],sequence_id=r['sequence_id'],sequence=r['sequence'],cohort=r['cohort'],family=r['family'],block=r['block'],**descriptors(r['sequence'])))
    for ci,cohort in enumerate(COHORTS):
        cc=[r for r in members if r['cohort']==cohort]
        groups,gi=np.unique([r['block'] for r in cc],return_inverse=True);G=len(groups);N=len(cc)
        family_ix={f:np.array([r['family']==f for r in cc]) for f in sorted({r['family'] for r in cc})}
        family_group={f:np.unique(gi[ix]) for f,ix in family_ix.items()}
        assert all(len(v)==1 for v in family_group.values()),'Family must nest in frozen dependence block'
        score={m:np.array([float(si[r['sequence_id'],m]['score']) for r in cc]) for m in MODELS}
        for b in BASELINES:score[b]=np.array([descriptors(r['sequence'])[b] for r in cc])
        # Shared draws within cohort across methods AND the two fixed endpoints.
        rng=np.random.default_rng(2026102601+ci*100)
        D=rng.multinomial(G,np.ones(G)/G,size=bootstrap)
        BW=D[:,gi].astype(float)
        for task in ['any_eligible_measured_strain','source_reference_Ecoli']:
            labelrows=[task_index[cohort,task,r['record_id']] for r in cc]
            y=np.array([int(r['label']) for r in labelrows]);assert np.isin(y,[0,1]).all()
            P=int(y.sum());info=dict(cohort=cohort,task=task,threshold=float(labelrows[0]['threshold']),unit=labelrows[0]['unit'],reference_strain=labelrows[0]['reference_strain'],n=N,positive=P,threshold_negative=N-P,blocks=G,budget=.2)
            informative=[f for f,ix in family_ix.items() if y[ix].min()!=y[ix].max()]
            fgroups=np.array([int(family_group[f][0]) for f in informative],int)
            effective=len(np.unique(fgroups))
            availability.append(dict(**info,total_families=len(family_ix),informative_families=len(informative),informative_blocks=effective,within_family_status='estimable' if informative else 'no_family_with_both_endpoint_classes',pooled_AUROC_status='estimable' if 0<P<N else 'single_true_class'))
            for r,lab in zip(cc,y):roster.append(dict(**info,record_id=r['record_id'],sequence_id=r['sequence_id'],family=r['family'],block=r['block'],label=int(lab)))
            boot_values={};point_values={}
            for method,sc in score.items():
                W=np.ones((1,N));S,K=selected(W,sc,.2);metric=budget_metrics(W,S,y)
                metric['pooled_AUROC']=auc(W,y,sc)
                BS,BK=selected(BW,sc,.2);bm=budget_metrics(BW,BS,y);bm['pooled_AUROC']=auc(BW,y,sc)
                fvals=[]
                for family,ix in family_ix.items():
                    yy=y[ix];ss=sc[ix];FW=np.ones((1,int(ix.sum())))
                    FS,FK=selected(FW,ss,.2);fm=budget_metrics(FW,FS,yy)
                    fa=scalar(auc(FW,yy,ss))
                    status='both_classes' if np.unique(yy).size==2 else 'single_true_class'
                    family_rows.append(dict(**{k:v for k,v in info.items() if k not in ['n','positive','threshold_negative','blocks']},method=method,family=family,block=groups[int(family_group[family][0])],n=int(ix.sum()),positive=int(yy.sum()),threshold_negative=int(ix.sum()-yy.sum()),family_AUROC=fa,within_family_status=status,budget_slots=scalar(FK),**{k:scalar(v) for k,v in fm.items()}))
                    if family in informative:fvals.append(fa)
                fv=np.array(fvals,float)
                assert len(fv)==len(informative)
                metric['macro_within_family_AUROC']=np.array([np.mean(fv) if len(fv) else np.nan])
                bm['macro_within_family_AUROC']=ratio((D[:,fgroups]*fv).sum(axis=1),D[:,fgroups].sum(axis=1)) if len(fv) else np.full(bootstrap,np.nan)
                key=dict(**info,method=method,method_type='released_model' if method in MODELS else 'unfitted_single_feature_control',score_definition='supplementary_minimum_branch_ranking_extension' if method=='c_AMPs' else ('saved_AMP_class_score' if method in MODELS else 'prespecified_sequence_descriptor'),informative_families=len(informative),informative_blocks=effective,smoke_only=smoke)
                points.append(dict(**key,budget_slots=scalar(K),**{k:scalar(v) for k,v in metric.items()}))
                for met,v in bm.items():
                    cg=effective if met=='macro_within_family_AUROC' else G
                    lo,hi,st,frac=interval(v,bootstrap,cg)
                    if smoke:lo=hi=np.nan;st='smoke_not_for_inference'
                    intervals.append(dict(**key,metric=met,estimate=scalar(metric[met]),ci_low=lo,ci_high=hi,status=st,valid_fraction=frac,bootstrap=bootstrap,contributing_blocks=cg))
                boot_values[method]=bm;point_values[method]=metric
            # No best-baseline selection. Complete 6 x 6 paired contrasts.
            for model,baseline in itertools.product(MODELS,BASELINES):
                for met in ['positive_hits','excess_hits_vs_random','precision','pooled_AUROC','macro_within_family_AUROC']:
                    v=boot_values[model][met]-boot_values[baseline][met]
                    cg=effective if met=='macro_within_family_AUROC' else G
                    lo,hi,st,frac=interval(v,bootstrap,cg)
                    if smoke:lo=hi=np.nan;st='smoke_not_for_inference'
                    pairs.append(dict(**info,model=model,baseline=baseline,metric=met,difference=scalar(point_values[model][met]-point_values[baseline][met]),ci_low=lo,ci_high=hi,status=st,valid_fraction=frac,contributing_blocks=cg,bootstrap=bootstrap,multiplicity_adjusted=False,smoke_only=smoke))
            print('BASELINE/FAMILY',cohort,task,'n',N,'P',P,'informative families',len(informative),'blocks',effective,flush=True)
    frozen={(r['cohort'],r['model']):r for r in read(ROOT/'inputs/original_budget_points.tsv') if r['task']=='any_eligible_measured_strain' and r['policy']=='score_topK' and float(r['budget'])==.2 and r['subset']=='all_generated'}
    reproduction=[]
    for row in points:
        if row['task']!='any_eligible_measured_strain' or row['method'] not in MODELS:continue
        old=frozen[row['cohort'],row['method']]
        for met,oldkey in [('n','n'),('positive','positive'),('budget_slots','budget_slots'),('positive_hits','positive_hits'),('precision','precision_at_budget'),('recovery','positive_recovery'),('enrichment','enrichment'),('excess_hits_vs_random','excess_hits_vs_random')]:
            a,b=float(row[met]),float(old[oldkey]);good=math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))
            reproduction.append(dict(cohort=row['cohort'],model=row['method'],metric=met,frozen=a if math.isnan(b) else b,recomputed=a,passed=good))
    write(out/'original_budget_reproduction.tsv',reproduction)
    assert len(reproduction)==192 and all(x['passed'] for x in reproduction),'Existing 151-member budget reproduction failed'
    print('ORIGINAL 151 BUDGET PASS 192 comparisons',flush=True)
    write(out/'descriptor_values.tsv',description)
    write(out/'endpoint_membership.tsv',roster)
    write(out/'endpoint_and_family_availability.tsv',availability)
    write(out/'baseline_and_model_points.tsv',points)
    write(out/'baseline_and_model_intervals.tsv',intervals)
    write(out/'paired_model_minus_baseline.tsv',pairs)
    write(out/'within_family_details.tsv',family_rows)
    dump(out/'run_settings.json',dict(analysis='baseline_family_v1',bootstrap=bootstrap,smoke_only=smoke,protocol_sha256=sha(ROOT/'protocol_v1.json'),input_manifest_sha256=sha(ROOT/'input_manifest.json'),rankings=len(MODELS)+len(BASELINES),endpoint_sets=len(availability),new_model_inference=False,new_learned_baseline=False,interval_scope='paired pointwise exploratory intervals; no multiplicity-adjusted significance',family_scope='descriptive within tested families; no new-family or new-publication validation'))
    return {'points':len(points),'paired_contrasts':len(pairs),'family_rows':len(family_rows),'smoke_only':smoke}
