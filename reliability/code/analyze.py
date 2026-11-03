"""Linked retrospective error diagnosis and group-separated retention utility.

All models, contrasts, targets and inference restrictions are frozen in protocol.json.
No AMP model inference, retraining, wet-lab experiment, or manuscript overwrite.
"""
from pathlib import Path
import sys,json,csv,gzip,math,time,hashlib,argparse
import numpy as np
import legacy_common as C

ROOT=Path(__file__).resolve().parents[1]
MODELS=C.MODELS[:-1]
METHODS=['Decision_confidence','Inventory_similarity','Development_similarity','Logistic_C_L','Logistic_C_L_I','Logistic_C_L_D','Logistic_C_L_I_D','Random_expected']
TARGETS=[.2,.4,.6,.8,1.]
CONTRASTS=[('Logistic_C_L_I','Logistic_C_L'),('Logistic_C_L_D','Logistic_C_L'),('Logistic_C_L_I_D','Logistic_C_L'),('Inventory_similarity','Decision_confidence'),('Development_similarity','Decision_confidence')]
METRICS=['conditional_error','positive_rejection','end_to_end_detection']

def dump(p,x):C.dump(p,x)
def write(p,rows):C.write(p,rows)
def confidence(p,d):return d*p+(1-d)*(1-p)
def sigmoid(z):return np.exp(-np.logaddexp(0.,-np.asarray(z)))

def fit_selector(X,y,w):
    """Fixed ridge logistic regression of error; no validation/test labels accepted."""
    X=np.asarray(X,float);y=np.asarray(y,float);w=np.asarray(w,float);w=w/w.sum()
    mean=(w[:,None]*X).sum(axis=0);sd=np.sqrt((w[:,None]*(X-mean)**2).sum(axis=0));sd=np.where(sd>1e-10,sd,1.)
    Z=np.column_stack([np.ones(len(X)),(X-mean)/sd]);P=len(mean)
    b=np.zeros(P+1);pre=float(w@y);b[0]=math.log(np.clip(pre,1e-8,1-1e-8)/(1-np.clip(pre,1e-8,1-1e-8)))
    status='converged';iterations=0
    if len(np.unique(y))==1:status='single_error_class_constant'
    else:
        ridge=np.diag([0.]+[.01]*P)
        def objective(beta):
            z=Z@beta;return float(w@(np.logaddexp(0,z)-y*z)+.5*beta@ridge@beta)
        old=objective(b);status='maximum_iterations'
        for iterations in range(1,61):
            p=sigmoid(Z@b);grad=Z.T@(w*(p-y))+ridge@b
            h=Z.T@((w*p*(1-p))[:,None]*Z)+ridge+np.eye(P+1)*1e-12
            step=np.linalg.solve(h,grad);scale=1.
            for _ in range(30):
                trial=b-scale*step;loss=objective(trial)
                if loss<=old+1e-14:break
                scale*=.5
            b=trial
            if np.max(np.abs(scale*step))<1e-7 or abs(old-loss)<1e-11:status='converged';break
            old=loss
    return dict(mean=mean,sd=sd,coef=b,status=status,iterations=iterations,error_prevalence=pre)

def predict_selector(fit,X):
    Z=np.column_stack([np.ones(len(X)),(np.asarray(X)-fit['mean'])/fit['sd']])
    return sigmoid(Z@fit['coef'])

def metrics(W,S,y,pred):
    W=np.atleast_2d(W);S=np.atleast_2d(S);y=np.asarray(y,int);pred=np.asarray(pred,int)
    n=W.sum(1);k=S.sum(1);P=W@y;ke=S@(pred!=y);kp=S@y;tp=S@(y*(pred==1))
    return dict(conditional_error=C.ratio(ke,k),positive_rejection=C.ratio(P-kp,P),end_to_end_detection=C.ratio(tp,P))

def selected(W,score,target):
    if score is None:
        W=np.atleast_2d(W);f=np.ceil(target*W.sum(1)-1e-10)/W.sum(1)
        return W*f[:,None]
    return C.selected(W,score,target)[0]

def point(y,pred,scores,target):
    W=np.ones((1,len(y)));out={}
    for method,score in scores.items():out[method]={k:float(v[0]) for k,v in metrics(W,selected(W,score,target),y,pred).items()}
    return out

def retention_plan(score,y,pred):
    if score is None:return None
    order=np.argsort(-np.asarray(score),kind='stable');s=np.asarray(score)[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];ends=np.r_[starts[1:],len(order)]
    tier=np.repeat(np.arange(len(starts)),ends-starts)
    yy=np.asarray(y)[order];pp=np.asarray(pred)[order]
    cats=[np.flatnonzero((yy==a)&(pp==b)) for a,b in [(1,1),(1,0),(0,1),(0,0)]]
    return order,starts[tier],ends[tier],cats

def fast_retention_metrics(W,y,pred,plan,target):
    """Compute exact fractional-tie selection from cumulative class masses.

    Independent of legacy selected(): no dense selection matrix is constructed.
    """
    W=np.atleast_2d(W);n=W.sum(1);K=np.ceil(target*n-1e-10);P=W@np.asarray(y)
    if plan is None:
        f=K/n;tp=f*(W@(np.asarray(y)*(np.asarray(pred)==1)));err=f*(W@(np.asarray(pred)!=np.asarray(y)))
        return dict(conditional_error=C.ratio(err,K),positive_rejection=C.ratio(P*(1-f),P),end_to_end_detection=C.ratio(tp,P))
    order,starts,ends,cats=plan;cum=np.cumsum(W[:,order],axis=1);B=len(W);rr=np.arange(B)
    boundary=(cum<K[:,None]).sum(1);lo=starts[boundary];hi=ends[boundary]
    before=np.where(lo>0,cum[rr,np.maximum(lo-1,0)],0.);total=cum[rr,hi-1]-before;fraction=(K-before)/total
    selected_classes=[]
    for pos in cats:
        if not len(pos):selected_classes.append(np.zeros(B));continue
        cc=np.cumsum(W[:,order[pos]],axis=1)
        low=np.searchsorted(pos,lo,side='left');high=np.searchsorted(pos,hi,side='left')
        left=np.where(low>0,cc[rr,np.maximum(low-1,0)],0.);right=np.where(high>0,cc[rr,np.maximum(high-1,0)],0.)
        selected_classes.append(left+fraction*(right-left))
    tp,fn,fp,tn=selected_classes
    assert np.allclose(tp+fn+fp+tn,K,atol=1e-8)
    return dict(conditional_error=C.ratio(fn+fp,K),positive_rejection=C.ratio(P-tp-fn,P),end_to_end_detection=C.ratio(tp,P))

def bootstrap(y,pred,groups,scores,contrasts,seed,target=.8):
    """Selectors fixed, group resampling paired, retention recalculated each draw."""
    unique,gi=np.unique(groups,return_inverse=True);G=len(unique);B=10000
    draws={m:{k:[] for k in METRICS} for m in scores};plans={m:retention_plan(s,y,pred) for m,s in scores.items()}
    rng=np.random.default_rng(seed)
    for start in range(0,B,100):
        D=rng.multinomial(G,np.ones(G)/G,size=min(100,B-start));W=D[:,gi].astype(float)
        for method,score in scores.items():
            vv=fast_retention_metrics(W,y,pred,plans[method],target)
            for k in METRICS:draws[method][k].append(vv[k])
    draws={m:{k:np.concatenate(v) for k,v in values.items()} for m,values in draws.items()}
    points=point(y,pred,scores,target);out=[]
    for a,b in contrasts:
        for metric in METRICS:
            values=draws[a][metric]-draws[b][metric];lo,hi,status,frac=C.interval(values,B,G)
            out.append(dict(method_a=a,method_b=b,metric=metric,coverage_target=target,estimate=points[a][metric]-points[b][metric],ci_low=lo,ci_high=hi,interval_status=status,valid_fraction=frac,bootstrap=B,contributing_groups=G,seed=seed,selector_refit=False))
    return out

def quantile(v,w,q):
    order=np.argsort(v,kind='stable');v=v[order];w=w[order];return float(v[np.searchsorted(np.cumsum(w)/w.sum(),q,side='left')])

def load():
    for name,item in json.loads((ROOT/'input_manifest.json').read_text(encoding='utf8')).items():
        assert hashlib.sha256((ROOT/'inputs'/name).read_bytes()).hexdigest()==item['sha256'],name
    h=C.load_historical();hist=json.loads((ROOT/'cache/historical_distance_index.json').read_text(encoding='utf8'))
    ids=[r['sequence_id'] for r in hist];ix={x:i for i,x in enumerate(ids)}
    ref=C.read(ROOT/'results/inventory_nearest_reference.tsv.gz');inv={(r['sequence_id'],r['model']):float(r['normalized_edit_distance']) for r in ref}
    return h,hist,ix,inv

def run_historical(resume=False):
    h,hist,ix,inv=load();N=len(hist);points=[];coeff=[];diag=[];intervals=[];audit=[];repeat_differences=[];auc_rows=[];record_rows=[]
    finished=set()
    if resume and (ROOT/'qa/analysis_split_audit.tsv').exists():
        audit=C.read(ROOT/'qa/analysis_split_audit.tsv')
        for held in C.SOURCES:
            reps={int(r['repeat']) for r in audit if r['held_source']==held}
            if reps==set(range(1,21)):finished.add(held)
        points=C.read(ROOT/'results/historical_retention_points.tsv');intervals=C.read(ROOT/'results/historical_paired_intervals_80.tsv');diag=C.read(ROOT/'results/diagnostic_cells.tsv');coeff=C.read(ROOT/'results/selector_coefficients.tsv');repeat_differences=C.read(ROOT/'results/repeated_partition_contrasts.tsv')
        print('RESUME COMPLETED FOLDS',sorted(finished),flush=True)
    record_uid=np.array([ix[r['sequence_id']] for r in h]);group=np.array([r['global_group'] for r in h]);source=np.array([r['dataset'] for r in h]);label=np.array([int(r['label']) for r in h]);L=np.log(np.array([C.sequence_length(r) for r in h]))
    histgroup=np.array([r['global_group'] for r in hist]);score_audit=[]
    for fi,held in enumerate(C.SOURCES):
        if held in finished:continue
        hd=np.flatnonzero(source==held);forbidden=set(group[hd]);dev=(source!=held)&~np.isin(group,list(forbidden))
        with np.load(ROOT/'cache'/f'development_{fi}.npz') as archive:ds={key:archive[key] for key in archive.files}
        for rep in range(20):
            sel=dev&ds['selection'][rep,record_uid];val=dev&~ds['selection'][rep,record_uid]
            assert not set(group[sel])&set(group[val]);assert not set(group[sel|val])&forbidden
            # Confirm exact original group assignment without using labels.
            dg=sorted(set(group[dev]));assign=np.random.default_rng(2026102602+fi*100000+rep).random(len(dg))<.7
            assert set(group[sel])=={g for g,a in zip(dg,assign) if a}
            fit_weights=np.zeros(len(h))
            for s in C.SOURCES:
                mask=sel&(source==s)
                if mask.any():fit_weights[mask]=1./mask.sum()
            fit_weights/=fit_weights.sum()
            D=ds['distance'][rep,record_uid].astype(float)
            for mi,model in enumerate(MODELS):
                p=np.array([float(r[model+'__score']) for r in h]);pred=np.array([int(r[model+'__prediction']) for r in h]);err=pred!=label
                c=confidence(p,pred);q=np.maximum(p,1-p);I=np.array([inv[r['sequence_id'],model] for r in h]);clipped=np.clip(c,1e-6,1-1e-6);logit=np.log(clipped/(1-clipped))
                X=np.column_stack([logit,L,I,D]);ss={'Decision_confidence':c,'Inventory_similarity':-I,'Development_similarity':-D}
                fits={}
                for method,cols in [('Logistic_C_L',[0,1]),('Logistic_C_L_I',[0,1,2]),('Logistic_C_L_D',[0,1,3]),('Logistic_C_L_I_D',[0,1,2,3])]:
                    fit=fit_selector(X[sel][:,cols],err[sel],fit_weights[sel]);assert fit['status']!='maximum_iterations',(held,rep,model,method)
                    ss[method]=-predict_selector(fit,X[:,cols]);fits[method]=fit
                    coeff.append(dict(held_source=held,repeat=rep+1,model=model,method=method,features=json.dumps(cols),means=json.dumps(fit['mean'].tolist()),scales=json.dumps(fit['sd'].tolist()),coefficients=json.dumps(fit['coef'].tolist()),status=fit['status'],iterations=fit['iterations'],selection_error_prevalence=fit['error_prevalence']))
                ss['Random_expected']=None
                for phase,mask in [('held_source',source==held),('same_source_validation',val)]:
                    for dataset in ([held] if phase=='held_source' else [s for s in C.SOURCES if s!=held]):
                        jj=np.flatnonzero(mask&(source==dataset))
                        if not len(jj):continue
                        these={m:(None if v is None else v[jj]) for m,v in ss.items()}
                        for target in TARGETS:
                            pt=point(label[jj],pred[jj],these,target)
                            for method,values in pt.items():points.append(dict(held_source=held,repeat=rep+1,phase=phase,dataset=dataset,model=model,method=method,coverage_target=target,actual_coverage=math.ceil(target*len(jj)-1e-10)/len(jj),n=len(jj),positive=int(label[jj].sum()),groups=len(set(group[jj])),**values))
                            if phase=='held_source':
                                for a,b in CONTRASTS:
                                    for met in METRICS:repeat_differences.append(dict(held_source=held,repeat=rep+1,model=model,coverage_target=target,method_a=a,method_b=b,metric=met,difference=pt[a][met]-pt[b][met],summary_role='repeated_development_partition_sensitivity_NOT_CI'))
                if rep==0:
                    score_audit.append(dict(held_source=held,model=model,q_vs_native_confidence_different_n=int(np.sum(np.abs(q[hd]-c[hd])>1e-12)),high_q_but_low_native_confidence_n=int(np.sum((q[hd]>=.95)&(c[hd]<.95)))))
                    for role,dist in [('inventory',I),('selection_development',D)]:
                        cuts=[quantile(dist[sel],fit_weights[sel],x) for x in [1/3,2/3]]
                        dbin=np.searchsorted(cuts,dist[hd],side='left');cb=np.searchsorted([.8,.95],c[hd],side='right')
                        for d in range(3):
                            for confbin in range(3):
                                jj=hd[(dbin==d)&(cb==confbin)];y=label[jj];pr=pred[jj]
                                diag.append(dict(held_source=held,model=model,reference_role=role,distance_tier=d,confidence_tier=confbin,distance_cut1=cuts[0],distance_cut2=cuts[1],n=len(jj),positive=int(y.sum()),negative=int(len(y)-y.sum()),groups=len(set(group[jj])),errors=int(np.sum(y!=pr)),false_positive=int(np.sum((y==0)&(pr==1))),false_negative=int(np.sum((y==1)&(pr==0))),error_rate=float(np.mean(y!=pr)) if len(jj) else np.nan,interval_status='point_diagnostic_counts_no_sampling_interval'))
                    for method,sv in ss.items():
                        if sv is None:continue
                        auc=float(C.auc(np.ones((1,len(hd))),err[hd],-sv[hd])[0])
                        auc_rows.append(dict(held_source=held,repeat=1,model=model,method=method,n=len(hd),errors=int(err[hd].sum()),error_prevalence=float(err[hd].mean()),error_detection_AUROC=auc,status='two_error_classes' if len(np.unique(err[hd]))==2 else 'single_error_class_undefined'))
                    for j in hd:
                        row=dict(dataset=held,sequence_id=h[j]['sequence_id'],global_group=h[j]['global_group'],model=model,label=int(label[j]),native_prediction=int(pred[j]),saved_score=p[j],native_decision_confidence=c[j],q=q[j],error=int(err[j]),length=C.sequence_length(h[j]),inventory_distance=I[j],selection_development_distance=D[j],nearest_development_sequence_id=hist[int(ds['nearest'][0,record_uid[j]])]['sequence_id'],primary_repeat=1)
                        row.update({m:float(sv[j]) for m,sv in ss.items() if sv is not None});record_rows.append(row)
                    these={m:(None if v is None else v[hd]) for m,v in ss.items()}
                    ci=bootstrap(label[hd],pred[hd],group[hd],these,CONTRASTS,2026110201+fi*100)
                    intervals.extend(dict(held_source=held,repeat=1,model=model,n=len(hd),positive=int(label[hd].sum()),**r) for r in ci)
            audit.append(dict(held_source=held,repeat=rep+1,selection_n=int(sel.sum()),validation_n=int(val.sum()),held_n=len(hd),selection_groups=len(set(group[sel])),validation_groups=len(set(group[val])),held_groups=len(forbidden),original_partition_reproduced=True,all_reference_neighbors_selection_only=True,no_query_group_reference=True,group_or_exact_overlap=False))
            print('FORMAL RETENTION',held,rep+1,'/20',flush=True)
        # Incremental checkpoints to preserve completed folds.
        write(ROOT/'results/historical_retention_points.tsv',points)
        write(ROOT/'results/historical_paired_intervals_80.tsv',intervals)
        write(ROOT/'results/diagnostic_cells.tsv',diag)
        write(ROOT/'results/selector_coefficients.tsv',coeff)
        write(ROOT/'results/repeated_partition_contrasts.tsv',repeat_differences)
        write(ROOT/'qa/analysis_split_audit.tsv',audit)
    # Reconstruct the repeat-1 row table from saved coefficients, including any
    # resumed folds; neither refitting nor test-label-dependent score choice.
    record_rows=[];auc_rows=[];score_audit=[]
    for fi,held in enumerate(C.SOURCES):
        hd=np.flatnonzero(source==held)
        with np.load(ROOT/'cache'/f'development_{fi}.npz') as archive:ds={key:archive[key] for key in archive.files}
        D=ds['distance'][0,record_uid].astype(float)
        for model in MODELS:
            p=np.array([float(r[model+'__score']) for r in h]);pr=np.array([int(r[model+'__prediction']) for r in h]);c=confidence(p,pr);q=np.maximum(p,1-p);I=np.array([inv[r['sequence_id'],model] for r in h]);z=np.clip(c,1e-6,1-1e-6);X=np.column_stack([np.log(z/(1-z)),L,I,D]);ss={'Decision_confidence':c,'Inventory_similarity':-I,'Development_similarity':-D}
            for method,cols in [('Logistic_C_L',[0,1]),('Logistic_C_L_I',[0,1,2]),('Logistic_C_L_D',[0,1,3]),('Logistic_C_L_I_D',[0,1,2,3])]:
                fitrow=[r for r in coeff if r['held_source']==held and int(r['repeat'])==1 and r['model']==model and r['method']==method];assert len(fitrow)==1;fr=fitrow[0];fit={k:np.array(json.loads(fr[v])) for k,v in [('mean','means'),('sd','scales'),('coef','coefficients')]};ss[method]=-predict_selector(fit,X[:,cols])
            for j in hd:
                row=dict(dataset=held,sequence_id=h[j]['sequence_id'],global_group=h[j]['global_group'],model=model,label=int(label[j]),native_prediction=int(pr[j]),saved_score=p[j],native_decision_confidence=c[j],q=q[j],error=int(pr[j]!=label[j]),length=C.sequence_length(h[j]),inventory_distance=I[j],selection_development_distance=D[j],nearest_development_sequence_id=hist[int(ds['nearest'][0,record_uid[j]])]['sequence_id'],primary_repeat=1)
                row.update({m:float(v[j]) for m,v in ss.items()});record_rows.append(row)
            score_audit.append(dict(held_source=held,model=model,q_vs_native_confidence_different_n=int(np.sum(np.abs(q[hd]-c[hd])>1e-12)),high_q_but_low_native_confidence_n=int(np.sum((q[hd]>=.95)&(c[hd]<.95)))))
            err=pr[hd]!=label[hd]
            for method,sv in ss.items():auc_rows.append(dict(held_source=held,repeat=1,model=model,method=method,n=len(hd),errors=int(err.sum()),error_prevalence=float(err.mean()),error_detection_AUROC=float(C.auc(np.ones((1,len(hd))),err,-sv[hd])[0]),status='two_error_classes' if len(np.unique(err))==2 else 'single_error_class_undefined'))
    write(ROOT/'results/error_detection_AUROC.tsv',auc_rows);write(ROOT/'qa/native_confidence_vs_q.tsv',score_audit)
    with gzip.open(ROOT/'results/historical_record_diagnostics_repeat1.tsv.gz','wt',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(record_rows[0]),delimiter='\t');writer.writeheader();writer.writerows(record_rows)
    sens=[]
    for held in C.SOURCES:
        for model in MODELS:
            for a,b in CONTRASTS:
                for met in METRICS:
                    rr=[float(r['difference']) for r in repeat_differences if r['held_source']==held and r['model']==model and r['method_a']==a and r['method_b']==b and r['metric']==met and float(r['coverage_target'])==.8]
                    v=np.array(rr,float);assert len(v)==20
                    sens.append(dict(held_source=held,model=model,method_a=a,method_b=b,metric=met,coverage_target=.8,repeats=20,minimum=float(np.min(v)),median=float(np.median(v)),maximum=float(np.max(v)),negative_count=int(np.sum(v<-1e-12)),zero_count=int(np.sum(np.abs(v)<=1e-12)),positive_count=int(np.sum(v>1e-12)),summary_role='NOT_CI_reused_held_records'))
    write(ROOT/'results/repeated_partition_sensitivity_80_NOT_CI.tsv',sens)

def run_experimental():
    panel,si,tasks=C.load_panel();sub={r['sequence_id']:r for r in C.read(ROOT/'inputs/amplify_disagreement.tsv')}
    inv={(r['sequence_id'],r['model']):float(r['normalized_edit_distance']) for r in C.read(ROOT/'results/inventory_nearest_reference.tsv.gz')}
    curves=[];intervals=[];records=[];counts=[]
    for ci,cohort in enumerate(C.COHORTS):
        rr=[r for r in panel if r['cohort']==cohort];ids=[r['sequence_id'] for r in rr]
        pred=np.array([int(si[x,'AMPlify']['prediction']) for x in ids]);p=np.array([float(si[x,'AMPlify']['score']) for x in ids]);conf=confidence(p,pred)
        sc={'Decision_confidence':conf,'Inventory_similarity':-np.array([inv[x,'AMPlify'] for x in ids]),'Submodel_SD':-np.array([float(sub[x]['submodel_sd_population']) for x in ids]),'Entropy_gap':-np.array([float(sub[x]['entropy_gap']) for x in ids]),'Random_expected':None}
        groups=np.array([r['block'] for r in rr]);G=len(set(groups))
        for ti,task in enumerate(['any_eligible_measured_strain','source_reference_Ecoli']):
            y=np.array([int(tasks[cohort,task,r['record_id']]['label']) for r in rr]);counts.append(dict(cohort=cohort,task=task,n=len(rr),positive=int(y.sum()),threshold_negative=len(y)-int(y.sum()),blocks=G,model='AMPlify',errors=int(np.sum(pred!=y))))
            for target in TARGETS:
                pp=point(y,pred,sc,target)
                for method,values in pp.items():curves.append(dict(cohort=cohort,task=task,model='AMPlify',method=method,coverage_target=target,n=len(rr),positive=int(y.sum()),blocks=G,**values))
            cs=[(m,'Decision_confidence') for m in sc if m not in ['Decision_confidence','Random_expected']]
            if G>=5:
                vv=bootstrap(y,pred,groups,sc,cs,2026111201+ci*100+ti)
            else:
                pp=point(y,pred,sc,.8);vv=[dict(method_a=a,method_b=b,metric=met,coverage_target=.8,estimate=pp[a][met]-pp[b][met],ci_low=np.nan,ci_high=np.nan,interval_status='fewer_than_five_contributing_blocks',valid_fraction=np.nan,bootstrap=0,contributing_groups=G,seed='',selector_refit=False) for a,b in cs for met in METRICS]
            intervals.extend(dict(cohort=cohort,task=task,model='AMPlify',n=len(rr),positive=int(y.sum()),**v) for v in vv)
            for i,r in enumerate(rr):
                records.append(dict(cohort=cohort,task=task,record_id=r['record_id'],sequence_id=r['sequence_id'],block=r['block'],length=C.sequence_length(r),activity_label=int(y[i]),native_prediction=int(pred[i]),error=int(pred[i]!=y[i]),confidence=float(conf[i]),inventory_distance=inv[r['sequence_id'],'AMPlify'],submodel_sd=float(sub[ids[i]]['submodel_sd_population']),entropy_gap=float(sub[ids[i]]['entropy_gap'])))
        print('EXPERIMENTAL UTILITY',cohort,len(rr),'blocks',G,flush=True)
    write(ROOT/'results/experimental_retention_points.tsv',curves);write(ROOT/'results/experimental_paired_intervals_80.tsv',intervals);write(ROOT/'results/experimental_record_diagnostics.tsv',records);write(ROOT/'results/experimental_task_denominators.tsv',counts)

def selftest():
    # Known tie expectation, class-specific denominators and constant selection.
    y=np.array([1,0,1,0]);pr=np.array([1,1,0,0]);W=np.ones((1,4));s=np.ones(4)
    S=selected(W,s,.5);assert np.allclose(S,.5)
    mm=metrics(W,S,y,pr);assert np.allclose([mm[k][0] for k in METRICS],[.5,.5,.25])
    # Weighted group duplication must preserve selection capacity.
    W=np.array([[2,2,0,1],[0,0,3,3]],float);S=selected(W,np.array([.9,.9,.2,.1]),.8);assert np.allclose(S.sum(1),np.ceil(.8*W.sum(1)))
    # A known increasing-error signal must give increasing fitted risk.
    X=np.arange(40.)[:,None];y=np.r_[np.zeros(20),np.ones(20)];f=fit_selector(X,y,np.ones(40));p=predict_selector(f,X);assert (np.diff(p)>0).all() and p[-1]>p[0]
    # Constant-feature / single-error-class cases remain explicit and finite.
    f=fit_selector(np.ones((20,2)),np.zeros(20),np.ones(20));assert f['status']=='single_error_class_constant';assert np.isfinite(predict_selector(f,np.ones((5,2)))).all()
    # Finite difference check for logistic derivative, independent of Newton loop.
    rng=np.random.default_rng(1);Z=np.c_[np.ones(12),rng.normal(size=(12,3))];y=rng.integers(0,2,12);b=rng.normal(size=4);w=np.ones(12)/12;ridge=np.diag([0,.01,.01,.01])
    def obj(bb):z=Z@bb;return w@(np.logaddexp(0,z)-y*z)+.5*bb@ridge@bb
    grad=Z.T@(w*(sigmoid(Z@b)-y))+ridge@b;num=np.array([(obj(b+np.eye(4)[j]*1e-6)-obj(b-np.eye(4)[j]*1e-6))/2e-6 for j in range(4)]);assert np.allclose(grad,num,atol=1e-7)
    # Exact independent prefix-mass implementation versus the legacy dense
    # selection matrix under ties, class imbalance and resampled multiplicity.
    for n in [4,17,100]:
        y=rng.integers(0,2,n);pr=rng.integers(0,2,n);W=rng.multinomial(n,np.ones(n)/n,size=13).astype(float)
        for score in [None,rng.integers(0,5,n).astype(float),rng.normal(size=n)]:
            plan=retention_plan(score,y,pr)
            for target in TARGETS:
                expected=metrics(W,selected(W,score,target),y,pr);actual=fast_retention_metrics(W,y,pr,plan,target)
                for key in METRICS:assert np.allclose(expected[key],actual[key],atol=1e-12,equal_nan=True),(n,target,key)
    dump(ROOT/'qa/statistical_selftests.json',dict(passed=True,checks=['tie allocation','weighted resampling capacity','positive denominators','logistic known-signal direction','constant-feature/single-class handling','independent numerical gradient','independent prefix-class versus dense retention under ties/resampling']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['test','historical','experimental','all']);ap.add_argument('--resume',action='store_true');a=ap.parse_args();selftest()
    if a.action in ['historical','all']:run_historical(a.resume)
    if a.action in ['experimental','all']:run_experimental()
    if a.action=='all':dump(ROOT/'results/ANALYSIS_COMPLETE.json',dict(bootstrap=10000,repeats=20,new_inference=False,wet_lab=False,protocol_sha256=C.sha(ROOT/'protocol.json'),primary_interval_scope='conditional on frozen first-partition development fit; selection repeated within held-group bootstrap',all_methods_retained=True,word_modified=False))
