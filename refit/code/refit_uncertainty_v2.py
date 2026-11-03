"""Paired global-group development refitting and held-source resampling.

Original AMP models and reference inventories remain fixed. All selection IDs
come from partition 1 of the earlier frozen group-separated analysis.
"""
from pathlib import Path
import os,sys,json,time,argparse,hashlib
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
from concurrent.futures import ProcessPoolExecutor
WORK=Path(__file__).resolve().parents[1]
OLD=Path(os.environ.get('AMP_LINKED_ANALYSIS',str(WORK.parent/'reliability')))
sys.path.insert(0,str(OLD/'code'))
import analyze as A
C=A.C

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cache_identity(fi,B):
    paths=[Path(__file__),WORK/'protocol_refit_v1.json',OLD/'input_manifest.json',
           OLD/'code/analyze.py',OLD/'code/legacy_common.py',
           OLD/'cache/historical_distance_index.json',OLD/'cache'/f'development_{fi}.npz',
           OLD/'results/inventory_nearest_reference.tsv.gz']
    manifest=json.loads((OLD/'input_manifest.json').read_text(encoding='utf8'))
    paths.extend(OLD/'inputs'/n for n in sorted(manifest))
    files={str(p):sha(p) for p in paths}
    info=dict(cache_schema=2,fold=fi,draws=B,files=files,numpy=np.__version__,
              python=sys.version,seeds=[2026110209+fi*1000,2026110209+fi*1000+499],
              models=list(A.MODELS),metrics=list(A.METRICS),retention=0.8)
    fingerprint=hashlib.sha256(json.dumps(info,sort_keys=True).encode('utf8')).hexdigest()
    return dict(fingerprint=fingerprint,details=info)

def validated_cache(path,audit_path,identity,B):
    if not path.exists():return False
    if not audit_path.exists():raise RuntimeError('Cache audit missing; use a fresh output folder')
    audit=json.loads(audit_path.read_text(encoding='utf8'))
    if audit.get('cache_identity',{}).get('fingerprint')!=identity['fingerprint']:
        raise RuntimeError('Cache identity mismatch; existing results preserved; use a fresh output folder')
    with np.load(path) as z:
        assert all(z[k].shape==(B,len(A.MODELS),len(A.METRICS)) for k in ['joint','refit','conditional'])
        assert list(z['models'])==list(A.MODELS) and list(z['metrics'])==list(A.METRICS)
        assert z['points'].shape==(len(A.MODELS),len(A.METRICS))
    return True

def get_data(fi):
    h,hist,ix,inv=A.load()
    src=np.array([r['dataset'] for r in h]);grp=np.array([r['global_group'] for r in h])
    label=np.array([int(r['label']) for r in h]);length=np.log([C.sequence_length(r) for r in h])
    uid=np.array([ix[r['sequence_id']] for r in h]);held=C.SOURCES[fi]
    test=np.flatnonzero(src==held);forbidden=np.unique(grp[test])
    dev=(src!=held)&~np.isin(grp,forbidden)
    with np.load(OLD/'cache'/f'development_{fi}.npz') as z:sel=dev&z['selection'][0,uid]
    train=np.flatnonzero(sel);dg,di=np.unique(grp[train],return_inverse=True)
    hg,hi=np.unique(grp[test],return_inverse=True)
    assert not set(dg)&set(hg)
    sources=[s for s in C.SOURCES if s!=held]
    tiers=[np.flatnonzero(src[train]==s) for s in sources]
    assert all(len(x)>0 for x in tiers)
    features={};predictions={}
    for model in A.MODELS:
        p=np.array([float(r[model+'__score']) for r in h]);pr=np.array([int(r[model+'__prediction']) for r in h])
        c=A.confidence(p,pr);z=np.clip(c,1e-6,1-1e-6)
        I=np.array([inv[r['sequence_id'],model] for r in h])
        features[model]=np.column_stack([np.log(z/(1-z)),length,I])
        predictions[model]=pr
    return h,held,train,test,dg,di,hg,hi,tiers,label,features,predictions

def weights(counts,tiers):
    w=np.zeros(len(counts),float)
    for jj in tiers:
        n=counts[jj].sum()
        if n==0:return None
        w[jj]=counts[jj]/n/len(tiers)
    return w

def metric(y,pr,score,w):
    # Explicitly drop zero-mass records, then apply frozen fractional-tie rule.
    use=w>0;w=w[use];y=y[use];pr=pr[use];score=score[use]
    sw=A.selected(w,score,.8)
    return np.array([A.metrics(w,sw,y,pr)[m][0] for m in A.METRICS])

def fitpair(X,y,w,H):
    use=w>0;out=[];status=[]
    for cols in [[0,1],[0,1,2]]:
        f=A.fit_selector(X[use][:,cols],y[use],w[use])
        status.append(f['status'])
        out.append(-A.predict_selector(f,H[:,cols]))
    return out,status

def run_fold(fi,B,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/f'fold_{fi}.npz'
    identity=cache_identity(fi,B)
    if validated_cache(path,outdir/f'fold_{fi}_audit.json',identity,B):return str(path)
    t=time.time()
    h,held,tr,te,dg,di,hg,hi,tiers,y,X,pr=get_data(fi)
    trainX={m:v[tr] for m,v in X.items()};testX={m:v[te] for m,v in X.items()}
    err={m:(pr[m][tr]!=y[tr]).astype(int) for m in A.MODELS}
    w0=weights(np.ones(len(tr)),tiers)
    originals={m:fitpair(trainX[m],err[m],w0,testX[m])[0] for m in A.MODELS}
    points=np.array([metric(y[te],pr[m][te],originals[m][1],np.ones(len(te)))-metric(y[te],pr[m][te],originals[m][0],np.ones(len(te))) for m in A.MODELS])
    # Independent streams; identical group draws across all models and methods.
    rd=np.random.default_rng(2026110209+fi*1000)
    rh=np.random.default_rng(2026110209+fi*1000+499)
    joint=np.full((B,5,3),np.nan);refit=joint.copy();conditional=joint.copy()
    failed=0;constant=0;empty=0
    for b in range(B):
        d=rd.multinomial(len(dg),np.full(len(dg),1/len(dg)))
        hh=rh.multinomial(len(hg),np.full(len(hg),1/len(hg)))
        wt=hh[hi].astype(float);ws=weights(d[di],tiers)
        for mi,m in enumerate(A.MODELS):
            conditional[b,mi]=metric(y[te],pr[m][te],originals[m][1],wt)-metric(y[te],pr[m][te],originals[m][0],wt)
            if ws is None:continue
            ss,status=fitpair(trainX[m],err[m],ws,testX[m])
            constant+=status.count('single_error_class_constant')
            if 'maximum_iterations' in status:failed+=1;continue
            joint[b,mi]=metric(y[te],pr[m][te],ss[1],wt)-metric(y[te],pr[m][te],ss[0],wt)
            refit[b,mi]=metric(y[te],pr[m][te],ss[1],np.ones(len(te)))-metric(y[te],pr[m][te],ss[0],np.ones(len(te)))
        if ws is None:empty+=1
        if (b+1)%100==0:print(f'REFIT {held} {b+1}/{B} elapsed={time.time()-t:.1f}s',flush=True)
    np.savez_compressed(path,joint=joint,refit=refit,conditional=conditional,points=points,models=A.MODELS,metrics=A.METRICS)
    C.dump(outdir/f'fold_{fi}_audit.json',dict(held_source=held,bootstrap=B,selection_n=len(tr),selection_groups=len(dg),held_n=len(te),held_groups=len(hg),empty_development_draws=empty,failed_model_draws=failed,constant_fits=constant,group_overlap=False,selection_labels_only=True,inventory_references_fixed=True,cache_identity=identity,seconds=time.time()-t))
    print('FOLD COMPLETE',held,round(time.time()-t,1),flush=True)
    return str(path)

def aggregate(B,outdir):
    outdir=Path(outdir);rows=[];audits=[]
    for fi,held in enumerate(C.SOURCES):
        audits.append(json.loads((outdir/f'fold_{fi}_audit.json').read_text(encoding='utf8')))
        with np.load(outdir/f'fold_{fi}.npz') as z:
            for mi,m in enumerate(A.MODELS):
                for ki,k in enumerate(A.METRICS):
                    for mode in ['conditional','refit','joint']:
                        v=z[mode][:,mi,ki];lo,up,status,fraction=C.interval(v,B,audits[-1]['held_groups'])
                        vl=v[:B//2];vl=vl[np.isfinite(vl)]
                        q=np.quantile(vl,[.025,.975]) if len(vl) else [np.nan,np.nan]
                        rows.append(dict(held_source=held,model=m,metric=k,uncertainty_scope=mode,estimate=float(z['points'][mi,ki]),ci_low=lo,ci_high=up,valid_fraction=fraction,interval_status=status,bootstrap=B,first_half_low=q[0],first_half_high=q[1],quantile_shift_pp=100*max(abs(q[0]-lo),abs(q[1]-up)) if np.isfinite(lo) else np.nan))
    C.write(outdir/'refit_intervals_80.tsv',rows)
    C.dump(outdir/'FORMAL_COMPLETE.json',dict(formal=True,bootstrap=B,folds=6,models=5,metrics=3,scopes=3,audits=audits,scope='joint global-group development refit and independent held-source resampling; fixed partition and inventory'))
    print('FORMAL REFIT COMPLETE',B,flush=True)

def selftest():
    # Weighted fit equals an explicitly expanded bootstrap dataset.
    rng=np.random.default_rng(86);x=rng.normal(size=(50,3));y=(rng.random(50)>.65).astype(int)
    counts=rng.integers(0,4,size=50);jj=np.repeat(np.arange(50),counts)
    a=A.fit_selector(x[counts>0],y[counts>0],counts[counts>0]);b=A.fit_selector(x[jj],y[jj],np.ones(len(jj)))
    assert np.allclose(A.predict_selector(a,x),A.predict_selector(b,x),atol=1e-10)
    for _ in range(20):
        p=rng.integers(0,2,size=50);s=rng.integers(0,5,size=50)
        assert np.allclose(metric(y,p,s,counts),metric(y[jj],p[jj],s[jj],np.ones(len(jj))),equal_nan=True)
    print('REFIT SELFTEST PASS',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--draws',type=int,default=2000);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    selftest();folder=WORK/'results'/('refit_smoke_v2' if a.smoke else 'refit_formal_v2')
    if a.smoke:run_fold(0,20,folder)
    else:
        assert a.draws==json.loads((WORK/'protocol_refit_v1.json').read_text(encoding='utf8'))['bootstrap_draws']
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_fold,range(6),[a.draws]*6,[str(folder)]*6))
        aggregate(a.draws,folder)
