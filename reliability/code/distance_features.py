"""Exact label-free normalized global edit distances and selection-only references."""
from pathlib import Path
import sys, csv, gzip, json, hashlib, time, argparse, math
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python_deps'))
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
import legacy_common as C
MODELS = C.MODELS[:-1]
_cache = {}

def read(path): return C.read(path)
def dump(path, x): C.dump(path, x)

def prepare():
    (ROOT/'cache').mkdir(exist_ok=True)
    h=C.load_historical(); panel,_,_=C.load_panel()
    seq={r['sequence_id']:r['sequence'] for r in h+panel}
    ids=sorted(seq); seqs=[seq[x] for x in ids]
    groups={r['sequence_id']:r['global_group'] for r in h}
    meta=[dict(sequence_id=x,sequence=seq[x],length=len(seq[x]),global_group=groups.get(x,''),historical=x in groups) for x in ids]
    dump(ROOT/'cache/query_index.json',meta)
    refs={r['reference_id']:r for r in read(ROOT/'inputs/references.tsv.gz')}
    pools={m:[] for m in MODELS}; exclusions={m:0 for m in MODELS}
    for r in read(ROOT/'inputs/reference_membership.tsv.gz'):
        m=r['tool']
        if m not in pools: continue
        q=refs[r['reference_id']]
        if q['canonical']=='1': pools[m].append((q['reference_id'],q['sequence']))
        else: exclusions[m]+=1
    for m,p in pools.items():
        assert len({x[0] for x in p})==len(p)
        with gzip.open(ROOT/'cache'/f'{m}_canonical_reference.json.gz','wt',encoding='utf8') as f:json.dump(sorted(p),f)
    dump(ROOT/'qa/reference_roles.json',dict(inventories=read(ROOT/'inputs/inventory.tsv'),canonical_reference_n={m:len(v) for m,v in pools.items()},excluded_noncanonical_n=exclusions,distance_definition='Levenshtein(unit cost) / max(lengths), global; not percent identity or old overlap criterion'))
    return meta

def worker(task):
    model,start,stop=task
    if 'meta' not in _cache:_cache['meta']=json.loads((ROOT/'cache/query_index.json').read_text(encoding='utf8'))
    if model not in _cache:
        with gzip.open(ROOT/'cache'/f'{model}_canonical_reference.json.gz','rt',encoding='utf8') as f:rr=json.load(f)
        buckets={};exact={}
        for rid,s in rr:
            buckets.setdefault(len(s),{})[rid]=s;exact[s]=rid
        _cache[model]=(buckets,exact)
    buckets,exact=_cache[model];out=[]
    for qi in range(start,stop):
        q=_cache['meta'][qi];s=q['sequence'];L=len(s)
        if s in exact:out.append((qi,0.,exact[s],L));continue
        best=1.+1e-12;bestid=None;bestlen=None
        for length in sorted(buckets,key=lambda k:(abs(k-L)/max(k,L),k)):
            bound=abs(length-L)/max(length,L)
            if bound>best+1e-12:break
            hit=process.extractOne(s,buckets[length],scorer=Levenshtein.normalized_distance,score_cutoff=min(1.,best+1e-12))
            if hit is not None:
                _,d,rid=hit
                if d<best-1e-12 or (abs(d-best)<=1e-12 and (bestid is None or rid<bestid)):
                    best,bestid,bestlen=float(d),rid,length
        assert bestid is not None and 0<=best<=1
        out.append((qi,best,bestid,bestlen))
    return model,out

def inventories(meta,workers):
    jobs=[(m,a,min(a+300,len(meta))) for m in MODELS for a in range(0,len(meta),300)]
    result={m:[None]*len(meta) for m in MODELS};done=0;start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(worker,t) for t in jobs]
        for fu in as_completed(futures):
            model,rows=fu.result()
            for qi,d,rid,length in rows:result[model][qi]=(d,rid,length)
            done+=1
            if done%20==0 or done==len(jobs):print('INVENTORY DISTANCE',done,'/',len(jobs),'seconds',round(time.perf_counter()-start),flush=True)
    records=[]
    for m in MODELS:
        for q,r in zip(meta,result[m]):
            assert r is not None
            d,rid,length=r
            records.append(dict(sequence_id=q['sequence_id'],model=m,normalized_edit_distance=d,nearest_reference_id=rid,reference_length=length,exact_overlap=int(d==0),reference_role='fixed_available_model_inventory',reference_status='computed_against_canonical_inventory'))
    C.write(ROOT/'results/inventory_nearest_reference.tsv.gz',records)
    # Independently compare bucket pruning against complete unpruned search.
    checks=[]
    for m in MODELS:
        with gzip.open(ROOT/'cache'/f'{m}_canonical_reference.json.gz','rt',encoding='utf8') as f:choices=dict(json.load(f))
        rng=np.random.default_rng(2026110203)
        for qi in rng.choice(len(meta),12,replace=False):
            hit=process.extractOne(meta[qi]['sequence'],choices,scorer=Levenshtein.normalized_distance)
            d=result[m][qi][0];assert abs(hit[1]-d)<1e-12
            checks.append(dict(model=m,query=int(qi),pruned_distance=d,unpruned_distance=hit[1],passed=True))
    C.write(ROOT/'qa/inventory_exact_search_checks.tsv',checks)

def matrix(meta,workers):
    hist=[x for x in meta if x['historical']];ids=[x['sequence_id'] for x in hist];seq=[x['sequence'] for x in hist]
    assert len(ids)==28481
    dump(ROOT/'cache/historical_distance_index.json',hist)
    p=ROOT/'cache/historical_distance.npy'
    if p.exists():
        a=np.load(p,mmap_mode='r');assert a.shape==(len(ids),len(ids));assert (ROOT/'cache/distance_matrix_complete.json').exists();return
    a=np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=(len(ids),len(ids)))
    for start in range(0,len(ids),128):
        stop=min(start+128,len(ids))
        a[start:stop]=process.cdist(seq[start:stop],seq,scorer=Levenshtein.normalized_distance,dtype=np.float32,workers=workers)
        if start%4096==0:print('HISTORICAL DISTANCE MATRIX',stop,'/',len(ids),flush=True)
    a.flush()
    rng=np.random.default_rng(2026110204);qs=rng.integers(0,len(ids),1000);rs=rng.integers(0,len(ids),1000)
    assert np.allclose(a[qs,rs],a[rs,qs]);assert np.all(a[qs,qs]==0)
    dump(ROOT/'cache/distance_matrix_complete.json',dict(n=len(ids),dtype='float32',metric='global normalized unit edit distance',symmetry_sample=1000,diagonal_sample=1000))

def development(meta):
    h=C.load_historical();hist=json.loads((ROOT/'cache/historical_distance_index.json').read_text(encoding='utf8'))
    ids=[r['sequence_id'] for r in hist];ix={v:i for i,v in enumerate(ids)};group=np.array([r['global_group'] for r in hist]);N=len(ids)
    D=np.load(ROOT/'cache/historical_distance.npy',mmap_mode='r')
    audit=[]
    for fi,held in enumerate(C.SOURCES):
        target=[r for r in h if r['dataset']==held];forbidden={r['global_group'] for r in target}
        dev=[r for r in h if r['dataset']!=held and r['global_group'] not in forbidden]
        groups=sorted({r['global_group'] for r in dev})
        selection=np.empty((20,N),bool)
        for rep in range(20):
            assignment=np.random.default_rng(2026102602+fi*100000+rep).random(len(groups))<.7
            selected={g for g,a in zip(groups,assignment) if a}
            selection[rep]=np.isin(group,list(selected))
            audit.append(dict(held_source=held,repeat=rep+1,held_records=len(target),development_records=len(dev),selection_groups=len(selected),validation_groups=len(groups)-len(selected),held_group_overlap=0,selection_validation_group_overlap=0,selection_sequence_n=int(selection[rep].sum())))
        cand=np.array(sorted({ix[r['sequence_id']] for r in dev}),int)
        near=np.full((20,N),np.nan,dtype=np.float32);nearest=np.full((20,N),-1,dtype=np.int32)
        for start in range(0,N,96):
            q=np.arange(start,min(start+96,N));a=np.array(D[np.ix_(q,cand)])
            a[group[q,None]==group[cand][None,:]]=np.inf
            k=min(32,len(cand));part=np.argpartition(a,k-1,axis=1)[:,:k]
            vals=np.take_along_axis(a,part,axis=1);order=np.argsort(vals,axis=1,kind='stable');part=np.take_along_axis(part,order,axis=1)
            cols=cand[part];vals=np.take_along_axis(a,part,axis=1)
            for rep in range(20):
                available=selection[rep,cols] & np.isfinite(vals)
                found=available.any(axis=1);first=available.argmax(axis=1)
                near[rep,q[found]]=vals[np.arange(len(q))[found],first[found]]
                nearest[rep,q[found]]=cols[np.arange(len(q))[found],first[found]]
                for local in np.flatnonzero(~found):
                    eligible=selection[rep,cand] & (group[cand]!=group[q[local]])
                    assert eligible.any()
                    choices=np.flatnonzero(eligible);j=choices[np.argmin(a[local,choices])]
                    near[rep,q[local]]=a[local,j];nearest[rep,q[local]]=cand[j]
            if start%6144==0:print('SELECTION REFERENCES',held,int(q[-1])+1,'/',N,flush=True)
        assert np.isfinite(near).all() and (nearest>=0).all()
        # Every recorded nearest belongs to selection and differs from query group.
        for rep in range(20):
            assert selection[rep,nearest[rep]].all()
            assert (group[nearest[rep]]!=group).all()
            assert not set(group[selection[rep]])&forbidden
            rng=np.random.default_rng(2026110210+fi*100+rep)
            for qi in rng.choice(N,5,replace=False):
                eligible=selection[rep]&(group!=group[qi]);truth=np.min(np.array(D[qi,eligible]))
                assert abs(float(truth)-float(near[rep,qi]))<1e-7
        np.savez_compressed(ROOT/'cache'/f'development_{fi}.npz',distance=near,nearest=nearest,selection=selection)
        print('SELECTION REFERENCES COMPLETE',held,flush=True)
    C.write(ROOT/'qa/reference_partition_audit.tsv',audit)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inventory','development','all']);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    if (ROOT/'cache/query_index.json').exists():meta=json.loads((ROOT/'cache/query_index.json').read_text(encoding='utf8'))
    else:meta=prepare()
    if args.action in ['inventory','all']:inventories(meta,args.workers)
    if args.action in ['development','all']:matrix(meta,args.workers);development(meta)
