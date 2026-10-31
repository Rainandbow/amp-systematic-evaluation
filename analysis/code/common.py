"""Shared, label-safe utilities for the two saved-output analyses."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODELS=['Macrel','AMPlify','ampir','tsAMP_I','HMD_AMP','c_AMPs']
SOURCES=['AMPlify','tsAMP_I','c_AMPs_benchmark_resplit','HMD_AMP','AmPEP_raw_source','ampir_raw_mature90']
COHORTS=['ApexGO_template_based','ApexGO_template_free','HydrAMP_auxiliary_preselected','HydrAMP_no_auxiliary_preselection']
BASELINES=['length_high','length_low','KR_minus_DE_high','KR_minus_DE_low','mean_Kyte_Doolittle_high','mean_Kyte_Doolittle_low']
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-.8,'T':-.7,'W':-.9,'Y':-1.3,'V':4.2}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=json_default)+'\n',encoding='utf-8')
def json_default(obj):
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,Path):return str(obj)
    raise TypeError(type(obj).__name__)
def ratio(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)
def scalar(x):return float(np.asarray(x).ravel()[0])
def verify_inputs():
    m=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
    for n,h in m['files'].items():
        p=ROOT/'inputs'/n
        if not p.is_file() or sha(p)!=h:raise RuntimeError('Input SHA256 mismatch: '+n)
    if np.__version__!='2.3.5':raise RuntimeError('Use NumPy 2.3.5 from requirements.txt; actual '+np.__version__)
    return m
def protocol():return json.loads((ROOT/'protocol_v1.json').read_text(encoding='utf-8'))

def selected(W,score,budget=.2):
    """Expected top-ceil(b*n) selection within label-blind equal-score tiers."""
    W=np.atleast_2d(W).astype(float);score=np.asarray(score,float)
    if not np.isfinite(score).all():raise ValueError('All ranking scores must be finite')
    order=np.argsort(-score,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(score[order])!=0)+1]
    mass=np.add.reduceat(W[:,order],starts,axis=1)
    before=np.cumsum(mass,axis=1)-mass
    capacity=np.ceil(budget*W.sum(axis=1)-1e-10)
    fraction=np.clip(np.divide(capacity[:,None]-before,mass,out=np.zeros_like(mass),where=mass>0),0,1)
    tier=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(order)]))
    S=np.zeros_like(W);S[:,order]=W[:,order]*fraction[:,tier]
    assert np.allclose(S.sum(axis=1),capacity,atol=1e-8)
    return S,capacity

def budget_metrics(W,S,y):
    W=np.atleast_2d(W);y=np.asarray(y,int)
    n=W.sum(axis=1);p=W[:,y==1].sum(axis=1);k=S.sum(axis=1);h=S[:,y==1].sum(axis=1)
    random=k*ratio(p,n);oracle=np.minimum(k,p);headroom=oracle-random
    return {'positive_hits':h,'precision':ratio(h,k),'recovery':ratio(h,p),
        'enrichment':ratio(ratio(h,k),ratio(p,n)),'random_expected_hits':random,
        'oracle_max_hits':oracle,'available_hit_headroom':headroom,
        'excess_hits_vs_random':h-random,'headroom_fraction':ratio(h-random,headroom)}

def auc(W,y,score):
    W=np.atleast_2d(W);y=np.asarray(y,int);score=np.asarray(score,float)
    order=np.argsort(-score,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(score[order])!=0)+1]
    pm=np.add.reduceat(W[:,order]*(y[order]==1),starts,axis=1)
    nm=np.add.reduceat(W[:,order]*(y[order]==0),starts,axis=1)
    return ratio((nm*(np.cumsum(pm,axis=1)-.5*pm)).sum(axis=1),pm.sum(axis=1)*nm.sum(axis=1))

def interval(v,B,G):
    v=np.asarray(v,float);ok=v[np.isfinite(v)];fraction=len(ok)/B
    if G<5:return np.nan,np.nan,'fewer_than_five_contributing_blocks',fraction
    if fraction<.95:return np.nan,np.nan,'insufficient_valid_replicates',fraction
    if not len(ok):return np.nan,np.nan,'undefined',fraction
    if np.ptp(ok)<1e-12:return np.nan,np.nan,'degenerate',fraction
    lo,hi=np.quantile(ok,[.025,.975])
    return float(lo),float(hi),'pointwise_exploratory_percentile',fraction

def descriptors(sequence):
    if not sequence or set(sequence)-set(KD):raise ValueError('Noncanonical or empty sequence')
    length=len(sequence);charge=sequence.count('K')+sequence.count('R')-sequence.count('D')-sequence.count('E')
    # Integer tenth-units make the descriptor exactly composition-invariant;
    # floating summation in residue order must not break otherwise exact ties.
    hydro=sum(int(round(KD[a]*10)) for a in sequence)/(10*length)
    return dict(length_high=float(length),length_low=-float(length),KR_minus_DE_high=float(charge),
        KR_minus_DE_low=-float(charge),mean_Kyte_Doolittle_high=hydro,mean_Kyte_Doolittle_low=-hydro)

def load_panel():
    members=read(ROOT/'inputs/members151.tsv');candidates=read(ROOT/'inputs/candidates.tsv')
    scores=read(ROOT/'inputs/scores151.tsv');labels=read(ROOT/'inputs/task_labels.tsv')
    assert len(members)==151 and len(scores)==906 and len(candidates)==169
    cid={c['record_id']:c for c in candidates};si={(r['sequence_id'],r['model']):r for r in scores}
    assert len(si)==906
    rows=sorted(members,key=lambda r:r['record_id']);by_id={r['record_id']:r for r in rows}
    assert len(by_id)==151
    for r in rows:
        c=cid[r['record_id']]
        assert r['sequence']==c['sequence']
        assert r['sequence_id']=='s_'+hashlib.sha256(r['sequence'].encode('ascii')).hexdigest()
        assert r['sequence_sha256']==r['sequence_id'][2:]
        r['family']=c['family'];r['block']=si[r['sequence_id'],'Macrel']['block']
        assert r['family'] and all(si[r['sequence_id'],m]['block']==r['block'] for m in MODELS)
        for m in MODELS:
            assert int(si[r['sequence_id'],m]['label'])==int(r['activity_label_binary'])
            assert np.isfinite(float(si[r['sequence_id'],m]['score']))
    task_index={}
    for r in labels:
        threshold=64 if r['cohort'].startswith('ApexGO') else 32
        if float(r['threshold'])==threshold and r['task'] in ['any_eligible_measured_strain','source_reference_Ecoli']:
            key=r['cohort'],r['task'],r['record_id']
            assert key not in task_index;task_index[key]=r
    return rows,si,task_index

def load_historical():
    members=read(ROOT/'inputs/historical_members.tsv.gz');scores=read(ROOT/'inputs/historical_scores.tsv.gz')
    ix={(r['dataset'],r['sequence_id']):r for r in scores}
    assert len(members)==29104 and len(ix)==len(scores)==41956
    aligned=[]
    for m in members:
        s=ix[m['dataset'],m['sequence_id']]
        assert s['sequence']==m['sequence'] and s['label']==m['label']
        assert m['sequence_id']=='s_'+hashlib.sha256(m['sequence'].encode('ascii')).hexdigest()
        for model in MODELS:assert s[model+'__prediction']==m[model+'__prediction']
        aligned.append(dict(s,global_group=m['global_group']))
    assert set(r['dataset'] for r in aligned)==set(SOURCES)
    return aligned
