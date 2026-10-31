"""Software-only numerical checks; fixtures never enter study output files."""
import itertools
import numpy as np
from common import selected,budget_metrics,auc,descriptors,interval
from honest_transfer import curve,choose,evaluate,POLICIES

def run():
    checks=[]
    W=np.ones((1,4));score=np.array([.9,.8,.8,.1]);y=np.array([1,1,0,0])
    S,K=selected(W,score,.5)
    assert np.allclose(S,[[1,.5,.5,0]]) and K[0]==2
    assert budget_metrics(W,S,y)['positive_hits'][0]==1.5
    checks.append('label-blind fractional selection at exact score ties')
    assert auc(W,y,np.ones(4))[0]==.5
    assert auc(W,y,np.array([2,2,1,1]))[0]==1
    assert np.isnan(auc(W,np.ones(4,int),score)[0])
    checks.append('half-credit AUROC ties; perfect ordering; single-class undefined')
    yall=np.ones(4,int);m=budget_metrics(W,S,yall)
    assert m['available_hit_headroom'][0]==0 and np.isnan(m['headroom_fraction'][0])
    checks.append('all-positive pool has zero hit headroom; normalized gain undefined')
    for weights in [[2,0,1,1],[0,1,0,1]]:
        ss,kk=selected(np.array([weights]),score,.2)
        assert np.isclose(ss.sum(),kk[0]) and (ss<=np.array(weights)+1e-12).all()
    checks.append('block multiplicity and zero-mass score tiers preserve capacity')
    assert descriptors('KRDE')['KR_minus_DE_high']==0
    assert descriptors('AAAA')['length_low']==-4
    assert descriptors('KRDEALIV')['mean_Kyte_Doolittle_high']==descriptors('VILAEDRK')['mean_Kyte_Doolittle_high']
    checks.append('unfitted residue-count proxy and direction definitions')
    checks.append('hydropathy composition invariance prevents floating-order tie breaking')
    assert interval(np.arange(100),100,4)[2]=='fewer_than_five_contributing_blocks'
    assert interval(np.ones(100),100,5)[2]=='degenerate'
    checks.append('small-group and degenerate intervals suppressed')
    # Abstract numeric fixture only, no peptide generation or activity data.
    sources=['s0','s1','s2','s3','s4'];rows=[]
    for s in sources:
        for i in range(100):
            error=s=='s4' and i<20
            rows.append(dict(dataset=s,global_group=s+'g'+str(i),label='1',Macrel__prediction='0' if error else '1',Macrel__score='.99'))
    c=curve(rows,'Macrel',sources)
    a=choose(c,'mixture_original');b=choose(c,'mixture_source_support');d=choose(c,'per_source_supported')
    assert a['policy_status']==b['policy_status']=='feasible'
    assert np.isclose(a['selection_error'],.04) and np.isclose(a['selection_worst_source_error'],.2)
    assert d['policy_status']=='no_feasible_policy'
    checks.append('mixture can meet 5% while a represented source exceeds it; sourcewise policy rejects')
    for policy in POLICIES:
        got=choose(c,policy);brute=[]
        for t in sorted(set(c['threshold'])):
            ev,detail=evaluate(rows,'Macrel',t,sources)
            kept=sum(x['retained_n'] for x in detail);groups=sum(x['retained_groups'] for x in detail)
            feasible=ev['coverage_balanced']>=.2-1e-12 and kept>=100 and groups>=30
            if policy!='mixture_original':feasible&=all(x['coverage']>=.2-1e-12 and x['retained_n']>=20 and x['retained_groups']>=5 for x in detail)
            if policy=='per_source_supported':feasible&=all(x['conditional_error']<=.05+1e-12 for x in detail)
            else:feasible&=ev['conditional_error_balanced']<=.05+1e-12
            if feasible:brute.append(t)
        assert bool(brute)==(got['policy_status']=='feasible')
        if brute:assert got['threshold']==min(brute)
    checks.append('cumulative selectors agree with direct threshold enumeration')
    ev,detail=evaluate(rows,'Macrel',np.nan,sources)
    assert ev['outcome_status']=='policy_unavailable' and np.isnan(ev['conditional_error_balanced'])
    ev,detail=evaluate(rows,'Macrel',1.0,sources)
    assert ev['retained_n']==0 and np.isnan(ev['conditional_error_balanced'])
    checks.append('unavailable/zero-retention rules never treated as zero error')
    # Selection function only receives development selection rows. Mutating a
    # separate validation set cannot alter the threshold it returns.
    held=[dict(r,label='0') for r in rows]
    evaluate(held,'Macrel',a['threshold'],sources)
    assert choose(curve(rows,'Macrel',sources),'mixture_original')['threshold']==a['threshold']
    checks.append('evaluation labels do not enter threshold selection function')
    return checks
