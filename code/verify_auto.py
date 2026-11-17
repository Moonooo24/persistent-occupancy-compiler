"""Independent canonical Feas reference and compiler regression fixtures."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import math
import sys
import time

from runtime import ROOT, QC, Statevector, np, core, io, G, sha
import auto_reuse as A

ANGLES = ((0.,0.,0.),(.37,.31,-.51),(3.7,-4.13,6.29),(-.417,1.117,-2.31))


def prescribed_values(case, index):
    p=int(case.get('sweeps',2))
    result={'beta'+str(k):0. if index==0 else (.31,-.51,1.117)[index-1]*(1+k) for k in range(p)}
    result.update({'gamma'+str(k):0. if index==0 else (.37,3.7,-.417)[index-1]*(1+k) for k in range(p-1)})
    return result


def required_parameters(case):
    p=int(case.get('sweeps',2))
    result={'beta'+str(k) for k in range(p)}
    if any(Fraction(str(w)) for w in case.get('weights',[1]*case['n'])):
        result|={'gamma'+str(k) for k in range(p-1)}
    return result


def feasible(case):
    ids = np.arange(1 << case['n'])
    ok = np.ones(len(ids),bool)
    for a,b in case['graph_edges']:
        ok &= ~(((ids >> a)&1).astype(bool) & ((ids >> b)&1).astype(bool))
    return ids[ok], ok


def canonical(v, case, mapping):
    """Literal original MIS updates; no compiler partition or GF2 routine."""
    n=case['n'];states=np.arange(1 << n)
    neighbors=[[] for _ in range(n)]
    for a,b in case['graph_edges']:
        neighbors[a].append(b);neighbors[b].append(a)
    p=int(case.get('sweeps',2));orders=case.get('sweep_orders',[case.get('target_order',list(range(n)))]*p)
    weights=[float(Fraction(str(w))) for w in case.get('weights',[1]*n)]
    phase=sum(weights[j]*((states >> j)&1) for j in range(n))
    out=v.copy()
    for k in range(p):
        beta=mapping['beta'+str(k)]
        for i in orders[k]:
            mask=sum(1 << j for j in neighbors[i]);lo=states[((states & mask)==0)&((states & (1 << i))==0)];hi=lo|(1 << i)
            x=out[lo].copy();y=out[hi].copy();out[lo]=np.cos(beta)*x-1j*np.sin(beta)*y;out[hi]=np.cos(beta)*y-1j*np.sin(beta)*x
        if k < p-1:
            factor=np.exp(1j*(mapping['gamma'+str(k)] if any(weights) else 0.)*phase)
            out*=factor if out.ndim==1 else factor[:,None]
    return out


def checks(q, case, all_columns=True):
    n=case['n'];ids,ok=feasible(case);rng=np.random.default_rng(20261109+n);rows=[]
    assert {p.name for p in q.parameters}==required_parameters(case), 'Circuit parameter family differs from prescribed problem'
    for index in range(4):
        mapping=prescribed_values(case,index);args={p:mapping[p.name] for p in q.parameters};bound=q.assign_parameters(args)
        maximum=leak=0.;tested=0
        if all_columns:
            batch=4
            for start in range(0,len(ids),batch):
                sel=ids[start:start+batch];x=np.zeros((1 << n,batch),complex);x[sel,np.arange(len(sel))]=1.;want=canonical(x,case,mapping)
                packed=x.T.reshape(-1)/math.sqrt(len(sel));got=Statevector(packed.copy()).evolve(bound,qargs=list(range(n))).data.reshape(batch,1 << n).T*math.sqrt(len(sel))
                maximum=max(maximum,float(np.max(abs(got-want))));leak=max(leak,float(np.max(abs(got[~ok]))) if (~ok).any() else 0.);tested+=len(sel)
        for _ in range(2):
            x=np.zeros(1 << n,complex);x[ids]=rng.normal(size=len(ids))+1j*rng.normal(size=len(ids));x/=np.linalg.norm(x);want=canonical(x,case,mapping);got=Statevector(x.copy()).evolve(bound).data;maximum=max(maximum,float(np.max(abs(got-want))))
        assert max(maximum,leak)<2e-9,(case['id'],index,maximum,leak)
        rows.append(dict(parameter_values=mapping,Feas_columns=tested,complex_superpositions=2,max_complex_error=maximum,max_leakage=leak,no_phase_alignment=True))
    return rows


def fixtures():
    return [
        dict(id='single_isolate_p1',n=1,graph_edges=[],sweeps=1,weights=['2/3']),
        dict(id='edgeless_p3',n=4,graph_edges=[],sweeps=3,weights=[0,-2,3,'1/7']),
        dict(id='isolated_clique_p1',n=4,graph_edges=[[i,j] for i in range(4) for j in range(i+1,4)],sweeps=1),
        dict(id='path_singletons_p3',n=5,graph_edges=[[i,i+1] for i in range(4)],sweeps=3,weights=[1,3,0,7,2]),
        dict(id='mixed_twin_p4',n=6,graph_edges=[[0,1],[0,2],[1,2],[2,3],[2,4],[3,4]],sweeps=4,weights=['1/3','2/3',1,-2,5,0]),
        dict(id='disconnected_modules_p2',n=7,graph_edges=[[0,1],[0,2],[1,2],[3,4],[3,5],[4,5]],sweeps=2,weights=[1,4,2,3,2,1,5]),
        dict(id='rank9_fallback_p1',n=10,graph_edges=[[0,j] for j in range(1,10)],sweeps=1),
        dict(id='rank4_module_p2',n=10,graph_edges=[[2*k,2*k+1] for k in range(5)]+[[i,j] for i in (0,1) for j in range(2,10)],sweeps=2),
    ]


def run():
    D=ROOT/'results/correctness_auto_v1';D.mkdir(parents=True,exist_ok=True);assert not (D/'summary.json').exists();rows=[];start=time.perf_counter()
    for c in fixtures():
        c.setdefault('target_order',list(reversed(range(c['n']))))
        for cache in ('none','all'):
            q,plan=A.build(c,cache=cache);one=core.native(q.copy(),1);three=core.native(one.copy(),3);stages=[]
            for label,obj in [('raw',q),('native1',one),('whole3',three)]:
                folder=D/c['id']/cache;folder.mkdir(parents=True,exist_ok=True);path=folder/(label+'.qpy');fp=G.fingerprint(obj)
                with path.open('wb') as stream:io.dump(obj,stream)
                with path.open('rb') as stream:loaded=io.load(stream)[0]
                assert fp==G.fingerprint(loaded);tests=checks(loaded,c,all_columns=c['n']<=7)
                stages.append(dict(stage=label,metrics=core.costs(loaded),file=str(path.relative_to(ROOT)),sha256=sha(path),tests=tests))
            rows.append(dict(case=c,cache=cache,plan=plan,stages=stages));print(json.dumps(dict(case=c['id'],cache=cache,status='passed',whole3=stages[-1]['metrics'])),flush=True)
    result=dict(status='all_fixtures_passed',source_sha256=sha(ROOT/'code/auto_reuse.py'),rows=rows,seconds=time.perf_counter()-start,
                max_complex_error=max(t['max_complex_error'] for r in rows for st in r['stages'] for t in st['tests']),scope='Independent canonical Feas reference, exact scalar, no phase alignment; full columns on n<=7, complex Feas inputs on n10')
    (D/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],cases=len(rows),max_error=result['max_complex_error'],seconds=result['seconds'])))


if __name__=='__main__':
    run()
