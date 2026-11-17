"""Certified contiguous shared-query route using the pinned v21 quantum kernel."""
from pathlib import Path
from itertools import combinations
from functools import lru_cache
from numbers import Integral
import importlib.util,sys,json
from runtime import ROOT,QC,Parameter,core,sha
import auto_reuse_v2 as A

SNAP=ROOT/'vendor/query_sharing_v21'
sys.path.insert(0,str(SNAP/'code'))
path=SNAP/'code/ours_shared_flag_core_v1.py'
assert sha(path)=='49a34c43ca3b03e95460f99e99326919fcd20cddf7e926cdda375c10e66d2137'
s=importlib.util.spec_from_file_location('journal_pinned_shared_query',path);K=importlib.util.module_from_spec(s);s.loader.exec_module(K)


def recipes(spec,targets):
    if any(j in spec['adj'][i] for i in targets for j in targets if i!=j):return []
    common=set.intersection(*(spec['adj'][i] for i in targets));out=[]
    for a in sorted(common):
        for size in (2,3):
            for S in combinations(sorted(common-{a}),size):
                counts=[len(spec['adj'][i]-{a,*S}) for i in targets]
                if any(c>size+1 for c in counts):continue
                bound=2*({2:3,3:6}[size])+sum(1 if c==0 else 6*c for c in counts)+2*len(targets)
                out.append((bound,a,tuple(S)))
    return out


def build(case):
    spec=A.graph_spec(case);n=spec['n'];p=case.get('sweeps',2)
    if not isinstance(p,Integral) or isinstance(p,bool) or p<1:raise ValueError('Positive exact integer sweep count required')
    p=int(p);orders=case.get('sweep_orders',[spec['order']]*p)
    if len(orders)!=p or any(any(not isinstance(i,Integral) or isinstance(i,bool) for i in order) or sorted(order)!=list(range(n)) for order in orders):raise ValueError('Invalid prescribed sweep orders')
    b=Parameter('shared_local_beta');single={};cost={};certs={}
    for i in range(n):
        single[i],certs[i]=A.query_kernel(n,i,sorted(spec['adj'][i]),b,'phi');cost[i]=core.costs(single[i])['cx']
    selected_orders=[];modules={}
    for order in orders:
        options={}
        for start in range(n):
            choices=[dict(targets=[order[start]],cost=cost[order[start]],kind='single',key=('single',order[start]))]
            for size in (2,3):
                T=order[start:start+size]
                if len(T)!=size:continue
                pool=recipes(spec,T)
                if pool:
                    bound,a,S=min(pool)
                    choices.append(dict(targets=T,cost=bound,kind='shared',a=a,S=list(S),key=('shared',tuple(T),a,S)))
            options[start]=choices
        dp=[None]*(n+1);dp[n]=(0,[])
        for start in reversed(range(n)):
            choices=[]
            for c in options[start]:
                rest=dp[start+len(c['targets'])];choices.append((c['cost']+rest[0],(0 if c['kind']=='single' else 1,len(c['targets']),tuple(c['targets'])),[c]+rest[1]))
            best=min(choices,key=lambda x:(x[0],x[1]));dp[start]=(best[0],best[2])
        selected_orders.append(dp[0][1])
        for rec in dp[0][1]:
            key=rec['key']
            if key in modules:continue
            if rec['kind']=='single':modules[key]=(single[rec['targets'][0]],dict(kind='single',primitive=certs[rec['targets'][0]]))
            else:
                raw,oracle,cert=K.build_kernel(dict(case,target_order=order),rec['targets'],rec['a'],rec['S'],beta=b)
                compiled=core.native(raw.copy(),1);assert core.costs(compiled)['cx']<=rec['cost']
                modules[key]=(compiled,dict(kind='shared',quantum_kernel_sha256=sha(path),certificate=cert,upper_bound=rec['cost'],native1=core.costs(compiled)))
    beta=[Parameter('beta'+str(k)) for k in range(p)];gamma=[Parameter('gamma'+str(k)) for k in range(p-1)];q=QC(n)
    for k,recs in enumerate(selected_orders):
        for rec in recs:q.compose(modules[rec['key']][0].assign_parameters({b:beta[k]}),inplace=True)
        if k<p-1:q.compose(A.weighted_cost(spec,set(),gamma[k],'clique')[0],inplace=True)
    serial=[]
    for recs in selected_orders:
        serial.append([{k:v for k,v in rec.items() if k!='key'} for rec in recs])
    return q,dict(schema='journal_contiguous_query_reuse_v1',route='certified_shared_queries_no_global_encoding',sweeps=p,orders=orders,
                  selected_groups=serial,selected_shared_groups=sum(r['kind']=='shared' for recs in selected_orders for r in recs),
                  grouping_rule='Exact interval cover for contiguous1/2/3-target candidates under declared constructive costs; ties preserve singletons',
                  noncommuting_original_order_unchanged=True,no_target_swaps=True,new_work_qubits=0,
                  scope='Feas coherent canonical product; no general full-domain even-sweep claim',
                  metadata=[v[1] for v in modules.values()],raw_metrics=core.costs(q))
