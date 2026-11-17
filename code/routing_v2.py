"""Fixed-layout routing with explicit restoration of logical data locations."""
from collections import deque
from pathlib import Path
import argparse,json,time,gc,hashlib
from runtime import ROOT, QC, core, G, io, sha, psutil, np, Statevector
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import auto_reuse_v2 as A
from experiments_v2 import construct,put
from phase_safe import native as safe_native,normalize_owned_phase,BACKEND
from verify_auto import checks,required_parameters,prescribed_values

SEEDS=(20261109,20261110,20261111)


def coupling(n,topology):
    if topology=='line':edges=[[i,i+1] for i in range(n-1)]
    elif topology=='grid':
        import math
        a=max(k for k in range(1,math.isqrt(n)+1) if n%k==0);b=n//a
        edges=[[i,i+1] for i in range(n) if i%b<b-1]+[[i,i+b] for i in range(n-b)]
    else:raise ValueError(topology)
    result=CouplingMap(edges+[list(reversed(e)) for e in edges])
    if n==1:result.add_physical_qubit(0)
    return result


def _reject_numeric_blocks(**kwargs):
    assert not kwargs['dag'].count_ops().get('unitary',0),'Unexpected numeric matrix synthesis block'


def routed(source,topology,seed):
    n=source.num_qubits;ledger=[];cmap=coupling(n,topology)
    with core.phase_policy(ledger):
        compiled=transpile(source.copy(),coupling_map=cmap,basis_gates=core.BASIS,optimization_level=0,
                           initial_layout=list(range(n)),layout_method='trivial',routing_method='sabre',
                           seed_transpiler=seed,qubits_initially_zero=False,callback=_reject_numeric_blocks)
    assert compiled.num_qubits==n
    physical=QC(n);physical.global_phase=compiled.global_phase
    for ins in compiled.data:
        physical.append(ins.operation.copy(),[compiled.find_bit(w).index for w in ins.qubits])
    final=compiled.layout.final_index_layout(filter_ancillas=True) if compiled.layout else list(range(n))
    assert sorted(final)==list(range(n));where=list(final);restore=[]
    for logical in range(n):
        position=where[logical]
        if position==logical:continue
        # Restore along the physical graph, never insert an illegal remote SWAP.
        adj={i:set() for i in range(n)}
        for a,b in cmap.get_edges():adj[a].add(b)
        parent={position:None};queue=deque([position])
        while logical not in parent:
            a=queue.popleft()
            for b in sorted(adj[a]):
                if b not in parent:parent[b]=a;queue.append(b)
        path=[logical]
        while path[-1]!=position:path.append(parent[path[-1]])
        path.reverse()
        # Endpoint transposition leaves intermediate locations unchanged.
        pairs=list(zip(path,path[1:]))+list(reversed(list(zip(path[:-1],path[1:-1]))))
        for a,b in pairs:
            physical.cx(a,b);physical.cx(b,a);physical.cx(a,b);restore.append([a,b])
            where=[b if x==a else a if x==b else x for x in where]
        assert where[logical]==logical
    assert where==list(range(n))
    physical=safe_native(physical,1)
    allowed=set(tuple(e) for e in cmap.get_edges())
    assert all(tuple(physical.find_bit(w).index for w in ins.qubits) in allowed for ins in physical.data if len(ins.qubits)==2)
    certificate=dict(initial_logical_layout=list(range(n)),routing_final_layout=final,
                     materialized_restore_SWAPS=restore,restore_CX=3*len(restore),final_logical_layout=list(range(n)),
                     coupling_edges=list(map(list,cmap.get_edges())),all_two_qubit_gates_legal=True,
                     physical_qubits=n,new_work_qubits=0,phase_policy=core.POLICY,backend_policy=BACKEND,routing_translation_preset=0,finite_post_local_rounds=1,no_numeric_two_qubit_synthesis=True,
                     seed=seed,all_repeats_reported=True,not_seed_search=True,qubits_initially_zero=False)
    return physical,certificate


def source_for(case,method,topology,main_root=None):
    if method=='ours_hop':
        ids,decision=A.select_cache(case,topology);raw,plan=A.build(case,cache=ids);one=core.native(raw.copy(),1);q=core.native(one.copy(),3)
        return q,dict(origin='same_frozen_scientific_core_topology_proxy',decision=decision,construction=plan)
    raw,plan=construct(case,method);one=core.native(raw.copy(),1);q=core.native(one.copy(),3)
    return q,dict(origin='same_V2_common_core_rebuild_on_unchanged_prespecified_input',construction=plan)


def run(casefile,method,main_root=None):
    casefile=Path(casefile);casefile=casefile if casefile.is_absolute() else ROOT/casefile
    case=json.loads(casefile.read_text());input_sha=sha(casefile);folder=ROOT/'results/routing_v2'/case['id']/method;folder.mkdir(parents=True,exist_ok=True);rp=folder/'result.json'
    if rp.exists():
        prior=json.loads(rp.read_text())
        assert prior['status']=='passed' and prior['case_id']==case['id'] and prior['method']==method and prior['n']==case['n']
        assert prior['source_sha256']==sha(ROOT/'code/routing_v2.py') and prior['input_sha256']==input_sha
        assert len(prior['rows'])==6 and {(x['topology'],x['seed']) for x in prior['rows']}=={(t,z) for t in ('line','grid') for z in SEEDS}
        for art in prior['rows']:assert sha(ROOT/art['file'])==art['sha256']
        return prior
    start=time.monotonic();rows=[]
    for top in ('line','grid'):
        q,source=source_for(case,method,top,main_root)
        before=G.fingerprint(q);assert {p.name for p in q.parameters}==required_parameters(case)
        for seed in SEEDS:
            t=time.perf_counter();out,cert=routed(q,top,seed);seconds=time.perf_counter()-t;assert G.fingerprint(q)==before
            path=folder/(top+'_'+str(seed)+'.qpy');fp=G.fingerprint(out)
            with path.open('wb') as stream:io.dump(out,stream)
            with path.open('rb') as stream:loaded=io.load(stream)[0]
            assert fp==G.fingerprint(loaded)
            tests=checks(loaded,case,all_columns=False) if case['n']<=16 else []
            regression=[]
            if case['n']<=12:
                rng=np.random.default_rng(20261109);v=rng.normal(size=1<<case['n'])+1j*rng.normal(size=1<<case['n']);v/=np.linalg.norm(v)
                for index in range(4):
                    d=prescribed_values(case,index);args={p:d[p.name] for p in q.parameters};a=q.assign_parameters(args);b=loaded.assign_parameters({p:d[p.name] for p in loaded.parameters});err=float(np.max(abs(Statevector(v.copy()).evolve(a).data-Statevector(v.copy()).evolve(b).data)));assert err<2e-9;regression.append(err)
            rows.append(dict(topology=top,seed=seed,source_metrics=core.costs(q),metrics=core.costs(loaded),route_seconds=seconds,certificate=cert,
                             file=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path),source_provenance=source,canonical_Feas_checks=tests,full_source_regression_errors=regression))
            print(json.dumps(dict(case=case['id'],method=method,topology=top,seed=seed,status='passed',CX=rows[-1]['metrics']['cx'],restoreCX=cert['restore_CX'],seconds=seconds)),flush=True);gc.collect()
    result=dict(status='passed',case_id=case['id'],method=method,n=case['n'],input_sha256=input_sha,rows=rows,seconds=time.monotonic()-start,source_sha256=sha(ROOT/'code/routing_v2.py'),
                all_six_repeats_retained=True,seed_best_selection=False,numerical_scope='Canonical Feas complex inputs on n<=16; full-source random vector regression on n<=12; structural/compiler/permutation certificate otherwise')
    put(rp,result);return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',required=True);a.add_argument('--method',required=True);a.add_argument('--main-root');x=a.parse_args()
    try:run(x.case,x.method,x.main_root)
    except BlockingIOError:raise SystemExit(75)
