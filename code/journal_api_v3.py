"""User-facing entry: certified representations and shared-query alternatives."""
import time
from runtime import core
import auto_reuse_v2 as A
import query_sharing_v2 as Q


def compile(problem,*,target=None,include_shared=False,routing_seed=20261109):
    """Return selected complete circuit, its proof metadata, and all measured variants.

    Selection is part of compilation on this input, not tuning against test labels.
    Source-proxy optimality applies only to the selective-cache candidate.
    """
    start=time.perf_counter();cache,decision=A.select_cache(problem,target);candidates=[]
    eligible={k for k,c in enumerate(A.graph_spec(problem)['parts']) if len(c)>1}
    specs=[('no_cache',set()),('all_cache',eligible),('selected_cache',cache)]
    compiled_sets={}
    for name,selection in specs:
        key=frozenset(selection)
        if key in compiled_sets:
            parent=compiled_sets[key]
            candidates.append(dict(name=name,circuit=parent['circuit'],proof=parent['proof'],metrics=parent['metrics'],alias_of=parent['name']))
            continue
        raw,proof=A.build(problem,cache=selection);one=core.native(raw.copy(),1);out=core.native(one.copy(),3)
        row=dict(name=name,circuit=out,proof=proof,metrics=core.costs(out));compiled_sets[key]=row;candidates.append(row)
    if include_shared:
        raw,proof=Q.build(problem);one=core.native(raw.copy(),1);out=core.native(one.copy(),3)
        candidates.append(dict(name='shared_query',circuit=out,proof=proof,metrics=core.costs(out)))
    if target is not None:
        from routing_v2 import routed
        routed_names={}
        for item in candidates:
            item['logical_metrics']=item['metrics']
            if item.get('alias_of') in routed_names:
                original=routed_names[item['alias_of']]
                item['circuit']=original['circuit'];item['routing_certificate']=original['routing_certificate']
            else:
                item['circuit'],item['routing_certificate']=routed(item['circuit'],target,routing_seed)
            item['metrics']=core.costs(item['circuit']);routed_names[item['name']]=item
    winner=min(candidates,key=lambda x:(x['metrics']['cx'],x['metrics']['parameterized_rz'],x['metrics']['depth'],x['name']))
    certificate=dict(selected_route=winner['name'],selected_proof=winner['proof'],source_cache_decision=decision,
                     measured_candidates=[dict(name=x['name'],metrics=x['metrics'],logical_metrics=x.get('logical_metrics',x['metrics']),alias_of=x.get('alias_of')) for x in candidates],
                     compilation_seconds=time.perf_counter()-start,target_proxy=target,distinct_cache_variants=len(compiled_sets),duplicate_candidates_not_independent=True,
                     actual_topology_routing_performed=target is not None,routing_seed=routing_seed,no_seed_selection=True,selected_routing_certificate=winner.get('routing_certificate'),scope='Exact prescribed dynamics on original Feas and physical I/O')
    return winner['circuit'],certificate,candidates
