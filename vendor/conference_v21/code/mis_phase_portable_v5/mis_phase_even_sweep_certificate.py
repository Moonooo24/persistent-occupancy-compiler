"""Sufficient full-Hilbert even-sweep certificate for canonical MIS programs."""
def certify(case,events):
    n=case['n']
    order=list(case.get('target_order',range(n)))
    if len(events)!=2*n:return dict(status='not_applicable_no_local_event_trace')
    if [e['target'] for e in events]!=order+order:raise ValueError('Event sequence differs from the frozen two-sweep program')
    edges={tuple(sorted(e)) for e in case['graph_edges']}
    local_parity={}
    or_parity=[0]*n
    free_parity={}
    for e in events:
        target=e['target'];kind=e['kind']
        if 'phase_table' in e:
            values=e['phase_table'];order=e['order']
            if len(values)!=(1<<len(order)) or values[0]!=0 or any(x not in (0,1) for x in values):raise ValueError('Invalid local blocking phase')
            if set(order)!={b if a==target else a for a,b in edges if target in (a,b)}:raise ValueError('Phase depends on non-neighbors')
            key=(target,tuple(order),tuple(values))
            local_parity[key]=local_parity.get(key,0)^1
        elif kind.startswith('one_mcx_'):or_parity[target]^=1
        elif kind in ('frame','open_walsh'):
            delta=e['L']^e['M']
            for other in range(n):
                if delta>>other&1 and tuple(sorted((target,other))) in edges:
                    edge=tuple(sorted((target,other)))
                    free_parity[edge]=free_parity.get(edge,0)^1
    nonfree_closed=True
    for a in range(n):
        for b in range(a+1,n):
            if (a,b) in edges:continue
            previous=0
            for e in events:
                if e['target'] not in (a,b):continue
                other=b if e['target']==a else a
                before=(e.get('L',0)>>other)&1
                after=(e.get('M',0)>>other)&1
                if before!=previous:nonfree_closed=False
                previous=after
            if previous:nonfree_closed=False
    remaining_or=[i for i,p in enumerate(or_parity) if p]
    remaining_edges=[list(e) for e,p in sorted(free_parity.items()) if p]
    remaining_local=[dict(target=k[0],order=list(k[1]),phase_table=list(k[2])) for k,p in local_parity.items() if p]
    exact=nonfree_closed and not remaining_or and not remaining_edges and not remaining_local
    return dict(status='certified_global_even_sweep_identity' if exact else 'feasible_only_prior_certificate',
                full_Hilbert_exact_to_canonical_program=exact,
                canonical_predicate='NOR of every original graph neighbor',
                edge_violation_projectors_conserved=True,parameter_independent_involutions=True,
                nonfree_frames_continuous_and_closed=nonfree_closed,
                residual_local_blocking_phases=remaining_local,residual_OR_phase_targets=remaining_or,residual_free_CZ_edges=remaining_edges,
                theorem='Each violated-edge projector commutes with every canonical local mixer. Any (-1)^(x_i f(neighbors)) with f(0)=0, D_OR, and original-edge CZ phases commute with all mixers and diagonal cost; their even multiplicities cancel without reordering the mixers.',
                sufficient_not_necessary=True,no_large_statevector_or_Operator=True,
                boundary='Not asserted for arbitrary Care/LP/NF/nonlocal completions or for omitted zero-angle phase kernels.')

