"""Local MIS kernels and literal two-sweep programs; zero added work."""
from pathlib import Path
import os,sys,math,hashlib,copy,importlib.util,json
from fractions import Fraction
from types import SimpleNamespace
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['QISKIT_PARALLEL']='FALSE'
os.environ['RUST_MIN_STACK']='67108864'
if os.environ.get('MIS_PHASE_SITE_PACKAGES'):
    sys.path.insert(0,os.environ['MIS_PHASE_SITE_PACKAGES'])
from qiskit import QuantumCircuit,transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Clifford
from qiskit.synthesis import synth_cnot_phase_aam,synth_mcx_noaux_v24,synth_mcx_n_dirty_i15,synth_mcx_1_dirty_kg24
import mis_phase_frame_core as frame
from mis_phase_numeric_policy import phase_policy,POLICY
BASIS=['rz','sx','x','cx']
SEED=20261101
CE_SHA='72596242fea34c4f3fcaeee6bcc046d89e5c203d9dae3cfeba3ef324741d17a7'
EXACT=('mcrx','walsh_target','walsh_highest','walsh_aam','two_mcx_noaux','two_mcx_dirty_many','two_mcx_dirty_one')
PHASE=('open_walsh','one_mcx_noaux','one_mcx_dirty_many','one_mcx_dirty_one')
TEMPLATE_METHODS=('rphase_pair_fixed','generic_exact_factor','matched_conditional_exact','phasefree_exact_factor','structured_exact_mixed','conditional_phase_mixed','structured_extension_mixed','qibo_vale_symbolic_rx')
METHODS=EXACT+('exact_native_mixed','closed_cycle','quclear_default','quclear_seq')+PHASE+('phase_native_mixed','frame_local_open','frame_coordinate')+TEMPLATE_METHODS
BRIDGE_METHODS=tuple('bridge_'+source+'_'+mode for source in ('rphase_pair_fixed','structured_exact_mixed','structured_extension_mixed') for mode in ('default','seq'))
METHODS=METHODS+BRIDGE_METHODS
P_CACHE={}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def native(qc,level=1):
    ledger=[]
    with phase_policy(ledger):
        compiled=transpile(qc,basis_gates=BASIS,optimization_level=level,seed_transpiler=SEED,qubits_initially_zero=False)
    permutation=compiled.layout.final_index_layout(filter_ancillas=True) if compiled.layout is not None else list(range(qc.num_qubits))
    assert sorted(permutation)==list(range(qc.num_qubits))
    physical=QuantumCircuit(qc.num_qubits);physical.global_phase=compiled.global_phase
    for instruction in compiled.data:physical.append(instruction.operation,[compiled.find_bit(w).index for w in instruction.qubits],[])
    locations=list(permutation);swaps=[]
    for logical in range(qc.num_qubits):
        position=locations[logical]
        if position==logical:continue
        physical.cx(position,logical);physical.cx(logical,position);physical.cx(position,logical);swaps.append([position,logical])
        locations=[logical if x==position else position if x==logical else x for x in locations]
    assert locations==list(range(qc.num_qubits))
    physical.metadata=dict(final_index_layout=permutation,materialized_swap_pairs=swaps,materialized_cx=3*len(swaps),compiler_phase_policy=POLICY,compiler_phase_ledger=[{k:v for k,v in row.items() if k not in ('phase_before','phase_upstream','phase_stable')} for row in ledger])
    return physical
def costs(qc):
    counts=qc.count_ops()
    return dict(cx=int(counts.get('cx',0)),qubits=qc.num_qubits,depth=int(qc.depth()),
                two_qubit_depth=int(qc.depth(filter_function=lambda x:len(x.qubits)==2)),
                rz=int(counts.get('rz',0)),total_1q=sum(len(x.qubits)==1 for x in qc.data),parameterized_rz=sum(x.operation.name=='rz' and bool(getattr(x.operation.params[0],'parameters',set())) for x in qc.data))
def neighbors(case):
    result=[[] for _ in range(case['n'])]
    for a,b in case['graph_edges']:
        result[a].append(b);result[b].append(a)
    return [sorted(v) for v in result]
def canonical_noncontrols(n,target,controls,count):
    candidates=[q for q in range(n) if q!=target and q not in controls]
    if len(candidates)<count:raise ValueError('Insufficient existing data to borrow')
    return candidates[:count]
def positive_mcx(degree,scheme):
    key=(degree,scheme)
    if key not in P_CACHE:
        if scheme=='noaux':
            raw=synth_mcx_noaux_v24(degree)
        elif scheme=='dirty_many':
            raw=synth_mcx_n_dirty_i15(degree,relative_phase=False,action_only=False)
        elif scheme=='dirty_one':
            raw=synth_mcx_1_dirty_kg24(degree) if degree>2 else synth_mcx_noaux_v24(degree)
        else:raise ValueError(scheme)
        P_CACHE[key]=native(raw)
    return P_CACHE[key]
def walsh_raw(n,target,order,beta,backend):
    degree=len(order)
    coeff={m:Fraction(1,1<<degree) for m in range(1<<degree)}
    if backend in ('target','highest'):
        qc=QuantumCircuit(n);qc.h(target);groups={}
        for mask,c in coeff.items():
            full=(1<<target)|sum(1<<w for j,w in enumerate(order) if mask>>j&1)
            anchor=target if backend=='target' else full.bit_length()-1
            controls=full^(1<<anchor)
            rank=frame.inverse_gray(mask if backend=='target' else controls)
            groups.setdefault(anchor,[]).append((rank,controls,c))
        for anchor,terms in sorted(groups.items()):
            previous=0
            for _,controls,c in sorted(terms):
                for w in range(n):
                    if (previous^controls)>>w&1:qc.cx(w,anchor)
                qc.rz(2*beta*float(c),anchor);previous=controls
            for w in range(n):
                if previous>>w&1:qc.cx(w,anchor)
        qc.h(target)
        return qc
    masks=sorted((1<<target)|sum(1<<w for j,w in enumerate(order) if m>>j&1) for m in coeff)
    tags=[(j+1)/(len(masks)+1) for j in range(len(masks))]
    lookup=dict(zip(tags,masks))
    matrix=[[int(mask>>w&1) for mask in masks] for w in range(n)]
    network=synth_cnot_phase_aam(matrix,list(tags),section_size=2)
    linear=[1<<w for w in range(n)]
    seen=[]
    qc=QuantumCircuit(n);qc.h(target)
    for ins in network.data:
        wires=[network.find_bit(q).index for q in ins.qubits]
        if ins.operation.name=='cx':
            qc.cx(*wires);linear[wires[1]]^=linear[wires[0]]
        else:
            assert ins.operation.name=='p'
            tag=float(ins.operation.params[0])
            mask=lookup[tag];assert linear[wires[0]]==mask
            seen.append(mask);qc.rz(2*beta/float(1<<degree),wires[0])
    assert sorted(seen)==masks and linear==[1<<w for w in range(n)] and float(network.global_phase)==0.
    qc.h(target)
    return qc
TEMPLATE_CACHE={}
def template_local(case,target,beta,method):
    import lossless_qpy_io as io
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/'mis_phase_templates_manifest_v5.json').read_text(encoding='utf-8'))
    order=neighbors(case)[target];degree=len(order)
    item=next(r for r in manifest['templates'] if r['family']==method and r['degree']==degree)
    path=here/item['file']
    assert sha(path)==item['sha256']
    if str(path) not in TEMPLATE_CACHE:
        with path.open('rb') as stream:[q]=io.load(stream)
        TEMPLATE_CACHE[str(path)]=q
    template=TEMPLATE_CACHE[str(path)]
    borrowed=canonical_noncontrols(case['n'],target,order,item['extra_existing_data'])
    mapping=[target]+order+borrowed
    assert len(mapping)==template.num_qubits and len(set(mapping))==len(mapping)
    qc=QuantumCircuit(case['n'])
    bound=template.assign_parameters({p:beta for p in template.parameters})
    qc.compose(bound,mapping,inplace=True)
    return qc,dict(target=target,order=order,method=method,scope=item['scope'],new_work=0,
         borrowed_data=[mapping[w] for w in item['borrowed_local_wires']],idle_template_padding=[mapping[w] for w in item['idle_template_padding_not_scratch']],local_to_physical=mapping,phase_table=item['phase_table'],
         development_selected_name=item['selected_name'],frozen_local_cost=item['cost'],
         template_sha256=item['sha256'],recipe=item['recipe'],
         control_scratch_local=item['recipe'].get('scratch',[]),
         control_scratch_physical=[mapping[w] for w in item['recipe'].get('control_scratch',item['recipe'].get('scratch',[]))],
         arbitrary_initial_scratch_and_borrowed_restored=True,
         primitive_contract='Explicit relative-phase compute followed by its actual inverse; any dirty controls restored. No implicit zero ancilla.')
def local(case,target,beta,method):
    if method in TEMPLATE_METHODS:return template_local(case,target,beta,method)
    n=case['n'];order=neighbors(case)[target];degree=len(order)
    cert=dict(target=target,order=order,method=method,
              coefficients={str(m):str(Fraction(1,1<<degree)) for m in range(1<<degree)},
              scope='global_exact' if method in EXACT else 'feasible_exact',new_work=0)
    if method.startswith('walsh_'):
        return walsh_raw(n,target,order,beta,method.removeprefix('walsh_')),cert
    if method=='open_walsh':
        item=dict(order=order,coefficients=cert['coefficients'],name=method)
        route=frame.routes_from_items(n,[item])[0]
        route['target']=target
        Es=sum(1<<w for w in order)
        route=frame.best_cycle(route,E=Es)
        L=route['sequence'][0]&Es;M=route['sequence'][-1]&Es
        cert.update(L=L,M=M,local_phase='CZ stars on budget-forbidden edges')
        return frame.raw_module(n,route,beta,L,M),cert
    qc=QuantumCircuit(n)
    for w in order:qc.x(w)
    if method=='mcrx':
        qc.mcrx(2*beta,order,target,use_basis_gates=True)
    else:
        scheme=method.split('_mcx_',1)[1]
        P=positive_mcx(degree,scheme)
        borrowed=canonical_noncontrols(n,target,order,P.num_qubits-degree-1)
        cert.update(borrowed_data=borrowed,primitive=scheme,dirty_restoration='Exact MCX tensor I on arbitrary borrowed data',
                    primitive_flags=dict(relative_phase=False,action_only=False),oracle_calls=2 if method.startswith('two_') else 1)
        mapping=order+[target]+borrowed
        if method.startswith('two_'):
            qc.h(target);qc.rz(beta,target);qc.compose(P,mapping,inplace=True)
            qc.rz(-beta,target);qc.compose(P,mapping,inplace=True);qc.h(target)
        else:
            qc.z(target);qc.rx(-beta,target);qc.h(target);qc.compose(P,mapping,inplace=True);qc.h(target);qc.rx(beta,target)
            cert['outside_phase']='(-1)^(x_target*OR(neighbor bits))'
    for w in order:qc.x(w)
    return qc,cert
def phase(n,gamma):
    qc=QuantumCircuit(n)
    qc.global_phase+=gamma*n/2
    for w in range(n):qc.rz(gamma,w)
    return qc
def items(case):
    return [dict(name='unique_local_NOR',order=order,
                 coefficients={str(m):str(Fraction(1,1<<len(order))) for m in range(1<<len(order))})
            for order in neighbors(case)]
def scalar_certificate(qc,strict=False):
    """Restore the constant Clifford scalar with phase-preserving polynomial CH."""
    from mis_phase_ch_scalar import scalar_identity
    zero=qc.assign_parameters({p:0. for p in qc.parameters})
    cert=scalar_identity(zero)
    qc.global_phase-=cert['eighth_root_k']*math.pi/4
    cert.update(zero_angle_tableau_identity=True,physical_tail_restored=True,
                identity_cost_phase_gamma_n_over_2_restored=True,angle_dependent_alignment=False,
                scalar_independent_of_parameters=True,global_scalar_restored=True,
                large_full_complex_tested=False,strict_argument_kept_for_compatibility=True,
                reason='Author decisions use only Pauli strings and signed RZ angles; the Clifford conversion discrepancy is parameter independent. CH retains omega and identifies the certified eighth root without a statevector.')
    return cert
def target_order(case):
    result=list(case.get('target_order',range(case['n'])))
    assert sorted(result)==list(range(case['n']))
    return result

def quclear(case,params,mode,strict=False):
    path=Path(__file__).resolve().parent/'mis_phase_CE_module.py'
    assert sha(path)==CE_SHA
    spec=importlib.util.spec_from_file_location('mis_phase_official_ce',path)
    CE=importlib.util.module_from_spec(spec);spec.loader.exec_module(CE)
    n=case['n'];paulis=[];angles=[]
    for sweep in range(2):
        neighbor_map=neighbors(case)
        for target in target_order(case):
            order=neighbor_map[target]
            for mask in sorted(range(1<<len(order)),key=frame.inverse_gray):
                phys=sum(1<<w for j,w in enumerate(order) if mask>>j&1)
                word=''.join('X' if w==target else 'Z' if phys>>w&1 else 'I' for w in reversed(range(n)))
                paulis.append(word);angles.append(2*params['beta'+str(sweep)]/float(1<<len(order)))
        if sweep==0:
            for target in range(n):
                paulis.append(''.join('Z' if w==target else 'I' for w in reversed(range(n))))
                angles.append(params['gamma'])
    fn=CE.CE_recur_tree if mode=='default' else CE.CE_recur_tree_seq
    reduced,tail,ordered=fn(entanglers=paulis,params=angles,barrier=False,threshold=1)
    tailqc=tail.to_circuit() if isinstance(tail,Clifford) else tail
    qc=reduced.compose(tailqc)
    qc.global_phase+=params['gamma']*n/2
    cert=scalar_certificate(qc,strict)
    return qc,dict(method='quclear_'+mode,author_source_sha256=CE_SHA,physical_tail=costs(tailqc),scalar=cert,
                   ordered_terms=len(ordered),scope='global_exact_with_CH_scalar_certificate')
def certify_frames(case,routes,plan,mode):
    free={tuple(sorted(e)) for e in case['graph_edges']}
    if mode in ('closed','local_open'):
        for r,L,M in zip(routes,plan['before'],plan['after']):
            if mode=='closed':assert L==M==0
            else:
                for mask in (L,M):
                    assert all(tuple(sorted((r['target'],w))) in free for w in range(case['n']) if mask>>w&1)
    else:
        for rec in plan['edge_records']:
            edge=tuple(rec['edge'])
            assert rec['free']==(edge in free)
            if not rec['free']:
                assert rec['initial']==rec['final']==0
                prev=0
                for k,L,M in rec['path']:
                    assert L==prev
                    prev=M
                assert prev==0
            else:
                assert rec['independent_between_events']
    return dict(status='compositional_frame_certificate',
                safe_edges_are_original_MIS_budgets=True,cost_phase_commutes_with_CZ_frames=True,
                nonfree_frames_close=True,free_frame_resets_act_as_identity_on_F=True,
                module_identity='raw(L,M)=CZ_star(M) U_exact CZ_star(L)',
                global_complex_measured=False)
def program(case,method,strict_scalar=False):
    if method not in METHODS:raise ValueError(method)
    n=case['n']
    params={name:Parameter(name) for name in ('gamma','beta0','beta1')}
    if method.startswith('bridge_'):
        from mis_phase_native_bridge import bridge
        source,mode=method.removeprefix('bridge_').rsplit('_',1)
        q,meta=program(case,source,strict_scalar)
        result,cert=bridge(native(q,1),mode,'clifford_normalized')
        return result,dict(meta,method=method,bridge=cert,bridge_source=source)
    if method.startswith('quclear_'):
        q,cert=quclear(case,params,method.removeprefix('quclear_'),strict_scalar)
        return q,dict(method=method,params=params,target_order=target_order(case),quclear=cert,events=[])
    events=[]
    qc=QuantumCircuit(n)
    if method in ('closed_cycle','frame_local_open','frame_coordinate'):
        original_routes=frame.routes_from_items(n,items(case))
        base=[original_routes[t] for t in target_order(case)]
        routes=copy.deepcopy(base+base)
        mode={'closed_cycle':'closed','frame_local_open':'local_open','frame_coordinate':'frame_coordinate'}[method]
        routes,plan=frame.plan_routes(n,routes,{tuple(sorted(e)) for e in case['graph_edges']},mode,iterations=8)
        cert=certify_frames(case,routes,plan,mode)
        for k,route in enumerate(routes):
            angle=params['beta'+str(k//n)]
            L,M=plan['before'][k],plan['after'][k]
            qc.compose(frame.raw_module(n,route,angle,L,M),inplace=True)
            events.append(dict(target=route['target'],kind='frame',L=L,M=M))
            if k==n-1:qc.compose(phase(n,params['gamma']),inplace=True)
        from mis_phase_even_sweep_certificate import certify
        return qc,dict(method=method,params=params,target_order=target_order(case),events=events,frame_plan=plan,certificate=cert,global_even_certificate=certify(case,events))
    choices={}
    for target in range(n):
        pool=EXACT if method=='exact_native_mixed' else EXACT+PHASE if method=='phase_native_mixed' else (method,)
        candidates=[]
        for candidate in pool:
            q,cert=local(case,target,params['beta0'],candidate)
            c=costs(native(q))
            candidates.append((c['cx'],c['parameterized_rz'],c['depth'],candidate,q,cert))
        selected=min(candidates,key=lambda row:row[:4])
        choices[target]=selected
    for sweep in range(2):
        for target in target_order(case):
            _,_,_,chosen,q,cert=choices[target]
            part=q if sweep==0 else q.assign_parameters({params['beta0']:params['beta1']})
            qc.compose(part,inplace=True)
            events.append(dict(target=target,kind=chosen,**{k:v for k,v in cert.items() if k not in ('target','method')}))
        if sweep==0:qc.compose(phase(n,params['gamma']),inplace=True)
    from mis_phase_even_sweep_certificate import certify
    return qc,dict(method=method,params=params,target_order=target_order(case),events=events,global_even_certificate=certify(case,events),
                   certificate=dict(local_unique_guard_full_care=True,arbitrary_dirty_data_restored=True,
                                    local_phase_identity_on_all_MIS_feasible_states=True,
                                    mathematical_inference='Every completed macro preserves F; ordered composition and the diagonal cost preserve the exact restricted operator.'))
def bind(qc,values):
    return qc.assign_parameters({p:dict(zip(('gamma','beta0','beta1'),values))[p.name] for p in qc.parameters})

