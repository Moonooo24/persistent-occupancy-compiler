"""Mode-B grouped MIS phase extension, separately declared after exact static v1.
Only exact-P one-query or phase-free Cdag CCX C oracles are admitted.
"""
from mis_multitarget_exact_core_v1 import *
from qiskit.synthesis import synth_mcx_1_clean_kg24
PHASE_METHODS=(dict(name='multitarget_phase_grouped',side='proposed'),dict(name='bridge_multitarget_phase_grouped_default',side='proposed'),dict(name='bridge_multitarget_phase_grouped_seq',side='proposed'))


def load_partial(case,group,beta,controls):
 q=QuantumCircuit(case['n']);meta=[];degree=len(controls);here=Path(core.__file__).resolve().parent
 manifest=json.loads((here/'mis_phase_templates_manifest_v5.json').read_text(encoding='utf8'));item=next(x for x in manifest['templates'] if x['family']=='conditional_phase_mixed' and x['degree']==degree);path=here/item['file'];assert core.sha(path)==item['sha256'];f=item['phase_table'];assert f[0]==0 and all(x in (0,1) for x in f)
 import lossless_qpy_io as io
 with path.open('rb') as stream:[template]=io.load(stream)
 for t in group['targets']:
  available=[w for w in range(case['n']) if w!=t and w not in controls];extra=template.num_qubits-degree-1;assert extra<=len(available)
  q.compose(template.assign_parameters({p:beta for p in template.parameters}),[t]+list(controls)+available[:extra],inplace=True)
 return q,dict(type='individual_frozen_partial',controls=list(controls),phase_table=f,DB='Z_T^f_L',DB_diagonal=True,DB_involution=True,DB_same_for_both_signs=True,DB_beta_independent=True,DB_on_Lzero_identity=True,source=str(path),sha256=core.sha(path))

def fallback(case,group,beta):
 if not group['controls']:
  q=QuantumCircuit(case['n'])
  for t in group['targets']:q.rx(2*beta,t)
  return q,dict(type='individual_frozen_partial',q_truth=[1],scope='global_exact')
 q,meta=load_partial(case,group,beta,group['controls']);meta['q_truth']=[1-int(v) for v in meta['phase_table']];return q,meta

def one_query(case,group,beta,P):
 q=QuantumCircuit(case['n'])
 for w in group['controls']:q.x(w)
 for t in group['targets']:q.z(t);q.rx(-beta,t);q.h(t)
 q.compose(P,inplace=True)
 for t in group['targets']:q.h(t);q.rx(beta,t)
 for w in group['controls']:q.x(w)
 return q

def fanout_oracle(case,group,A):
 F=QuantumCircuit(case['n']);a=group['targets'][0]
 for t in group['targets'][1:]:F.cx(a,t)
 return F.compose(A).compose(F)

def exact_oneP_candidates(case,group,beta):
 rows=[]
 for source in ('exact_noaux','exact_dirty_many','exact_dirty_one'):
  for policy in ('external_first','canonical_any'):
   A,cert=embed_A(case,group,source,policy)
   if A is None:continue
   assert not cert['relative_phase'];P=fanout_oracle(case,group,A);raw=one_query(case,group,beta,P);q=core.native(raw,1)
   rows.append(dict(name='oneP_'+source+'_'+policy,q=q,cost=core.costs(q),q_truth=[1]+[0]*((1<<len(group['controls']))-1),certificate=dict(cert,type='global_exact_P_one_query_phase_extension',oracle_phasefree=True,oracle_calls=1,fanout=True,scope='U_group times Z_T^(1-NOR(N))',unsafe_relative_one_query_excluded=True)))
 return rows

def kg_phasefree(case,group,L):
 N=group['controls'];R=[w for w in N if w not in L];a=group['targets'][0];n=case['n'];compute=QuantumCircuit(n);steps=[]
 if len(R)==1:middle=R;kind='cx'
 elif len(R)==2:middle=R;kind='ccx'
 elif len(R)>=3:
  source=synth_mcx_1_clean_kg24(len(R));positions=[k for k,row in enumerate(source.data) if row.operation.name=='ccx'];assert len(positions)==1;position=positions[0];mapping=R+[a,L[0]]
  for row in source.data[:position]:
   wires=[mapping[source.find_bit(w).index] for w in row.qubits];name=row.operation.name;assert a not in wires and name in ('rccx','rccx_dg','x');compute.append(row.operation,wires);steps.append((name,wires))
  middle=[mapping[source.find_bit(w).index] for w in source.data[position].qubits];assert middle[-1]==a;middle=middle[:-1];kind='ccx'
 else:return None,None
 A=QuantumCircuit(n)
 for w in R:A.x(w)
 A.compose(compute,inplace=True)
 if kind=='cx':A.cx(middle[0],a)
 else:A.ccx(*middle,a)
 A.compose(compute.inverse(),inplace=True)
 for w in R:A.x(w)
 truth=[]
 for state in range(1<<len(N)):
  bits={w:(state>>j)&1 for j,w in enumerate(N)}
  for w in R:bits[w]^=1
  for name,wires in steps:
   if name=='x':bits[wires[0]]^=1
   else:bits[wires[2]]^=bits[wires[0]]&bits[wires[1]]
  value=bits[middle[0]] if kind=='cx' else bits[middle[0]]&bits[middle[1]];truth.append(value)
  if not any(state>>N.index(w)&1 for w in L):assert value==int(not any(state>>N.index(w)&1 for w in R))
 assert truth[0]==1
 return A,dict(type='KG_phasefree_control_scratch',L=list(L),R=R,compute_steps=steps,middle_controls=middle,middle='exact_'+kind,compute_does_not_read_any_target=True,dirty_N_restored_for_all_states=True,phasefree=True,q_truth=truth,q_zero=1,source='Qiskit2.1.2 synth_mcx_1_clean_kg24 prefix, exact middle, actual inverse')

def child(case,group,L):
 sub=dict(group,controls=list(L));raw,meta=load_partial(case,sub,BETA,L);q=core.native(raw,1);pool=[dict(name='individual_C'+str(len(L)),q=q,cost=core.costs(q),certificate=meta)]
 A,cert=embed_A(case,sub,'exact_dirty_many','canonical_any');assert A is not None and not cert['relative_phase'];P=fanout_oracle(case,sub,A);raw=one_query(case,sub,BETA,P);q=core.native(raw,1)
 f=[int(x!=0) for x in range(1<<len(L))];pool.append(dict(name='group_exact_oneP_C'+str(len(L)),q=q,cost=core.costs(q),certificate=dict(type='same_group_exact_oneP_child',controls=list(L),phase_table=f,DB='Z_T^OR(L)',DB_diagonal=True,DB_involution=True,DB_same_for_both_signs=True,DB_beta_independent=True,DB_on_Lzero_identity=True,exact_P=cert)))
 winner=min(pool,key=lambda p:(p['cost']['cx'],p['cost']['parameterized_rz'],p['cost']['depth'],p['name']))
 return winner,[{k:v for k,v in p.items() if k!='q'} for p in pool]

def phase_candidates(case,group,beta=BETA):
 # The frozen full-degree fallback bank uses historical padding. Generic minimal
 # widths or degrees outside that bank go directly to the linear constructor.
 degree=len(group['controls'])
 if degree>8 or case['n']<2*degree-1:
  from mis_multitarget_constructive_phase_v1 import constructive_phase_group
  raw,cert=constructive_phase_group(case['n'],group['controls'],group['targets'],beta);q=core.native(raw,1)
  return [dict(name='constructive_minimal_width_KG',q=q,cost=core.costs(q),q_truth=None,certificate=cert)]
 raw,meta=fallback(case,group,beta);q=core.native(raw,1);rows=[dict(name='individual_frozen_C_fallback',q=q,cost=core.costs(q),q_truth=meta['q_truth'],certificate=meta)]
 if not group['controls']:return rows
 rows.extend(exact_oneP_candidates(case,group,beta))
 N=group['controls']
 for size in (1,2):
  if size>=len(N):continue
  for L in sorted({tuple(N[:size]),tuple(N[-size:])}):
   A,cert=kg_phasefree(case,group,L)
   if A is None:continue
   P=fanout_oracle(case,group,A);selected,child_pool=child(case,group,L);Bminus=selected['q'].assign_parameters({BETA:-beta/2});Bplus=selected['q'].assign_parameters({BETA:beta/2});raw=QuantumCircuit(case['n'])
   for t in group['targets']:raw.z(t)
   raw.compose(Bminus,inplace=True)
   for t in group['targets']:raw.h(t)
   raw.compose(P,inplace=True)
   for t in group['targets']:raw.h(t)
   raw.compose(Bplus,inplace=True);q=core.native(raw,1)
   rows.append(dict(name='KG_L'+','.join(map(str,L))+'_'+selected['name'],q=q,cost=core.costs(q),q_truth=cert['q_truth'],certificate=dict(cert,selected_child=selected['certificate'],child_pool=child_pool,actual_child_signs=[-.5,.5],residual='Z_T^(1-q_N)',same_recipe_both_sweeps=True,full_global_even_p2_certificate=True)))
 return rows

def phase_program(case,method='multitarget_phase_grouped'):
 if method not in [x['name'] for x in PHASE_METHODS]:raise ValueError(method)
 groups=grouping(case);choices=[];events=[];params={name:Parameter(name) for name in ('gamma','beta0','beta1')}
 for group in groups:
  pool=phase_candidates(case,group);winner=min(pool,key=lambda x:(x['cost']['cx'],x['cost']['parameterized_rz'],x['cost']['depth'],x['name']));choices.append(winner);assert (winner['q_truth'][0] if winner.get('q_truth') is not None else winner['certificate']['q_zero'])==1
  events.append(dict(group,selected_name=winner['name'],selected_cost=winner['cost'],q_truth=winner['q_truth'],candidate_rows=[{k:v for k,v in p.items() if k!='q'} for p in pool],residual='Z_T^(1-q_N)',q_zero=1,diagonal_involution=True))
 q=QuantumCircuit(case['n'])
 for sweep in range(2):
  for item in choices:q.compose(item['q'].assign_parameters({BETA:params['beta'+str(sweep)]}),inplace=True)
  if sweep==0:q.compose(core.phase(case['n'],params['gamma']),inplace=True)
 meta=dict(method=method,side='proposed',groups=events,events=events,params=params,target_order=core.target_order(case),new_work=0,scope='Feas exact each layer; full-Hilbert exact for fixed identical two-sweep recipes',global_even_certificate=dict(full_Hilbert_exact_to_canonical_program=True,type='central_Z_T_group_residuals_even_two_sweep',proof='First regroup exact canonical Ui using certified nonadjacent swaps. Each chosen W=Ui_group D with diagonal involution D=Z_T^(1-q_N), q_N(0)=1. D commutes with every canonical U_j: j in T requires N0; j in N requires allT0; elsewhere changes neither T nor N. All Ds and cost commute; identical two sweeps cancel each D squared. No movement of relaxed blocks is assumed.'))
 if method.startswith('bridge_'):
  from mis_phase_native_bridge import bridge
  q,cert=bridge(core.native(q,1),method.rsplit('_',1)[1],'clifford_normalized');meta['bridge']=cert
 return q,meta
