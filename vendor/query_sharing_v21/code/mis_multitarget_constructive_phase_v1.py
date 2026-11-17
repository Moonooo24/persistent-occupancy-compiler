"""O(d+m)-size constructive grouped phase kernel, no full-degree template/table.
Public construction for minimal n=d+m; benchmark catalogs are unchanged.
"""
from mis_multitarget_exact_core_v1 import *
from mis_multitarget_phase_core_v2 import load_partial,fanout_oracle
from qiskit.synthesis import synth_mcx_1_clean_kg24

def constructive_phase_group(n,controls,targets,beta,L_size=None,layout='prefix'):
 N=sorted(controls);T=list(targets);d=len(N);m=len(T)
 assert m>=1 and len(set(N+T))==d+m and all(0<=w<n for w in N+T) and n>=d+m
 case=dict(n=n);group=dict(controls=N,targets=T)
 if d==0:
  q=QuantumCircuit(n)
  for t in T:q.rx(2*beta,t)
  return q,dict(recipe='unconditional',q_zero=1,support_N_only=True,q_truth_enumeration_used=False,new_work=0,physical_width=n)
 if d<=2:
  q,child=load_partial(case,group,beta,N)
  return q,dict(recipe='constant_arity_partial_C'+str(d),child=child,q_zero=1,support_N_only=True,fixed_small_phase_table=child['phase_table'],q_truth_enumeration_used=False,new_work=0,physical_width=n)
 k=L_size if L_size is not None else (2 if m==1 else 1);assert k in (1,2) and k<d;assert layout in ('prefix','suffix');L=tuple(N[:k] if layout=='prefix' else N[-k:]);R=[w for w in N if w not in L];a=T[0];compute=QuantumCircuit(n);steps=[]
 if len(R)==1:middle=R;kind='cx'
 elif len(R)==2:middle=R;kind='ccx'
 else:
  source=synth_mcx_1_clean_kg24(len(R));positions=[i for i,row in enumerate(source.data) if row.operation.name=='ccx'];assert len(positions)==1;position=positions[0];mapping=R+[a,L[0]]
  for row in source.data[:position]:
   wires=[mapping[source.find_bit(w).index] for w in row.qubits];assert not(set(wires)&set(T)) and row.operation.name in ('rccx','rccx_dg','x');compute.append(row.operation,wires);steps.append((row.operation.name,wires))
  middle=[mapping[source.find_bit(w).index] for w in source.data[position].qubits];assert middle[-1]==a;middle=middle[:-1];kind='ccx'
 A=QuantumCircuit(n)
 for w in R:A.x(w)
 A.compose(compute,inplace=True)
 if kind=='cx':A.cx(middle[0],a)
 else:A.ccx(*middle,a)
 A.compose(compute.inverse(),inplace=True)
 for w in R:A.x(w)
 P=fanout_oracle(case,group,A);B,child=load_partial(case,group,BETA,L);minus=B.assign_parameters({BETA:-beta/2});plus=B.assign_parameters({BETA:beta/2});q=QuantumCircuit(n)
 for t in T:q.z(t)
 q.compose(minus,inplace=True)
 for t in T:q.h(t)
 q.compose(P,inplace=True)
 for t in T:q.h(t)
 q.compose(plus,inplace=True)
 certificate=dict(recipe='constructive_KG_L'+str(k),controls=N,targets=T,L=list(L),R=R,anchor=a,compute_steps=steps,middle_controls=middle,middle='exact_'+kind,phasefree_oracle=True,compute_does_not_read_any_target=True,dirty_N_restored=True,support_N_only=True,q_zero=1,q_truth_enumeration_used=False,clean_contract='On L0, the official one-clean KG compute realizes AND of opened R. Else arbitrary q_N is allowed; exact middle and actual inverse restore all N.',q_when_Lzero='NOR(R)',residual='Z_T^(1-q_N)',child=child,new_work=0,physical_width=n,even_two_sweep_global_equality=True)
 if len(R)>=2:certificate['constructive_raw_CX_upper']=6*len(R)-6+2*(m-1)+2*m*(1 if k==1 else 3)
 return q,certificate

def evaluate_q(certificate,assignment):
 N=certificate['controls'];bits={w:(assignment>>j)&1 for j,w in enumerate(N)}
 for w in certificate['R']:bits[w]^=1
 for name,wires in certificate['compute_steps']:
  if name=='x':bits[wires[0]]^=1
  else:bits[wires[2]]^=bits[wires[0]]&bits[wires[1]]
 c=certificate['middle_controls'];return bits[c[0]] if certificate['middle']=='exact_cx' else bits[c[0]]&bits[c[1]]
