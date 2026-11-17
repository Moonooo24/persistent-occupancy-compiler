"""OURS-only shared data flag for different-neighborhood independent targets."""
from pathlib import Path
import sys,os,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='1'
os.environ['QISKIT_PARALLEL']='FALSE';os.environ['QISKIT_NUM_PROCS']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(ROOT.parent/'research/.venv_qiskit/Lib/site-packages'),str(ROOT/'code')]
import mis_multitarget_phase_core_v2 as old
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RCCXGate,RC3XGate
BETA=old.BETA;core=old.core

def build_A(case,T,a,S):
 n=case['n'];T=list(T);S=list(S);adj=core.neighbors(case);assert len(S) in (2,3) and a not in S and len(set(T+[a]+S))==len(T)+len(S)+1
 assert all(j not in adj[i] for i in T for j in T if i!=j);assert all({a,*S}<=set(adj[i]) for i in T)
 CS=QuantumCircuit(n)
 for w in S:CS.x(w)
 CS.append(RCCXGate() if len(S)==2 else RC3XGate(),S+[a])
 for w in S:CS.x(w)
 assert not(set(T)&{CS.find_bit(w).index for row in CS.data for w in row.qubits})
 A=QuantumCircuit(n);A.compose(CS,inplace=True);recipes=[]
 for i in T:
  U=[w for w in adj[i] if w!=a and w not in S];c=len(U);assert c<=len(S)+1
  P=QuantumCircuit(n);work=S[:max(0,c-1)];steps=[]
  if c==0:P.cx(a,i)
  elif c==1:P.x(U[0]);P.ccx(a,U[0],i);P.x(U[0])
  else:
   C=QuantumCircuit(n)
   for u in U:C.x(u)
   C.append(RCCXGate(),U[:2]+[work[0]]);steps.append([U[0],U[1],work[0]])
   for j in range(1,c-1):C.append(RCCXGate(),[work[j-1],U[j+1],work[j]]);steps.append([work[j-1],U[j+1],work[j]])
   assert not(set(T+[a])&{C.find_bit(w).index for row in C.data for w in row.qubits})
   P.compose(C,inplace=True);P.ccx(a,work[-1],i);P.compose(C.inverse(),inplace=True)
  A.compose(P,inplace=True);recipes.append(dict(target=i,N=adj[i],U=U,c=c,work=work,AND_steps=steps,exact_middle=True,compute_does_not_read_a_or_T=True,data_restored=True))
 A.compose(CS.inverse(),inplace=True)
 return A,dict(T=T,a=a,S=S,recipes=recipes,CS='negative_RCCX' if len(S)==2 else 'negative_RC3X',CS_actual_inverse=True,CS_no_target_borrow=True,phasefree_completed_A=True,q_support='each q_i depends only N_i',raw_CX_upper=2*(3 if len(S)==2 else 6)+sum(1 if r['c']==0 else 6 if r['c']==1 else 6*r['c'] for r in recipes))

def build_kernel(case,T,a,S,beta=BETA):
 A,cert=build_A(case,T,a,S);child,child_meta=old.load_partial(case,dict(targets=list(T),controls=[a]),BETA,[a]);assert child_meta['DB_involution'] and child_meta['DB_same_for_both_signs'] and child_meta['DB_on_Lzero_identity'];minus=child.assign_parameters({BETA:-beta/2});plus=child.assign_parameters({BETA:beta/2});q=QuantumCircuit(case['n'])
 for i in T:q.z(i)
 q.compose(minus,inplace=True)
 for i in T:q.h(i)
 q.compose(A,inplace=True)
 for i in T:q.h(i)
 q.compose(plus,inplace=True)
 cert.update(child=child_meta,small_child_parameter_scales=[-.5,.5],raw_CX_upper=cert['raw_CX_upper']+2*len(T),residual='product Z_i^(1-q_i(N_i))',same_recipe_two_sweeps=True)
 return q,A,cert

def q_from_bits(recipe,a,S,bits):
 flag=bits[a]^int(not any(bits[s] for s in S));U=recipe['U'];c=len(U)
 if c==0:f=1
 elif c==1:f=1-bits[U[0]]
 else:
  values=dict(bits);values[recipe['work'][0]]^=(1-bits[U[0]])&(1-bits[U[1]])
  for j in range(1,c-1):values[recipe['work'][j]]^=values[recipe['work'][j-1]]&(1-bits[U[j+1]])
  f=values[recipe['work'][-1]]
 return flag&f

def old_modules(case):
 groups=old.grouping(case);modules=[];records=[]
 for g in groups:
  pool=old.phase_candidates(case,g);selected=min(pool,key=lambda x:(x['cost']['cx'],x['cost']['parameterized_rz'],x['cost']['depth'],x['name']));modules.append(selected['q']);records.append(dict(group=g,selected=selected['name'],cost=selected['cost']))
 return groups,modules,records

def complete_programs(case,T,a,S):
 assert core.target_order(case)[:len(T)]==list(T), 'Shared toy targets must be a contiguous literal prefix'
 groups,modules,records=old_modules(case);T=list(T);assert [g['targets'] for g in groups[:len(T)]]==[[i] for i in T], 'Do not weaken old non-singleton OURS groups'
 shared,A,cert=build_kernel(case,T,a,S);params={k:Parameter(k) for k in ('gamma','beta0','beta1')};parents=QuantumCircuit(case['n']);new=QuantumCircuit(case['n'])
 for sweep in range(2):
  b=params['beta'+str(sweep)]
  for q in modules:parents.compose(q.assign_parameters({BETA:b}),inplace=True)
  new.compose(shared.assign_parameters({BETA:b}),inplace=True)
  for q in modules[len(T):]:new.compose(q.assign_parameters({BETA:b}),inplace=True)
  if sweep==0:
   phase=core.phase(case['n'],params['gamma']);parents.compose(phase,inplace=True);new.compose(phase,inplace=True)
 return parents,new,shared,A,cert,records
