"""Strong exact false-twin multi-target supplement; frozen MIS assets are read-only.
Known multi-target fanout/relative-phase pair adaptation, not a new proposal.
"""
from pathlib import Path
import os,sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=Path(os.environ.get('MIS_MULTITARGET_BASE_BUNDLE',str(ROOT/'code/mis_phase_portable_v5')))
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='1'
os.environ['QISKIT_PARALLEL']='FALSE'
site=ROOT.parent/'research/.venv_qiskit/Lib/site-packages'
if site.exists():sys.path.insert(0,str(site))
sys.path.insert(0,str(BUNDLE))
import mis_phase_core as core
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RCCXGate,RC3XGate
from qiskit.synthesis import synth_mcx_n_dirty_i15,synth_mcx_noaux_v24,synth_mcx_1_dirty_kg24
METHODS=(dict(name='multitarget_rphase_exact',side='static'),dict(name='bridge_multitarget_rphase_exact_default',side='static'),dict(name='bridge_multitarget_rphase_exact_seq',side='static'))
BETA=Parameter('multitarget_beta')
A_SOURCES=('I15_relative_full','RCCX','RC3X','exact_noaux','exact_dirty_many','exact_dirty_one')

def grouping(case):
 adjacency=[set(v) for v in core.neighbors(case)];remaining=core.target_order(case);groups=[]
 while remaining:
  first=remaining[0];N=adjacency[first];block=next((j for j,v in enumerate(remaining[1:],1) if v in N),len(remaining));T=[v for v in remaining[:block] if adjacency[v]==N]
  assert first==T[0] and all(v not in N for v in T)
  work=list(remaining);swaps=[]
  for dest,t in enumerate(T):
   pos=work.index(t)
   while pos>dest:
    crossed=work[pos-1];assert crossed not in adjacency[t];swaps.append([crossed,t]);work[pos-1],work[pos]=work[pos],work[pos-1];pos-=1
  assert work[:len(T)]==T
  groups.append(dict(targets=T,controls=sorted(N),first_blocker=remaining[block] if block<len(remaining) else None,legal_adjacent_commuting_swaps=swaps,all_open_neighborhoods_identical=True,scope='global_exact'))
  remaining=work[len(T):]
 assert sorted(t for g in groups for t in g['targets'])==list(range(case['n']))
 return groups

def primitive(degree,name):
 if name=='I15_relative_full':return synth_mcx_n_dirty_i15(degree,relative_phase=True,action_only=False)
 if name=='RCCX':
  if degree!=2:return None
  q=QuantumCircuit(3);q.append(RCCXGate(),range(3));return q
 if name=='RC3X':
  if degree!=3:return None
  q=QuantumCircuit(4);q.append(RC3XGate(),range(4));return q
 if name=='exact_noaux':return synth_mcx_noaux_v24(degree)
 if name=='exact_dirty_many':return synth_mcx_n_dirty_i15(degree,relative_phase=False,action_only=False)
 if name=='exact_dirty_one':return synth_mcx_1_dirty_kg24(degree) if degree>2 else synth_mcx_noaux_v24(degree)
 raise ValueError(name)

def embed_A(case,group,name,dirty_policy='canonical_any'):
 n=case['n'];N=group['controls'];anchor=group['targets'][0];raw=primitive(len(N),name)
 if raw is None:return None,None
 needed=raw.num_qubits-len(N)-1;available=[w for w in range(n) if w!=anchor and w not in N]
 if needed>len(available):return None,dict(reason='insufficient physical existing dirty data',needed=needed,available=len(available))
 external=[w for w in available if w not in group['targets']]
 if dirty_policy=='external_first' and len(external)<needed:return None,dict(reason='insufficient external-to-T existing dirty data',needed=needed,available_external=len(external))
 borrowed=(external if dirty_policy=='external_first' else available)[:needed];q=QuantumCircuit(n);q.compose(raw,N+[anchor]+borrowed,inplace=True)
 return q,dict(primitive=name,dirty_policy=dirty_policy,relative_phase=name in ('I15_relative_full','RCCX','RC3X'),action_only=False,borrowed_existing_data=borrowed,borrowed_group_targets=[t for t in borrowed if t in group['targets']],no_added_qubits=True,arbitrary_initial_dirty_data_restored=True,permutation='X_anchor^AND(controls) up to arbitrary diagonal')

def group_raw(case,group,beta,A,inverse_first=False):
 n=case['n'];T=group['targets'];N=group['controls'];anchor=T[0];first=A.inverse() if inverse_first else A
 fan=QuantumCircuit(n)
 for t in T[1:]:fan.cx(anchor,t)
 P=fan.compose(first).compose(fan)
 q=QuantumCircuit(n)
 for w in N:q.x(w)
 for t in T:q.h(t)
 for t in T:q.rz(beta,t)
 q.compose(P,inplace=True)
 for t in T:q.rz(-beta,t)
 q.compose(P.inverse(),inplace=True)
 for t in T:q.h(t)
 for w in N:q.x(w)
 return q

def individual(case,group,beta,dirty_policy='canonical_any'):
 q=QuantumCircuit(case['n']);certs=[]
 for t in group['targets']:
  if not group['controls']:
   part=QuantumCircuit(case['n']);part.rx(2*beta,t);cert=dict(target=t,method='unconditional_rx',scope='global_exact')
  else:
   part,cert=core.local(case,t,beta,'structured_exact_mixed')
   if dirty_policy=='external_first':
    original=cert['local_to_physical'];degree=len(group['controls']);needed=len(original)-degree-1
    external=[w for w in range(case['n']) if w not in group['targets'] and w not in group['controls']]
    if len(external)<needed:return None,dict(reason='insufficient external existing data for individual frozen template',target=t,needed=needed)
    mapped=original[:degree+1]+external[:needed];perm=[None]*case['n']
    for src,dst in zip(original,mapped):perm[src]=dst
    unused=iter(w for w in range(case['n']) if w not in mapped)
    for w in range(case['n']):
     if perm[w] is None:perm[w]=next(unused)
    remapped=QuantumCircuit(case['n']);remapped.compose(part,perm,inplace=True);part=remapped;cert=dict(cert,original_local_to_physical=original,local_to_physical=mapped,dirty_policy=dirty_policy,borrowed_mapping_relabel_only=True)
  q.compose(part,inplace=True);certs.append(cert)
 return q,dict(primitive='frozen_individual_structured_exact_mixed',dirty_policy=dirty_policy,scope='global_exact',individual_certificates=certs)

def candidates(case,group,beta=BETA):
 rows=[]
 for policy in ('external_first','canonical_any'):
  raw,cert=individual(case,group,beta,policy)
  if raw is not None:
   q=core.native(raw,1);rows.append(dict(name='individual_structured_exact_mixed_'+policy,dirty_policy_priority=0 if policy=='external_first' else 1,q=q,cost=core.costs(q),certificate=cert))
 if len(group['targets'])<2 or not group['controls']:return rows
 for source in A_SOURCES:
  for policy in ('external_first','canonical_any'):
   A,contract=embed_A(case,group,source,policy)
   if A is None:continue
   for orientation in (False,True):
    raw=group_raw(case,group,beta,A,orientation);native=core.native(raw,1);rows.append(dict(name=source+'_'+policy+('_inverse_first' if orientation else '_forward'),dirty_policy_priority=0 if policy=='external_first' else 1,q=native,cost=core.costs(native),certificate=dict(contract,inverse_first=orientation,actual_inverse_paired=True,fanout_anchor=group['targets'][0],fanout_edges=[[group['targets'][0],t] for t in group['targets'][1:]],algebra='P=F A F=Dprime product(X_T)^g; Pdag Rz_T P removes Dprime exactly',scope='global_exact')))
 return rows

def rank(item):return item['cost']['cx'],item['cost']['parameterized_rz'],item['cost']['depth'],item.get('dirty_policy_priority',0),item['name']

def program(case,method='multitarget_rphase_exact',strict_scalar=True):
 if method not in [x['name'] for x in METHODS]:raise ValueError(method)
 params={name:Parameter(name) for name in ('gamma','beta0','beta1')};groups=grouping(case);choices=[];events=[]
 for group in groups:
  pool=candidates(case,group);winner=min(pool,key=rank);choices.append(winner);events.append(dict(group,selected_name=winner['name'],selected_native1=winner['cost'],candidate_rows=[{k:v for k,v in x.items() if k!='q'} for x in pool],selection_rule='local native1 cx, parameterized_rz, depth, external-first priority, then name; individual frozen exact always retained'))
 q=QuantumCircuit(case['n'])
 for sweep in range(2):
  for selected in choices:q.compose(selected['q'].assign_parameters({BETA:params['beta'+str(sweep)]}),inplace=True)
  if sweep==0:q.compose(core.phase(case['n'],params['gamma']),inplace=True)
 meta=dict(method=method,side='static',params=params,target_order=core.target_order(case),groups=events,events=events,new_work=0,scope='global_exact',literal_order_preserved_by_certified_commuting_swaps=True,phase_proof='arbitrary diagonal relative-oracle phase cancels with actual inverse under diagonal Rz_T; no projective alignment',global_even_certificate=dict(full_Hilbert_exact_to_canonical_program=True,type='global_exact_false_twin_commuting_reorder_and_paired_fanout',not_dependent_on_even_sweep_phase_cancellation=True))
 if method.startswith('bridge_'):
  from mis_phase_native_bridge import bridge
  mode=method.rsplit('_',1)[1];q,cert=bridge(core.native(q,1),mode,'clifford_normalized');meta['bridge']=cert
 return q,meta
