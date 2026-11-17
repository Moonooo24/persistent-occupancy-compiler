"""Generic clique-partition parity bridge using unchanged local partial kernels."""
from pathlib import Path
import os,sys,json,hashlib,math,time,gc,argparse,struct
ROOT=Path(__file__).resolve().parents[2];sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[k]='1'
os.environ.update(QISKIT_PARALLEL='FALSE',QISKIT_IN_PARALLEL='TRUE',QISKIT_NUM_PROCS='1',CUDA_VISIBLE_DEVICES='')
BUNDLE=ROOT/'code/mis_phase_portable_v5';sys.path[:0]=[str(ROOT.parent/'research/.venv_qiskit/Lib/site-packages'),str(BUNDLE)]
import numpy as np,psutil
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
import mis_phase_core as core
import lossless_qpy_io as io
PROTO=ROOT/'protocols/ours_generic_clique_partition_development_v1.json';OUT=ROOT/'results/ours_generic_clique_partition_development_v1';BETA=Parameter('bridge_beta');ANGLES=((0.,0.,0.),(.37,.31,-.51),(3.7,-4.13,6.29),(-.417,1.117,-2.31));CACHE={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(json.dumps(x,indent=2).encode()+b'\n')
def memory():
 v=psutil.Process().memory_info().rss
 if v>256*2**20:raise MemoryError('256MiB RSS cap')
 return v

def kernel_template(d):
 if d in CACHE:return CACHE[d]
 manifest=BUNDLE/'mis_phase_templates_manifest_v5.json';item=next(x for x in json.loads(manifest.read_text(encoding='utf8'))['templates'] if x['degree']==d and x['family']=='conditional_phase_mixed');path=BUNDLE/item['file'];assert sha(path)==item['sha256']
 with path.open('rb') as f:[original]=io.load(f)
 used=sorted({original.find_bit(w).index for row in original.data for w in row.qubits});required=list(range(d+1));extras=sorted(set(used)-set(required));active=required+extras;idle=sorted(set(range(original.num_qubits))-set(active));assert all(j not in used for j in idle)
 compact=QuantumCircuit(len(active));compact.global_phase=original.global_phase
 for row in original.data:
  assert not row.clbits and row.operation.name in core.BASIS;compact.append(row.operation.copy(),[active.index(original.find_bit(w).index) for w in row.qubits])
 assert int(compact.count_ops().get('cx',0))==item['cost']['cx'];assert set(extras)<=set(item.get('borrowed_local_wires',[]))
 cert=dict(family='conditional_phase_mixed',degree=d,template_file=item['file'],template_sha256=item['sha256'],manifest_sha256=sha(manifest),original_template_width=original.num_qubits,compact_width=compact.num_qubits,active_template_indices=active,active_extra_template_indices=extras,removed_padding=idle,padding_proved_no_instruction_touch=True,global_phase_original_object_copied=True,local_care_contract='target0 arbitrary controls; target1 only controls0; coherent exact RX(2beta), non-target/dirty restored',arbitrary_dirty_initialization_allowed=True,initial_zero_qubits_assumed=False,CX=int(compact.count_ops().get('cx',0)),phase_table=item['phase_table'],source_verification=item['verification']['type'])
 CACHE[d]=(compact,cert);return CACHE[d]

def local_kernel(n,target,controls,beta):
 assert target not in controls and len(set(controls))==len(controls)
 if not controls:
  q=QuantumCircuit(n);q.rx(2*beta,target);return q,dict(degree=0,CX=0,source='uncontrolled RX(2beta)',physical_mapping=[target],borrowed_original_data=[],initial_zero_qubits_assumed=False)
 template,cert=kernel_template(len(controls));available=[j for j in range(n) if j!=target and j not in controls];extra=template.num_qubits-len(controls)-1;assert extra<=len(available);mapping=[target]+list(controls)+available[:extra];q=QuantumCircuit(n);q.compose(template.assign_parameters({p:beta for p in template.parameters}),mapping,inplace=True)
 return q,dict(cert,physical_mapping=mapping,borrowed_original_data=available[:extra],non_target_and_dirty_restoration_required=True)

def check_partition(case):
 n=case['n'];parts=case['partition'];edges={tuple(sorted(e)) for e in case['graph_edges']};assert sorted(v for C in parts for v in C)==list(range(n)) and all(len(C)>0 and C==sorted(C) for C in parts)
 assert all((i,j) in edges for C in parts for z,i in enumerate(C) for j in C[z+1:]);return [min(C) for C in parts]

def F(case):
 anchors=check_partition(case);q=QuantumCircuit(case['n'])
 for C,a in zip(case['partition'],anchors):
  for j in C:
   if j!=a:q.cx(j,a)
 return q

def cx_pairs(q):return [[q.find_bit(x.qubits[0]).index,q.find_bit(x.qubits[1]).index] for x in q.data if x.operation.name=='cx']

def macro(case,i,beta=BETA):
 parts=case['partition'];anchors=check_partition(case);ci=next(z for z,C in enumerate(parts) if i in C);C=parts[ci];a=anchors[ci];N=core.neighbors(case)[i];external=QuantumCircuit(case['n']);pred=[];pivots=[]
 for z,D in enumerate(parts):
  if z==ci:continue
  S=sorted(set(N)&set(D))
  if not S:continue
  support=([anchors[z]]+[j for j in D if j not in S]) if anchors[z] in S else list(S);assert support and len(set(support))==len(support) and all(j in D for j in support);support=sorted(support);pivot=min(support);pivots.append(pivot);pairs=[]
  for j in support:
   if j!=pivot:external.cx(j,pivot);pairs.append([j,pivot])
  pred.append(dict(clique_index=z,clique=D,intersection=S,parity_support=support,pivot=pivot,compute_CX=pairs,uncompute_CX=list(reversed(pairs)),semantic='parity equals original OR(intersection) on encoded code; no zero pivot assumption'))
 own=QuantumCircuit(case['n']);owncontrol=None;kind='singleton'
 if len(C)>1:
  if i!=a:kind='leaf';own.cx(i,a);owncontrol=a
  else:
   kind='anchor';leaves=[j for j in C if j!=a];w=min(leaves)
   for j in leaves:
    if j!=w:own.cx(j,w)
   own.cx(w,a);owncontrol=w
 controls=([owncontrol] if owncontrol is not None else [])+pivots;K,kcert=local_kernel(case['n'],i,controls,beta);q=external.compose(own).compose(K).compose(own.inverse()).compose(external.inverse());extcost=len(cx_pairs(external));owncost=len(cx_pairs(own));upper=2*(extcost+owncost)+kcert['CX'];assert int(q.count_ops().get('cx',0))==upper
 cert=dict(target=i,clique_index=ci,clique=C,anchor=a,kind=kind,original_neighbors=N,external_predicates=pred,external_compute_CX=cx_pairs(external),external_actual_inverse_CX=list(reversed(cx_pairs(external))),own_compute_CX=cx_pairs(own),own_actual_inverse_CX=list(reversed(cx_pairs(own))),controls=controls,kernel=kcert,CX_upper=upper,CX_parts=dict(external_compute_uncompute=2*extcost,own_compute_uncompute=2*owncost,old_kernel=kcert['CX']),local_care_only=True,intermediate_global_Feas_not_assumed=True,decoded_non_target_data_restored=True,external_pivots_restored=True,own_leaf_parity_restored=True,encoded_anchor_updates_with_leaf_flip=i!=a)
 return q,cert

def permute(bits,pairs):
 out=bits
 for c,t in pairs:out^=((out>>c)&1)<<t
 return out

def feas_ids(case):
 ids=np.arange(1<<case['n']);good=np.ones(len(ids),bool)
 for i,j in case['graph_edges']:good&=~(((ids>>i)&1).astype(bool)&((ids>>j)&1).astype(bool))
 return ids[good].tolist(),good

def truth_certificate(case,certs):
 ids,_=feas_ids(case);fp=cx_pairs(F(case));records=[]
 for c in certs:
  i=c['target'];prefix=c['external_compute_CX']+c['own_compute_CX'];inverse=c['own_actual_inverse_CX']+c['external_actual_inverse_CX'];active=0;target1=0
  for s in ids:
   enc=permute(s,fp);temp=permute(enc,prefix);guard=not any((temp>>j)&1 for j in c['controls']);original_guard=not any((s>>j)&1 for j in c['original_neighbors']);assert guard==original_guard and ((temp>>i)&1)==((s>>i)&1);assert permute(temp,inverse)==enc
   if (temp>>i)&1:assert guard;target1+=1
   for pred in c['external_predicates']:
    val=(temp>>pred['pivot'])&1;assert val==int(any((s>>j)&1 for j in pred['intersection']))
   if guard:assert permute(temp^(1<<i),inverse)==permute(s^(1<<i),fp);active+=1
  records.append(dict(target=i,Feas_basis_states=len(ids),no_flip_full_prefix_restoration_branches=len(ids),active_flip_recode_branches=active,temporary_target1_implies_all_controls_zero=target1,all_original_decoded_non_target_bits_preserved=True))
 return dict(status='passed',records=records,scope='actual planned CX permutations and code care, not intermediate global feasibility')

def program(case):
 q=F(case);params={x:Parameter(x) for x in ('gamma','beta0','beta1')};certs=[];macros=[]
 for i in case['target_order']:
  m,c=macro(case,i);certs.append(c);macros.append(m)
 truth=truth_certificate(case,certs)
 for sweep in range(2):
  for m in macros:q.compose(m.assign_parameters({BETA:params['beta'+str(sweep)]}),inplace=True)
  if sweep==0:
   for a in check_partition(case):q.p(params['gamma'],a)
 q.compose(F(case).inverse(),inplace=True);bound=2*len(cx_pairs(F(case)))+2*sum(c['CX_upper'] for c in certs);assert int(q.count_ops().get('cx',0))==bound
 return q,dict(partition=case['partition'],anchors=check_partition(case),F_CX=cx_pairs(F(case)),F_inverse_CX=list(reversed(cx_pairs(F(case)))),macro_certificates=certs,truth_certificate=truth,constructive_CX_upper=bound,fully_decomposed_raw_CX=bound,encoded_cost='-sum clique occupancy anchors; P(gamma) on each retains #cliques*gamma/2 in Rz form',same_generic_macro_recipe_both_sweeps=True,literal_order_preserved=True,no_new_qubits=True,scope='Feas/code only; not_full_Hilbert_even')

def reference(v,case,angles):
 out=v.copy();ids=np.arange(1<<case['n']);N=core.neighbors(case);pairs=[]
 for i in case['target_order']:
  mask=sum(1<<j for j in N[i]);lo=ids[((ids&mask)==0)&((ids&(1<<i))==0)];pairs.append((lo,lo|(1<<i)))
 for sweep,b in enumerate(angles[1:]):
  for lo,hi in pairs:
   x=out[lo].copy();y=out[hi].copy();out[lo]=math.cos(b)*x-1j*math.sin(b)*y;out[hi]=math.cos(b)*y-1j*math.sin(b)*x
  if sweep==0:
   phase=np.exp(1j*angles[0]*np.asarray([s.bit_count() for s in range(1<<case['n'])]));out*=phase if out.ndim==1 else phase[:,None]
 return out

def care_columns(q,case,angles):
 n=case['n'];ids,good=feas_ids(case);maximum=leak=0.;batch=16
 for start in range(0,len(ids),batch):
  memory();sel=ids[start:start+batch];v=np.zeros((1<<n,batch),complex);v[sel,np.arange(len(sel))]=1;want=reference(v,case,angles);vec=v.T.reshape(-1)/math.sqrt(len(sel));got=Statevector(vec.copy()).evolve(q,qargs=list(range(n))).data.reshape(batch,1<<n).T*math.sqrt(len(sel));maximum=max(maximum,float(np.max(abs(got-want))));leak=max(leak,float(np.max(abs(got[~good]))) if (~good).any() else 0.);assert maximum<2e-9
 return dict(Feas_columns=len(ids),stream_batch=16,max_complex_error=maximum,max_outside_Feas_leakage=leak,algorithm_qubits=n,reference_qubits_classical_batch_only=4,no_phase_alignment=True)

def superpositions(q,case,angles,seed):
 ids,_=feas_ids(case);rng=np.random.default_rng(seed);rows=[]
 for k in range(2):
  v=np.zeros(1<<case['n'],complex);v[ids]=rng.normal(size=len(ids))+1j*rng.normal(size=len(ids));v/=np.linalg.norm(v);h=hashlib.sha256(v.tobytes()).hexdigest();want=reference(v,case,angles);got=Statevector(v.copy()).evolve(q).data;err=float(np.max(abs(got-want)));assert err<2e-9 and hashlib.sha256(v.tobytes()).hexdigest()==h;rows.append(dict(index=k,max_complex_error=err,input_unchanged=True,no_phase_alignment=True))
 return rows

def fingerprint(q):
 def tok(v):
  ps=getattr(v,'parameters',set())
  if not ps:
   try:return ['f64',struct.pack('!d',float(v)).hex()]
   except (TypeError,ValueError):pass
  return ['expr',str(v),sorted((p.name,str(p.uuid)) for p in ps)]
 return hashlib.sha256(json.dumps(dict(n=q.num_qubits,phase=tok(q.global_phase),ops=[[x.operation.name,[q.find_bit(w).index for w in x.qubits],[tok(v) for v in x.operation.params]] for x in q.data]),sort_keys=True).encode()).hexdigest()

def local_checks():
 rows=[]
 for d in (1,2,3):
  template,c=kernel_template(d);n=template.num_qubits;ids=[s for s in range(1<<n) if not(s&1) or (s&((1<<(d+1))-2))==0];assert not c['active_extra_template_indices'];case=dict(n=n,graph_edges=[[0,j] for j in range(1,d+1)],target_order=[0]);one=core.native(template.copy(),1)
  for stage,q in [('raw',template),('native1',one),('whole3',core.native(one.copy(),3))]:
   tests=[]
   for b in (0.,.31,-4.13,6.29):
    bound=q.assign_parameters({p:b for p in q.parameters});v=np.zeros((1<<n,16),complex);v[ids,np.arange(len(ids))]=1;want=v.copy();x=want[0].copy();y=want[1].copy();want[0]=math.cos(b)*x-1j*math.sin(b)*y;want[1]=math.cos(b)*y-1j*math.sin(b)*x;vec=v.T.reshape(-1)/math.sqrt(len(ids));got=Statevector(vec.copy()).evolve(bound,qargs=list(range(n))).data.reshape(16,1<<n).T*math.sqrt(len(ids));err=float(np.max(abs(got-want)));assert err<2e-9;tests.append(dict(beta=b,care_columns=len(ids),max_complex_error=err))
   rows.append(dict(degree=d,stage=stage,metrics=core.costs(q),template_certificate=c,checks=tests))
 put(OUT/'local_kernel_care_checks.json',dict(status='passed',rows=rows,local_target1_controls0_or_target0_contract=True,initial_zero_assumptions=False,no_dense_Operator=True))

def run_case(cid):
 p=json.loads(PROTO.read_text(encoding='utf8'));assert sha(__file__)==p['source_sha256'] and core.SEED==20261101 and core.POLICY=='native_phase_fsum_x_v2';case=next(c for c in p['cases'] if c['id']==cid);folder=OUT/cid;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'result.json').exists();start=time.monotonic();raw,plan=program(case);ids,_=feas_ids(case);assert len(ids)==case['expected_Feas'];one=core.native(raw.copy(),1);stages=[('raw',raw),('native1',one),('whole3',core.native(one.copy(),3))];rows=[]
 for stage,q in stages:
  path=folder/(stage+'.qpy');fp=fingerprint(q)
  with path.open('wb') as f:io.dump(q,f)
  with path.open('rb') as f:[loaded]=io.load(f)
  assert fingerprint(loaded)==fp==fingerprint(q) and loaded.num_qubits==case['n'] and sorted(p.name for p in loaded.parameters)==['beta0','beta1','gamma'];tests=[]
  for k,a in enumerate(ANGLES):
   bound=core.bind(loaded,a);tests.append(dict(angles=a,**care_columns(bound,case,a),superpositions=superpositions(bound,case,a,20261111+case['n']*31+k)))
  s=(1<<case['partition'][0][0])|(1<<case['partition'][0][1]);v=np.zeros(1<<case['n'],complex);v[s]=1;a=(.37,0.,0.);got=Statevector(v.copy()).evolve(core.bind(loaded,a)).data;err=float(np.max(abs(got-reference(v,case,a))));assert err>1e-5
  rows.append(dict(stage=stage,metrics=core.costs(loaded),QPY=str(path.relative_to(ROOT)).replace('\\','/'),QPY_sha256=sha(path),IO3_strict_fingerprint=fp,checks=tests,offcode_counterexample=dict(input_basis=s,angles=a,max_complex_error=err,expected_failure_outside_Feas=True,not_full_Hilbert_even=True)));gc.collect()
 assert rows[1]['metrics']['cx']<=plan['constructive_CX_upper'];result=dict(status='passed',case=case,Feas_count=len(ids),protocol_sha256=sha(PROTO),source_sha256=sha(__file__),independent_derivation_sha256=p['independent_derivation_sha256'],plan=plan,rows=rows,new_work_qubits=0,physical_qubits=case['n'],scope='Feas/code only',not_full_Hilbert_even=True,all_external_and_own_parities_uncomputed=True,no_intermediate_global_Feas_assumption=True,baseline_calls=0,old45_and_isolated_clique_v1_unchanged=True,seconds=time.monotonic()-start,RSS_bytes=memory());put(folder/'result.json',result);print(json.dumps(dict(case=cid,Feas=len(ids),upper=plan['constructive_CX_upper'],native1=rows[1]['metrics'],whole3=rows[2]['metrics'],seconds=result['seconds'],RSS_bytes=result['RSS_bytes'])),flush=True)

def summary():
 p=json.loads(PROTO.read_text(encoding='utf8'));rows=[];maximum=0.
 def errors(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k=='max_complex_error' and x.get('expected_failure_outside_Feas') is not True:yield v
    elif k!='offcode_counterexample':yield from errors(v)
  elif isinstance(x,list):
   for v in x:yield from errors(v)
 for c in p['cases']:
  path=OUT/c['id']/'result.json';r=json.loads(path.read_text(encoding='utf8'));assert r['status']=='passed';maximum=max(maximum,max(errors(r)));rows.append(dict(case_id=c['id'],n=c['n'],order=c['target_order'],Feas_count=r['Feas_count'],upper=r['plan']['constructive_CX_upper'],raw=r['rows'][0]['metrics'],native1=r['rows'][1]['metrics'],whole3=r['rows'][2]['metrics'],offcode_error=r['rows'][2]['offcode_counterexample']['max_complex_error'],QPY=[dict(stage=x['stage'],file=x['QPY'],sha256=x['QPY_sha256']) for x in r['rows']],result=str(path.relative_to(ROOT)).replace('\\','/'),result_sha256=sha(path)))
 put(OUT/'metrics_index.json',dict(status='passed',rows=rows,maximum_strict_Feas_complex_error=maximum,scope='generic non-module clique-partition Feas/code bridge acceptance only',not_full_Hilbert_even=True,no_dense_Operator=True,baseline_calls=0,old45_and_isolated_clique_v1_unchanged=True));print('GENERIC_PARTITION_ALL_PASS',maximum,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['local','case','summary']);p.add_argument('--case-id');a=p.parse_args();{'local':local_checks,'case':lambda:run_case(a.case_id),'summary':summary}[a.action]()
