from pathlib import Path
import sys,json,time
R=Path.cwd();sys.path.insert(0,str(R/'code'));from runtime import QC,Parameter,core,G,io,sha,np
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RCCXGate
import auto_reuse_v2 as A
D=R/'results/explicit_kernel_recipe_QA_v2';D.mkdir(exist_ok=False);b=Parameter('recipe_beta')
def tau(q,a,c,t):
 q.h(t);q.t(t);q.cx(c,t);q.tdg(t);q.cx(a,t);q.t(t);q.cx(c,t);q.tdg(t);q.h(t)
def child(n,t,l1,l2,delta):
 q=QC(n);q.h(t);q.rz(delta/2,t);q.cx(l1,t);q.rz(delta/2,t);q.cx(l2,t);q.rz(delta/2,t);q.cx(l1,t);q.rz(delta/2,t);q.h(t);return q
prefix={4:[],5:[('tau',[3,4,1])],6:[('tau',[3,4,1]),('tau',[1,5,2])],7:[('tau',[3,4,1]),('tau',[5,6,4]),('x',[4]),('tau',[7,4,3]),('x',[3])],8:[('tau',[3,4,1]),('tau',[5,6,4]),('x',[4]),('tau',[7,8,6]),('x',[6]),('tau',[6,4,3]),('x',[3])]}
middle={4:[3,4],5:[1,5],6:[2,6],7:[1,3],8:[1,3]}
def recipe(r):
 V=QC(r+1)
 for name,w in prefix[r]:
  if name=='tau':tau(V,*w)
  else:V.x(w[0])
 aa=QC(r+1)
 for j in range(3,r+1):aa.x(j)
 aa.compose(V,inplace=True);aa.ccx(*middle[r],0);aa.compose(V.inverse(),inplace=True)
 for j in range(3,r+1):aa.x(j)
 q=QC(r+1);q.z(0);q.compose(child(r+1,0,1,2,-b/2),inplace=True);q.h(0);q.compose(aa,inplace=True);q.h(0);q.compose(child(r+1,0,1,2,b/2),inplace=True);return q
x=QC(3);tau(x,0,1,2);tauerr=float(np.max(abs(Operator(x).data-Operator(RCCXGate()).data)));assert tauerr<1e-12;rows=[]
# Explicit B2 word versus actual frozen degree2 child, including global scalar.
b2,_=A.compact_template('conditional_phase_mixed',2)
for angle in [0.,.371,-1.207]:
 c=child(3,0,1,2,angle);m=Operator(c).data;want=Operator(b2.assign_parameters({p:angle for p in b2.parameters})).data;err=float(np.max(abs(m-want)));assert err<1e-12;rows.append(dict(kind='B2_full_space',angle=angle,columns=8,max_error=err))
for r in range(4,9):
 raw=recipe(r);q=core.native(raw,1);template,cert=A.compact_template('conditional_phase_mixed',r);assert template.num_qubits==r+1 and core.costs(q)['cx']<=6*r-12
 for angle in [0.,.371,-1.207]:
  m=Operator(q.assign_parameters({b:angle})).data;want=Operator(template.assign_parameters({p:angle for p in template.parameters})).data;err=float(np.max(abs(m-want)));assert err<2e-12,(r,angle,err);rows.append(dict(kind='production_full_space_recipe_match',rank=r,angle=angle,columns=2**(r+1),max_error=err,artifact_SHA=cert['source_sha256'],CX_recipe=core.costs(q)['cx'],CX_frozen=core.costs(template)['cx']))
 print('Verified_all_columns_rank',r,flush=True)
out=dict(status='explicit_tau_B2_and_all5production_recipes_match_frozen_artifacts',tau_error=tauerr,rows=rows,max_error=max(x['max_error'] for x in rows),phase_alignment=False,analytic_definition_written_separately=True,numeric_tests_not_general_parameter_proof=True,prefix=prefix,middle=middle,source_lock=sha(R/'protocols/source_lock_v2r1.json'))
(D/'summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=out['status'],max_error=out['max_error'])),flush=True)
