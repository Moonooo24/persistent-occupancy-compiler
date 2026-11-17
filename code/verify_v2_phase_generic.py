from pathlib import Path
import sys,json,time,hashlib
sys.path.insert(0,'code')
from runtime import ROOT,core,io,G,np,Statevector,sha
import auto_reuse_v2 as A
import baselines_v2 as B
import matched_reference_v2 as M
from verify_auto import checks,required_parameters
D=ROOT/'results/v2_phase_generic_QA';D.mkdir(exist_ok=True);rows=[];start=time.monotonic()
cases=[dict(id='affine_p8_weighted',n=3,graph_edges=[[0,1],[0,2],[1,2]],sweeps=8,weights=[2,-3,1],target_order=[2,0,1]),dict(id='mixed_p4',n=6,graph_edges=[[0,1],[0,2],[1,2],[2,3],[2,4],[3,4]],sweeps=4,weights=['1/3','2/3',1,-2,5,0]),dict(id='rank9_p1',n=10,graph_edges=[[0,j] for j in range(1,10)],sweeps=1),dict(id='rank10_p2',n=11,graph_edges=[[0,j] for j in range(1,11)],sweeps=2)]
for c in cases:
 for mode in ('none','all'):
  raw,plan=A.build(c,cache=mode);one=core.native(raw.copy(),1);post=core.native(one.copy(),3);tests=[]
  for label,q in [('raw',raw),('native',one),('post',post)]:
   tests.append(dict(stage=label,metrics=core.costs(q),checks=checks(q,c,all_columns=c['n']<=6)))
  rows.append(dict(case=c['id'],mode=mode,tests=tests));print(json.dumps(dict(case=c['id'],mode=mode,status='passed',CX=tests[-1]['metrics']['cx'])),flush=True)
# Relevant fixed T3/strong exact constructions under the same finite backend.
c=cases[0]
for method in ('ref_t3_original','ref_t3_clique','ref_cpg_closed'):
 raw,cert=B.build(c,method);post=core.native(core.native(raw,1),3);t=checks(post,c,all_columns=True);rows.append(dict(case=c['id'],method=method,tests=[dict(stage='post',metrics=core.costs(post),checks=t)]))
c=cases[-1]
raw,cert=M.build(c,'ref_matched_exact_clique');post=core.native(core.native(raw,1),3);t=checks(post,c,all_columns=False);rows.append(dict(case=c['id'],method='generic_exact_pair',tests=[dict(stage='post',metrics=core.costs(post),checks=t)]))
result=dict(status='all12_v2_program_configurations_passed',rows=rows,seconds=time.monotonic()-start,source_sha256={x:sha(ROOT/'code'/x) for x in ('auto_reuse_v2.py','phase_safe.py','generic_query_v2.py','baselines_v2.py','matched_reference_v2.py')},max_complex_error=max(t['max_complex_error'] for r in rows for v in r['tests'] for t in v['checks']),no_phase_alignment=True)
(D/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
