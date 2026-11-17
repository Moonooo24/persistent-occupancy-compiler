"""One immutable compiled program per input/method, with provenance and IO checks."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import time
import traceback

from runtime import ROOT, G, QC, np, Statevector, core, io, sha, psutil
import auto_reuse_v2 as A
import baselines_v2 as B
from verify_auto import checks, required_parameters

OURS = ('ours_none','ours_all','ours_selected','ours_walsh_all')
MATCHED=('ref_matched_exact_original','ref_matched_exact_clique')
METHODS = OURS + B.REFERENCES + MATCHED


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(str(path))
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def construct(case,method):
    if method in MATCHED:
        import matched_reference_v2 as M
        return M.build(case,method)
    if method in B.REFERENCES:return B.build(case,method)
    if method=='ours_none':return A.build(case,cache='none')
    if method=='ours_all':return A.build(case,cache='all')
    if method=='ours_walsh_all':return A.build(case,cache='all',route='walsh')
    if method=='ours_selected':
        selected,decision=A.select_cache(case)
        q,plan=A.build(case,cache=selected);plan['cache_decision']=decision
        return q,plan
    raise ValueError(method)


def run(casefile,method,outroot='results/main_matrix_v1'):
    casefile=Path(casefile);casefile=casefile if casefile.is_absolute() else ROOT/casefile
    case=json.loads(casefile.read_text(encoding='utf-8-sig'));folder=ROOT/outroot/case['id']/method;rp=folder/'result.json'
    lockpath=ROOT/'protocols/artifact_source_lock.json'
    lock=json.loads(lockpath.read_text())
    for item in lock['code_files']:
        assert sha(ROOT/item['file'])==item['sha256'],item['file']
    if rp.exists():
        previous=json.loads(rp.read_text());assert previous['status']=='passed' and previous['source_lock_sha256']==sha(lockpath)
        for art in previous['rows']:assert sha(ROOT/art['file'])==art['sha256']
        print(json.dumps(dict(case=case['id'],method=method,status='verified_existing')),flush=True);return previous
    folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();proc=psutil.Process();t=time.perf_counter();raw,plan=construct(case,method);tb=time.perf_counter()-t
    put(folder/'construction.json',plan);print(json.dumps(dict(case=case['id'],method=method,stage='raw_built',CX=core.costs(raw)['cx'],seconds=tb)),flush=True)
    t=time.perf_counter();one=core.native(raw.copy(),1);t1=time.perf_counter()-t
    t=time.perf_counter();three=core.native(one.copy(),3);t3=time.perf_counter()-t
    rows=[]
    for label,q in [('raw',raw),('native',one),('post',three)]:
        assert {p.name for p in q.parameters}==required_parameters(case),'Lost or extra original parameter'
        qp=folder/(label+'.qpy');fp=G.fingerprint(q)
        with qp.open('wb') as stream:io.dump(q,stream)
        with qp.open('rb') as stream:loaded=io.load(stream)[0]
        assert fp==G.fingerprint(loaded) and loaded.num_qubits==case['n']
        tests=checks(loaded,case,all_columns=case['n']<=8) if case['n']<=16 else []
        rows.append(dict(stage=label,file=str(qp.relative_to(ROOT)).replace('\\','/'),sha256=sha(qp),metrics=core.costs(loaded),strict_IO_fingerprint=fp,checks=tests))
        gc.collect()
    result=dict(status='passed',case_id=case['id'],split=case.get('split'),method=method,n=case['n'],sweeps=case.get('sweeps',2),input_sha256=sha(casefile),
                source_lock_sha256=sha(lockpath),construction_sha256=sha(folder/'construction.json'),rows=rows,
                build_seconds=tb,native1_seconds=t1,whole3_seconds=t3,total_seconds=time.monotonic()-start,
                RSS_bytes=proc.memory_info().rss,logical_all_to_all=True,new_work_qubits=0,backend_policy='phase_safe_bounded_v2',stage_meaning='native=one finite local round; post=three finite local rounds, not SDK optimization-level3',
                numerical_scope='all Feas columns+complex on n<=8; two complex Feas inputs/four angles on n<=16; structural contracts+compiled provenance otherwise',
                numeric_statevector_performed=case['n']<=16,no_phase_alignment=True,
                native_policy=core.POLICY,seed_transpiler=core.SEED,author_configuration_search=False)
    put(rp,result);print(json.dumps(dict(case=case['id'],method=method,status='passed',whole3=rows[-1]['metrics'],seconds=result['total_seconds'])),flush=True);return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',required=True);a.add_argument('--method',choices=METHODS,required=True);a.add_argument('--out',default='results/main_matrix_v1');x=a.parse_args()
    try:run(x.case,x.method,x.out)
    except Exception as e:
        cpath=Path(x.case);cpath=cpath if cpath.is_absolute() else ROOT/cpath
        case=json.loads(cpath.read_text(encoding='utf-8-sig'));p=ROOT/x.out/case['id']/x.method/'failure.json'
        if not p.exists():put(p,dict(status='failed',exception_type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),source_sha256=sha(ROOT/'code/experiments_v2.py')))
        raise
