from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,time,threading,platform
from runtime import ROOT,core,G,io,sha,psutil
import auto_reuse_v2 as A
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--mode',choices=['all','selected'],required=True);p.add_argument('--trial',type=int,required=True);x=p.parse_args();case=json.loads((ROOT/x.case).read_text());D=ROOT/'results/scaling_v2'/case['id']/x.mode/f'trial{x.trial}';D.mkdir(parents=True,exist_ok=False);proc=psutil.Process();peak=[proc.memory_info().rss];done=threading.Event()
def monitor():
 while not done.wait(.03):peak[0]=max(peak[0],proc.memory_info().rss)
threading.Thread(target=monitor,daemon=True).start();t=time.perf_counter();spec=A.graph_spec(case);discover=time.perf_counter()-t;dec=None;t=time.perf_counter()
if x.mode=='selected':cache,dec=A.select_cache(case)
else:cache='all'
select=time.perf_counter()-t;t=time.perf_counter();raw,cert=A.build(case,cache=cache);build=time.perf_counter()-t;t=time.perf_counter();one=core.native(raw.copy(),1);t1=time.perf_counter()-t;t=time.perf_counter();three=core.native(one.copy(),3);t3=time.perf_counter()-t;art=[]
if x.trial==0:
 for label,q in [('raw',raw),('native1',one),('whole3',three)]:
  f=D/(label+'.qpy');fp=G.fingerprint(q)
  with f.open('wb') as stream:io.dump(q,stream)
  with f.open('rb') as stream:v=io.load(stream)[0]
  assert fp==G.fingerprint(v);art.append(dict(stage=label,file=str(f.relative_to(ROOT)),sha256=sha(f),metrics=core.costs(v)))
 (D/'construction.json').write_text(json.dumps(cert,indent=2)+'\n')
done.set();peak[0]=max(peak[0],proc.memory_info().rss)
r=dict(status='passed',case_id=case['id'],n=case['n'],mode=x.mode,trial=x.trial,discovery_seconds=discover,selection_seconds=select,construction_seconds=build,native1_seconds=t1,whole3_seconds=t3,peak_RSS_bytes=peak[0],metrics=core.costs(three),artifacts=art,cache_decision=dec,source_sha256=sha(ROOT/'code/auto_reuse_v2.py'),input_sha256=sha(ROOT/x.case),statevector_calls=0,enumerated_states=0,highrank_Walsh_terms=0,python=platform.python_version(),clock='perf_counter',single_thread=True,import_time_excluded=True)
(D/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(n=r['n'],mode=x.mode,trial=x.trial,build=build,select=select,native1=t1,whole3=t3,RSS_MiB=peak[0]/2**20)),flush=True)
