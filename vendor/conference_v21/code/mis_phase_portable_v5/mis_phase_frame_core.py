"""CZ-star frame synthesis, fixed-route additive CX, budget-identity quotient.
Portable development core. No truth-table enumeration or legacy dependencies.
The raw-CX optimality applies only to fixed routes in this frame class.
"""
from copy import deepcopy
from fractions import Fraction
from qiskit import QuantumCircuit

def inverse_gray(value):
 result=0
 while value:result^=value;value>>=1
 return result

def routes_from_items(n,items):
 routes=[]
 for target,item in enumerate(items):
  order=item['order'];coeff={int(k):Fraction(v) for k,v in item['coefficients'].items()}
  masks=sorted(coeff,key=inverse_gray)
  physical=[sum(1<<w for j,w in enumerate(order) if u>>j&1) for u in masks]
  routes.append(dict(target=target,order=order,sequence=physical,coefficients=[coeff[u] for u in masks],name=item.get('name','')))
 return routes

def endpoint_cost(route,other,L,M):
 s=route['sequence']
 if not s:return L^M
 return ((s[0]>>other&1)^L)+((s[-1]>>other&1)^M)

def _strict_edge_optimize(n,routes,free_edges):
 before=[0]*len(routes);after=[0]*len(routes);start=[];end=[];edge_records=[];endpoint_total=0
 for i in range(n):
  for j in range(i+1,n):
   events=[k for k,r in enumerate(routes) if r['target'] in (i,j)];free=(i,j) in free_edges
   dp={s:(0,[],s) for s in ((0,1) if free else (0,))}
   for k in events:
    other=j if routes[k]['target']==i else i;new={}
    for L,(cost,path,initial) in dp.items():
     for M in (0,1):
      value=(cost+endpoint_cost(routes[k],other,L,M),path+[(k,L,M)],initial)
      if M not in new or value<new[M]:new[M]=value
    dp=new
   cost,path,initial=min(v for s,v in dp.items() if free or s==0)
   final=path[-1][2] if path else initial;endpoint_total+=cost
   if initial:start.append((i,j))
   if final:end.append((i,j))
   for k,L,M in path:
    other=j if routes[k]['target']==i else i
    before[k]|=L<<other;after[k]|=M<<other
   edge_records.append(dict(edge=[i,j],free=free,cost=cost,initial=initial,final=final,path=path))
 internal=sum(sum((a^b).bit_count() for a,b in zip(r['sequence'],r['sequence'][1:])) for r in routes)
 return dict(before=before,after=after,start_edges=start,end_edges=end,edge_records=edge_records,endpoint_cx=endpoint_total,internal_cx=internal,raw_mixer_cx=endpoint_total+internal)

def edge_optimize(n,routes,free_edges):
 # Solve all nonfree two-state edge chains with zero boundary states.
 plan=_strict_edge_optimize(n,routes,set());before=plan['before'];after=plan['after'];saved=0
 for rec in plan['edge_records']:
  i,j=rec['edge']
  if (i,j) not in free_edges:continue
  saved+=rec['cost'];rec['cost']=0;rec['free']=True;rec['independent_between_events']=True
  new=[]
  for k,_,_ in rec['path']:
   other=j if routes[k]['target']==i else i;s=routes[k]['sequence']
   L=(s[0]>>other&1) if s else 0;M=(s[-1]>>other&1) if s else 0
   before[k]=(before[k]&~(1<<other))|(L<<other);after[k]=(after[k]&~(1<<other))|(M<<other);new.append((k,L,M))
  rec['path']=new
 plan['endpoint_cx']-=saved;plan['raw_mixer_cx']-=saved
 plan['scope']='Fixed route raw CX in CZ-star frames modulo the safe budget-forbidden-edge subgroup'
 plan['budget_edges_can_reset_between_events']=True
 return plan

def best_cycle(route,L=0,M=0,E=0):
 seq=route['sequence'];co=route['coefficients']
 if not seq:return deepcopy(route)
 assert len(set(seq))==len(seq),'Each exact Walsh mask must occur once'
 cycle=sum((a^b).bit_count() for a,b in zip(seq,seq[1:]+seq[:1]))
 # With unique masks, first mask separates cuts and the second separates
 # the two orientations. This is exactly lex(sequence), in O(k) time.
 options=[]
 for direction in (1,-1):
  for cut in range(len(seq)):
   first=seq[cut];last=seq[(cut-direction)%len(seq)]
   score=cycle-(first^last).bit_count()+((first^L)&~E).bit_count()+((last^M)&~E).bit_count()
   second=seq[(cut+direction)%len(seq)]
   options.append((score,first,second,direction,cut))
 _,_,_,direction,cut=min(options);ids=[(cut+direction*k)%len(seq) for k in range(len(seq))]
 out=deepcopy(route);out['sequence']=[seq[k] for k in ids];out['coefficients']=[co[k] for k in ids];return out


def masks_for_free(n,free):
 return [sum(1<<(j if i==t else i) for i,j in free if t in (i,j)) for t in range(n)]

def plan_routes(n,routes,free_edges,mode,iterations=8):
 Es=masks_for_free(n,free_edges)
 routes=[best_cycle(r,E=Es[r['target']] if mode!='closed' else 0) for r in deepcopy(routes)]
 def local_plan():
  before=[];after=[]
  for r in routes:
   E=Es[r['target']] if mode!='closed' else 0;s=r['sequence'];before.append(s[0]&E if s else 0);after.append(s[-1]&E if s else 0)
  total=sum(sum((a^b).bit_count() for a,b in zip(r['sequence'],r['sequence'][1:]))+((r['sequence'][0]^L).bit_count()+(r['sequence'][-1]^M).bit_count() if r['sequence'] else (L^M).bit_count()) for r,L,M in zip(routes,before,after))
  return dict(before=before,after=after,raw_mixer_cx=total)
 if mode in ('closed','local_open'):return routes,local_plan()
 history=[]
 for iteration in range(iterations if mode=='frame_coordinate' else 1):
  plan=edge_optimize(n,routes,free_edges);history.append(plan['raw_mixer_cx']);changed=False
  if mode!='frame_coordinate':break
  for k,r in enumerate(routes):
   E=Es[r['target']];L=plan['before'][k];M=plan['after'][k]
   candidate=best_cycle(r,L,M,E)
   def cost(route):
    s=route['sequence']
    return sum((a^b).bit_count() for a,b in zip(s,s[1:]))+(((s[0]^L)&~E).bit_count()+((s[-1]^M)&~E).bit_count() if s else (L^M).bit_count())
   if cost(candidate)<cost(r):routes[k]=candidate;changed=True
  if not changed:break
 final=edge_optimize(n,routes,free_edges);history.append(final['raw_mixer_cx'])
 assert all(a>=b for a,b in zip(history,history[1:])),history
 final['coordinate_descent_raw_cx']=history;return routes,final

def raw_module(n,route,beta,L,M):
 target=route['target'];assert not((L|M)>>target&1)
 qc=QuantumCircuit(n);qc.h(target);current=0
 for mask,c in zip(route['sequence'],route['coefficients']):
  wanted=mask^L
  for w in range(n):
   if (current^wanted)>>w&1:qc.cx(w,target)
  qc.rz(2*beta*float(c),target);current=wanted
 # Desired net parity L xor M; outer H maps it to the CZ frame change.
 for w in range(n):
  if (current^L^M)>>w&1:qc.cx(w,target)
 qc.h(target);return qc
