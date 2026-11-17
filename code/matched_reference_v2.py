"""Frozen prior-project exact query library, applied to the same CPG factors.
This is a matched mechanism control, not author-algorithm HPO.
"""
from pathlib import Path
from numbers import Integral
import argparse,json,hashlib,time,traceback
from runtime import ROOT,QC,Parameter,core,io,G,sha
import auto_reuse_v2 as A
import experiments_v2 as E

FAMILY='structured_exact_mixed'
METHODS=('ref_matched_exact_original','ref_matched_exact_clique')


def kernel(n,i,controls,b):
    r=len(controls)
    if r==0:return A.exact_kernel(n,i,controls,b)
    if r<=8:
        template,cert=A.compact_template(FAMILY,r);available=[j for j in range(n) if j!=i and j not in controls];extra=template.num_qubits-r-1
        if extra<=len(available):
            mapping=[i]+controls+available[:extra];q=QC(n);q.compose(template.assign_parameters({p:b for p in template.parameters}),mapping,inplace=True)
            assert cert['scope']=='global_exact_local_macro'
            return q,dict(cert,physical_mapping=mapping,borrowed_original_data=available[:extra],full_domain_exact=True)
    from generic_query_v2 import kernel as generic_kernel
    if r>8:return generic_kernel(n,i,controls,b,exact=True)
    q,c=A.exact_kernel(n,i,controls,b);c['fallback_reason']='Frozen matched width unavailable';return q,c


def build(case,method):
    assert method in METHODS;spec=A.graph_spec(case);n=spec['n'];p=case.get('sweeps',2)
    assert isinstance(p,Integral) and not isinstance(p,bool) and p>=1
    orders=case.get('sweep_orders',[spec['order']]*p);assert len(orders)==p and all(sorted(o)==list(range(n)) for o in orders)
    b=Parameter('matched_local_beta');local={};certs=[]
    for i in range(n):
        blocks=A.supports(spec,i);factors=[A.mask_of(C) for C in blocks];pp,pivots=A.parity_network(n,factors);q=QC(n)
        for a,t in pp:q.cx(a,t)
        k,c=kernel(n,i,pivots,b);q.compose(k,inplace=True)
        for a,t in reversed(pp):q.cx(a,t)
        local[i]=q;certs.append(dict(target=i,blocks=blocks,pivots=pivots,parity_CX=pp,actual_inverse=list(reversed(pp)),primitive=c))
    beta=[Parameter('beta'+str(k)) for k in range(p)];gamma=[Parameter('gamma'+str(k)) for k in range(p-1)];out=QC(n);completion='original' if method.endswith('original') else 'clique'
    for k in range(p):
        for i in orders[k]:out.compose(local[i].assign_parameters({b:beta[k]}),inplace=True)
        if k<p-1:out.compose(A.weighted_cost(spec,set(),gamma[k],completion)[0],inplace=True)
    return out,dict(reference='Prior frozen structured exact query library on identical automatic CPG factors',family=FAMILY,
                    method=method,cost_completion=completion,macros=certs,new_work_qubits=0,fullspace_scope='Completed CPG unitary and declared cost; canonical equality only on Feas',
                    author_algorithm_unchanged=True,no_HPO_or_seed_search=True,original_frozen_library_not_modified=True,
                    adapter_source_sha256=sha(ROOT/'code/matched_reference_v2.py'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',required=True);a.add_argument('--method',choices=METHODS,required=True);a.add_argument('--out',required=True);x=a.parse_args()
    # Explicit construction hook; the immutable common pipeline/source checks remain unchanged.
    E.construct=build
    E.run(x.case,x.method,x.out)
