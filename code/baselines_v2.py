"""Fixed reference constructions; author algorithms and settings are unchanged."""
from fractions import Fraction
from pathlib import Path
import importlib.util

from runtime import QC, Parameter, core, VENDOR, sha
from qiskit.quantum_info import Clifford
import auto_reuse_v2 as A


REFERENCES = ('ref_exact_original','ref_t3_original','ref_t3_clique',
              'ref_cpg_open','ref_cpg_closed','ref_quclear_tree','ref_quclear_seq')


def closed_cpg(case):
    spec=A.graph_spec(case);p=int(case.get('sweeps',2));beta=[Parameter('beta'+str(k)) for k in range(p)];gamma=[Parameter('gamma'+str(k)) for k in range(p-1)];local_beta=Parameter('local_beta');local={};records=[]
    for i in range(spec['n']):
        blocks=A.supports(spec,i);factors=[A.mask_of(c) for c in blocks];r=len(factors);q=QC(spec['n'])
        if r<=8:
            q.h(i);current=0
            for k in range(1<<r):
                g=k^(k>>1);mask=0
                for j,f in enumerate(factors):
                    if g>>j&1:mask^=f
                for j in range(spec['n']):
                    if (current^mask)>>j&1:q.cx(j,i)
                q.rz(2*local_beta/float(1<<r),i);current=mask
            for j in range(spec['n']):
                if current>>j&1:q.cx(j,i)
            q.h(i)
            cert=dict(target=i,query='fixed_closed_target_Gray',blocks=blocks,rank=r)
        else:
            pp,controls=A.parity_network(spec['n'],factors)
            for a,b in pp:q.cx(a,b)
            kernel,kc=A.exact_kernel(spec['n'],i,controls,local_beta);q.compose(kernel,inplace=True)
            for a,b in reversed(pp):q.cx(a,b)
            cert=dict(target=i,query='exact_affine_fallback',blocks=blocks,rank=r,kernel=kc)
        local[i]=q;records.append(cert)
    q=QC(spec['n']);orders=case.get('sweep_orders',[spec['order']]*p)
    for k in range(p):
        for i in orders[k]:q.compose(local[i].assign_parameters({local_beta:beta[k]}),inplace=True)
        if k<p-1:q.compose(A.weighted_cost(spec,set(),gamma[k],'clique')[0],inplace=True)
    return q,dict(reference='fixed_closed_CPG_and_clique_cost',macros=records,author_parameter_search=False)


def canonical_exact(case):
    spec=A.graph_spec(case);p=int(case.get('sweeps',2));beta=[Parameter('beta'+str(k)) for k in range(p)];gamma=[Parameter('gamma'+str(k)) for k in range(p-1)];b=Parameter('local_beta');local={};certs=[]
    for i in range(spec['n']):
        local[i],cert=A.exact_kernel(spec['n'],i,sorted(spec['adj'][i]),b);certs.append(cert)
    q=QC(spec['n']);orders=case.get('sweep_orders',[spec['order']]*p)
    for k in range(p):
        for i in orders[k]:q.compose(local[i].assign_parameters({b:beta[k]}),inplace=True)
        if k<p-1:q.compose(A.weighted_cost(spec,set(),gamma[k],'original')[0],inplace=True)
    return q,dict(reference='pinned_exact_original_NOR_noextra',macros=certs,full_canonical_domain=True,parameter_search=False)


def pauli_source(case):
    spec=A.graph_spec(case);p=int(case.get('sweeps',2));beta=[Parameter('beta'+str(k)) for k in range(p)];gamma=[Parameter('gamma'+str(k)) for k in range(p-1)];words=[];angles=[];scalar=0;orders=case.get('sweep_orders',[spec['order']]*p)
    def word(i,mask):
        return ''.join('X' if j==i else 'Z' if mask>>j&1 else 'I' for j in reversed(range(spec['n'])))
    for k in range(p):
        for i in orders[k]:
            blocks=A.supports(spec,i);r=len(blocks)
            for t in range(1<<r):
                g=t^(t>>1);mask=A.mask_of([j for z,c in enumerate(blocks) if g>>z&1 for j in c]);words.append(word(i,mask));angles.append(2*beta[k]/float(1<<r))
        if k<p-1:
            for c in spec['parts']:
                a=c[0];wa=spec['weights'][a]
                if wa:words.append(word(-1,A.mask_of(c)));angles.append(gamma[k]*float(wa));scalar+=gamma[k]*float(wa)/2
                for j in c[1:]:
                    w=spec['weights'][j]-wa
                    if w:words.append(word(-1,1<<j));angles.append(gamma[k]*float(w));scalar+=gamma[k]*float(w)/2
    return words,angles,scalar


def quclear(case,mode):
    spec=A.graph_spec(case);max_rank=max(len(A.supports(spec,i)) for i in range(spec['n']))
    if max_rank>12:
        raise ValueError('Materialized CPG Pauli source rank exceeds explicit memory guard 12; no resource result fabricated')
    words,angles,scalar=pauli_source(case);path=VENDOR/'code/mis_phase_portable_v5/mis_phase_CE_module.py';assert sha(path)==core.CE_SHA
    sp=importlib.util.spec_from_file_location('fixed_author_QuCLEAR_'+mode,path);CE=importlib.util.module_from_spec(sp);sp.loader.exec_module(CE)
    fn=CE.CE_recur_tree if mode=='tree' else CE.CE_recur_tree_seq
    body,tail,ordered=fn(entanglers=words,params=angles,barrier=False,threshold=1)
    tail=tail.to_circuit() if isinstance(tail,Clifford) else tail;q=body.compose(tail);q.global_phase+=scalar;cert=core.scalar_certificate(q,strict=True)
    return q,dict(reference='unchanged_official_QuCLEAR_'+mode,author_source_sha256=core.CE_SHA,
                  threshold=1,barrier=False,full_tail_reconnected=True,CH_scalar=cert,
                  Pauli_terms=len(words),cost='same_CPG_projectors_and_clique_cost',no_HPO_or_seed_search=True,
                  literal_noncommuting_order_preserved=True,fullspace_reference='completed CPG product, canonical only on Feas')


def build(case,method):
    if method=='ref_exact_original':return canonical_exact(case)
    if method in ('ref_t3_original','ref_t3_clique'):
        q,cert=A.build(case,cache='none',route='t3',cost_completion='original' if method.endswith('original') else 'clique')
        cert.update(reference='Compact2504 Thm3 fixed min-pivot affine instance plus its cited frozen Vale primitive',
                    not_author_reported_or_unique_default=True,not_Thm3_optimality=True,no_basis_or_parameter_search=True)
        return q,cert
    if method=='ref_cpg_open':return A.build(case,cache='none',route='walsh')
    if method=='ref_cpg_closed':return closed_cpg(case)
    if method.startswith('ref_quclear_'):return quclear(case,method.rsplit('_',1)[1])
    raise ValueError(method)
