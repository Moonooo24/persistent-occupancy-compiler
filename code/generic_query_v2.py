"""Previously proved L2/KG construction extended beyond the stored small bank."""
from pathlib import Path
from functools import lru_cache
import sys,importlib.util
from runtime import ROOT,QC,Parameter,core,sha
from phase_safe import native,normalize_owned_phase

SNAP=ROOT/'vendor/query_sharing_v21';sys.path.insert(0,str(SNAP/'code'))
s=importlib.util.spec_from_file_location('journal_generic_conditional_phase',SNAP/'code/mis_multitarget_constructive_phase_v1.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)

@lru_cache(maxsize=None)
def template(rank,exact=False):
    assert rank>=4;b=Parameter('generic_beta');raw,cert=C.constructive_phase_group(rank+1,list(range(1,rank+1)),[0],b,L_size=2,layout='prefix')
    raw,phasecert=normalize_owned_phase(raw)
    if exact:
        half=raw.assign_parameters({b:b/2});q=half.compose(half);q,pc=normalize_owned_phase(q)
        cert=dict(parent=cert,exact_pair_of_identical_half_angle_phase_kernels=True,
                  proof='K(beta/2)^2=U_NOR(beta) because the completed residual is a beta-independent diagonal involution, commutes with this local NOR rotation and fixes the active sector')
        raw=q
    out=native(raw,1);bound=(12*rank-24) if exact else (6*rank-12)
    assert core.costs(out)['cx']<=bound and out.num_qubits==rank+1
    return out,dict(source_constructor_sha256=sha(SNAP/'code/mis_multitarget_constructive_phase_v1.py'),rank=rank,
                    full_rank_truth_table_created=False,support='target and all controls only',new_work=0,
                    scope='global exact NOR rotation' if exact else 'canonical on target0 arbitrarycontrols or target1 controls0; full controlled diagonal-extension contract',
                    exact=exact,certificate=cert,phase_normalization=phasecert,CX_bound=bound)


def kernel(n,target,controls,beta,exact=False):
    q,cert=template(len(controls),exact);mapping=[target]+list(controls);out=QC(n)
    assert len(set(mapping))==len(mapping) and len(mapping)==q.num_qubits
    out.compose(q.assign_parameters({p:beta for p in q.parameters}),mapping,inplace=True)
    return out,dict(cert,physical_mapping=mapping,arbitrary_original_spectators_untouched=True)
