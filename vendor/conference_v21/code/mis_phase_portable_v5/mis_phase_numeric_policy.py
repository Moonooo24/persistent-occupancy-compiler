"""Qiskit 2.1.2 native scalar ledger, native_phase_fsum_x_v2.

Only CommutativeCancellation's scalar is replaced. Its output instructions,
parameters and float64 values are untouched. The caller must disable compiler
parallelism and use this context only for the common whole3 stage.

Native input has no RX. The source groups X gates by certified commutation sets:
an even group disappears; an odd group of k>=3 becomes RX(theta), theta=k*pi
up to bounded float summation error. X^k=exp(i*k*pi/2)*RX(k*pi), so each emitted
RX contributes theta/2 to the scalar ledger. Per-wire X-count parity and a
conservative aggregate angle-error bound certify this finite extension.

The RZ ledger sums the exact float definition phases used by this Qiskit
version, with math.fsum. This is a scalar policy, not a repair of upstream gate
angle approximation. Strict output-unitary validation remains required.
Unknown scope leaves the upstream output entirely unchanged.
"""
from collections import Counter
from contextlib import contextmanager
import math
import qiskit
from qiskit.transpiler.passes import CommutativeCancellation

POLICY='native_phase_fsum_x_v2'
MAX_ABS_ANGLE=(1<<20)*math.pi
MAX_LEDGER_TERMS=1<<20
MAX_X_ANGLE_ERROR_BOUND=1e-10
_active=False

def _finite_angle(value):
    return isinstance(value,(int,float)) and math.isfinite(float(value)) and abs(float(value))<=MAX_ABS_ANGLE

def scope_snapshot(dag,after=False):
    allowed={'rz','sx','x','cx'}
    if after:allowed.add('rx')
    symbolic=Counter();sx=Counter();xs=Counter();rz=Counter();rx=[]
    if not getattr(dag.global_phase,'parameters',set()) and not _finite_angle(dag.global_phase):
        return None,'global phase is not a finite supported numeric scalar or a free symbolic expression'
    for node in dag.op_nodes():
        name=node.op.name
        if name not in allowed:return None,'unsupported operation '+name
        wires=tuple(dag.find_bit(w).index for w in node.qargs)
        if name=='sx':sx[wires]+=1
        elif name=='x':xs[wires]+=1
        elif node.op.is_parameterized():
            if name!='rz':return None,'unsupported symbolic operation '+name
            symbolic[(wires,node.op.params[0])]+=1
        elif name in ('rz','rx'):
            angle=node.op.params[0]
            if not _finite_angle(angle):return None,name+' angle is not a finite bounded native numeric scalar'
            if name=='rz':rz[float(angle)]+=1
            else:rx.append((wires,float(angle)))
    return dict(symbolic=symbolic,sx=sx,x=xs,rz=rz,rx=rx),None

def definition_phase(angle):
    # The exact float expression in this Qiskit RZ definition, not -sum(angle)/2.
    return (-0.5*angle)%math.tau

def _ledger(before,after):
    if before['symbolic']!=after['symbolic']:
        return None,'symbolic RZ multiset changed'
    if before['sx']!=after['sx']:
        return None,'untouched SX multiset changed'
    lost=before['x'].copy();lost.subtract(after['x'])
    consumed=Counter();x_terms=[];x_records=[];error_terms=[]
    for wires,angle in after['rx']:
        k=round(angle/math.pi)
        if k<3 or k%2!=1:
            return None,'new RX is not an odd positive X group of size at least3'
        # The subtraction uses the existing output angle; no angle is rounded or replaced.
        bound=abs(angle-k*math.pi)+k*math.ulp(math.pi)/2
        error_terms.append(bound)
        consumed[wires]+=k
        x_terms.append(angle/2)
        x_records.append(dict(wires=list(wires),x_count=k,rx_angle=angle,scalar_term=angle/2,angle_error_bound=bound))
    for wires in set(lost)|set(consumed):
        remaining=lost[wires]-consumed[wires]
        if remaining<0 or remaining%2:
            return None,'per-wire lost X count does not match new RX groups plus even cancellations'
    angle_error_bound=math.fsum(error_terms)
    if angle_error_bound>MAX_X_ANGLE_ERROR_BOUND:
        return None,'aggregate X-to-RX angle error bound exceeds the finite scope'
    differences=before['rz'].copy();differences.subtract(after['rz'])
    term_count=sum(abs(count) for count in differences.values())+len(x_terms)
    if term_count>MAX_LEDGER_TERMS:
        return None,'scalar ledger exceeds the finite term bound'
    terms=list(x_terms)
    for angle,count in differences.items():
        if count:
            phase=definition_phase(angle)
            terms.extend([phase if count>0 else -phase]*abs(count))
    raw=math.fsum(terms)
    return dict(delta=math.remainder(raw,math.tau),raw=raw,term_count=term_count,
                rz_changed_terms=term_count-len(x_terms),x_merge_terms=x_records,
                x_angle_error_bound=angle_error_bound),None

@contextmanager
def phase_policy(audit=None):
    global _active
    if qiskit.__version__!='2.1.2':
        raise RuntimeError('This native scalar ledger is scoped to Qiskit2.1.2')
    if _active:raise RuntimeError('Nested native scalar policy contexts are unsupported')
    original=CommutativeCancellation.run
    def stable_run(self,dag):
        phi=dag.global_phase
        before,reason=scope_snapshot(dag)
        out=original(self,dag)
        if out is None:out=dag
        after,after_reason=scope_snapshot(out,after=True)
        correction=None
        if before is not None and after is not None:
            correction,reason=_ledger(before,after)
        if correction is None:
            if audit is not None:
                audit.append(dict(policy=POLICY,applied=False,reason=reason or after_reason,
                                  upstream_preserved=True,body_mutated_by_policy=False))
            return out
        upstream_phase=out.global_phase
        out.global_phase=phi+correction['delta']
        if audit is not None:
            audit.append(dict(policy=POLICY,applied=True,scope='native_input_no_RX; exact_X_commutation_groups_and_RZ_definition_phases',
                symbolic_rz_preserved=True,sx_preserved=True,input_rx_absent=True,
                phase_before=str(phi),phase_upstream=str(upstream_phase),phase_stable=str(out.global_phase),
                compensated_raw_delta=correction['raw'],compensated_reduced_delta=correction['delta'],
                changed_rz_terms=correction['rz_changed_terms'],ledger_terms=correction['term_count'],
                x_merge_terms=correction['x_merge_terms'],x_angle_error_bound=correction['x_angle_error_bound'],
                body_mutated_by_policy=False,arbitrary_angle_alignment=False))
        return out
    _active=True
    CommutativeCancellation.run=stable_run
    try:yield
    finally:
        CommutativeCancellation.run=original
        _active=False

