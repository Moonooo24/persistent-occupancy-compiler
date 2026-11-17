"""Owned affine phase canonicalisation and finite phase-safe native pipeline."""
from fractions import Fraction
from collections import defaultdict
import math
import sympy as sp
from runtime import QC,core
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition,InverseCancellation
from qiskit.circuit.library import CXGate,XGate

BACKEND='phase_safe_bounded_v2'

def real_number(x):
    v=complex(x)
    if v.imag or not math.isfinite(v.real):raise ValueError('Nonfinite/nonreal affine phase')
    return float(v.real)


def canonical_affine(phi):
    params=list(getattr(phi,'parameters',()))
    if not params:return real_number(phi),dict(affine=True,constant_only=True,parameter_binding_used=False)
    symbolic=sp.expand(phi.sympify());residual=symbolic;coefficients=[]
    for p in sorted(params,key=lambda x:str(x.uuid)):
        derivative=sp.diff(symbolic,p.sympify())
        if derivative.free_symbols:raise ValueError('Nonaffine owned global phase')
        public=phi.gradient(p)
        if getattr(public,'parameters',set()):
            sy=public.sympify()
            if sy.free_symbols:raise ValueError('Nonconstant public phase derivative')
            public=sy
        a=real_number(public)
        if a!=real_number(derivative):raise ValueError('Public and symbolic phase derivatives disagree')
        residual-=derivative*p.sympify()
        if a:coefficients.append((p,a,sp.Rational(derivative)))
    residual=sp.simplify(residual)
    if residual.free_symbols:raise ValueError('Constant phase retains true free symbols')
    c=real_number(residual);out=c;errors=[]
    for p,a,exact in coefficients:
        out=out+a*p
        error=abs(Fraction.from_float(a)-Fraction(int(exact.p),int(exact.q)))
        errors.append(dict(parameter=p.name,UUID=str(p.uuid),coefficient=a,extraction_absolute_error=str(error)))
    const_exact=sp.Rational(residual);constant_error=abs(Fraction.from_float(c)-Fraction(int(const_exact.p),int(const_exact.q)))
    return out,dict(affine=True,constant=c,constant_extraction_absolute_error=str(constant_error),coefficients=errors,
                    removed_only_exact_zero_derivatives=True,parameter_binding_used=False,
                    error_scope='Phase representation: Econstant + sum Eparameter*abs(parameter), plus final floating evaluation/modulo; no nontrivial uniform small error claimed on unbounded angle space')


def normalize_owned_phase(q):
    value,cert=canonical_affine(q.global_phase);out=q.copy();out.global_phase=value
    return out,cert


def reject_embedded_numeric_blocks(q):
    """Inspect definitions, without matrix evaluation, before any synthesis pass."""
    from qiskit.circuit.library import UnitaryGate,get_standard_gate_name_mapping
    trusted={op.base_class for op in get_standard_gate_name_mapping().values()}
    active=set();visited=set()
    def visit(op):
        if isinstance(op,UnitaryGate) or op.name=='unitary':raise ValueError('Numeric matrix blocks, including embedded definitions, are not accepted')
        if op.base_class in trusted:return
        key=id(op)
        if key in active:raise ValueError('Cyclic custom gate definition')
        if key in visited:return
        active.add(key);definition=op.definition
        if definition is None:raise ValueError('Opaque nonstandard operation: '+op.name)
        for child in definition.data:visit(child.operation)
        active.remove(key);visited.add(key)
    for ins in q.data:visit(ins.operation)


def native(q,level=1):
    """No iterative fixed point or numeric two-qubit UnitarySynthesis is used."""
    reject_embedded_numeric_blocks(q)
    source,initial_phase=normalize_owned_phase(q);ledger=[]
    if any(x.operation.name=='unitary' for x in source.data):raise ValueError('Numeric matrix blocks are not accepted by the phase-safe owned pipeline')
    def no_matrix_blocks(**kwargs):
        if kwargs['dag'].count_ops().get('unitary',0):raise ValueError('Unexpected numeric matrix block in phase-safe translation')
    with core.phase_policy(ledger):
        out=transpile(source,basis_gates=core.BASIS,optimization_level=0,
                      seed_transpiler=core.SEED,qubits_initially_zero=False,callback=no_matrix_blocks)
        for _ in range(1 if level<3 else 3):
            out=PassManager([Optimize1qGatesDecomposition(basis=core.BASIS),InverseCancellation([CXGate(),XGate()])]).run(out)
            out,_=normalize_owned_phase(out)
    assert out.num_qubits==q.num_qubits and all(x.operation.name in core.BASIS for x in out.data)
    perm=out.layout.final_index_layout(filter_ancillas=True) if out.layout else list(range(q.num_qubits))
    assert perm==list(range(q.num_qubits)),'Unexpected native-only permutation'
    physical=QC(q.num_qubits);physical.global_phase=out.global_phase
    for x in out.data:physical.append(x.operation.copy(),[out.find_bit(w).index for w in x.qubits])
    physical.metadata=dict(backend_policy=BACKEND,translation_preset_level=0,finite_local_rounds=1 if level<3 else 3,
                           no_numeric_two_qubit_synthesis=True,no_fixed_point_iterations=True,
                           initial_phase_certificate=initial_phase,phase_guard=core.POLICY,qubits_initially_zero=False)
    return physical
