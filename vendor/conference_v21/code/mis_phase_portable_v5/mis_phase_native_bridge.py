"""MIS portable Clifford+RZ bridge, exact source6 lift with scalable phase repair."""
from pathlib import Path
import math,time,hashlib,importlib.util
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Clifford,Pauli
from mis_phase_ch_scalar import scalar_identity
def exact_clifford_offset(theta):
    """No tolerance, no replacement of the original angle by a rounded value."""
    parameters=getattr(theta,'parameters',set())
    try:
        bound=theta.bind({p:0 for p in parameters}) if parameters else theta
        offset=float(bound)
        if not math.isfinite(offset):return None
        k=round(offset/(math.pi/2))
        if offset!=k*(math.pi/2):return None
        remainder=theta-offset
        if not bool(theta==remainder+offset):return None
        return remainder,offset,int(k)
    except (TypeError,ValueError,ArithmeticError):return None

def lift(qc,variant='all_rz_lift'):
    """Invariant after each prefix: U_prefix = C_raw * R_sequence, excluding phi.
    C tableau has the same conjugation action as exact C_raw, including SX.
    RZ on q becomes C_raw^dagger Z_q C_raw; raw Clifford factors append unchanged."""
    n=qc.num_qubits;C=Clifford(QuantumCircuit(n));raw=QuantumCircuit(n)
    words=[];symbols=[];substitution={};records=[];offsets=[];source_rz_count=0
    for position,ins in enumerate(qc.data):
        name=ins.operation.name;wires=[qc.find_bit(q).index for q in ins.qubits]
        assert not ins.clbits,('classical operation unsupported',name)
        if name in ('barrier','id'):continue
        if name=='rz':
            source_rz_count+=1
            q=wires[0];theta=ins.operation.params[0]
            if variant=='clifford_normalized':
                split=exact_clifford_offset(theta)
                if split is not None:
                    remainder,offset,k=split
                    offsets.append(dict(position=position,multiple_pi_over2=k,exact_numeric_offset=str(offset),
                                        original_angle=str(theta),remainder=str(remainder),symbolic_recomposition_equal=True))
                    if offset!=0:
                        # Keep RZ(offset) itself, including RZ(2pi)=-I and RZ(pi/2)=e^-i*pi/4 S.
                        raw.rz(offset,q)
                        D=QuantumCircuit(n);D.rz(offset,q);C=C.compose(Clifford(D))
                    theta=remainder
                    if not getattr(theta,'parameters',set()) and float(theta)==0.:continue
            label=['I']*n;label[n-1-q]='Z'
            p=Pauli(''.join(label)).evolve(C,frame='h')
            text=p.to_label();sign=1
            if text.startswith('-') and not text.startswith('-i'):sign=-1;text=text[1:]
            elif text.startswith('+'):text=text[1:]
            assert len(text)==n and set(text)<={'I','X','Y','Z'} and text!='I'*n,text
            symbol=Parameter('bridge_rz_'+str(len(symbols)).zfill(5))
            words.append(text);symbols.append(sign*symbol);substitution[symbol]=theta
            records.append(dict(position=position,word=text,sign=sign,symbol=symbol.name,
                                original_angle=str(theta),fixed=not bool(getattr(theta,'parameters',set()))))
        else:
            assert name in ('sx','sxdg','x','y','z','h','s','sdg','cx','cz','swap','ecr'),name
            raw.append(ins.operation,wires,[])
            D=QuantumCircuit(n);D.append(ins.operation,wires,[])
            C=C.compose(Clifford(D)) # chronological append: C_new = D C
    assert Clifford(raw)==C
    return dict(words=words,angles=symbols,substitution=substitution,raw_tail=raw,
                original_global_phase=qc.global_phase,records=records,clifford_offsets=offsets,source_rz_count=source_rz_count,variant=variant)
def ce_module():
    path=Path(__file__).resolve().parent/'mis_phase_CE_module.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='72596242fea34c4f3fcaeee6bcc046d89e5c203d9dae3cfeba3ef324741d17a7'
    spec=importlib.util.spec_from_file_location('mis_phase_bridge_official_CE',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def restore_scalar(qc):
    zero=qc.assign_parameters({p:0. for p in qc.parameters})
    cert=scalar_identity(zero)
    qc.global_phase-=cert['eighth_root_k']*math.pi/4
    return cert

def bridge(qc,mode='default',variant='clifford_normalized'):
    start=time.monotonic();n=qc.num_qubits
    lifted=lift(qc,variant)
    if lifted['words']:
        CE=ce_module();fn=CE.CE_recur_tree if mode=='default' else CE.CE_recur_tree_seq
        reduced,tail,ordered=fn(entanglers=lifted['words'],params=lifted['angles'],barrier=False,threshold=1)
        tail=tail.to_circuit() if isinstance(tail,Clifford) else tail
        network=reduced.compose(tail)
        ce_certificate=restore_scalar(network)
        restored=network.assign_parameters(lifted['substitution'])
    else:restored=QuantumCircuit(n);ce_certificate=dict(no_rotations=True)
    original_tail=lifted['raw_tail'];compressed=Clifford(original_tail).to_circuit()
    loop=compressed.compose(original_tail.inverse())
    tail_certificate=scalar_identity(loop)
    compressed.global_phase-=tail_certificate['eighth_root_k']*math.pi/4
    restored.compose(compressed,inplace=True)
    restored.global_phase+=lifted['original_global_phase']
    return restored,dict(mode=mode,variant=variant,source_rz_count=lifted['source_rz_count'],
       independent_rotation_symbols=len(lifted['words']),promoted_fixed_rz=sum(r['fixed'] for r in lifted['records']),
       clifford_offsets=lifted['clifford_offsets'],ce_scalar_certificate=ce_certificate,
       tail_scalar_certificate=tail_certificate,original_global_phase_restored=True,
       full_tail_physically_restored=True,original_zero_angle_identity_assumed=False,
       angle_dependent_alignment=False,large_Statevector_used=False,
       scope='Exact full-Hilbert rewrite of the supplied source including its outside-domain phase',
       source_records_sha256=hashlib.sha256(repr(lifted['records']).encode()).hexdigest(),
       seconds=time.monotonic()-start)
