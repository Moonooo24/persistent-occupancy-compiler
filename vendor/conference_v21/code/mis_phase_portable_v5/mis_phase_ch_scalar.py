"""Polynomial-memory global-phase-preserving Clifford simulation for CE scalar."""
import math,cmath
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from mis_phase_ch_form import StabilizerStateChForm
def _steps(angle):
    angle=float(angle)
    steps=round(angle/(math.pi/2))
    if abs(angle-steps*(math.pi/2))>2e-10:
        raise ValueError('Non-Clifford rotation in zero-angle scalar circuit')
    return steps
def simulate(circuit,initial_state=0):
    state=StabilizerStateChForm(circuit.num_qubits,int(initial_state))
    count=Counter()
    def apply(qc,mapping):
        state.apply_global_phase(cmath.exp(1j*float(qc.global_phase)))
        for ins in qc.data:
            if ins.clbits:raise ValueError('Classical/measurement operation is not a Clifford unitary')
            physical=[mapping[qc.find_bit(q).index] for q in ins.qubits]
            axes=[state.n-1-q for q in physical]  # Cirq big endian versus Qiskit little endian.
            name=ins.operation.name
            count[name]+=1
            if name in ('barrier','id'):continue
            if name=='h':state.apply_h(axes[0])
            elif name=='s':state.apply_z(axes[0],.5)
            elif name=='sdg':state.apply_z(axes[0],-.5)
            elif name=='x':state.apply_x(axes[0])
            elif name=='y':state.apply_y(axes[0])
            elif name=='z':state.apply_z(axes[0])
            elif name=='sx':state.apply_x(axes[0],.5)
            elif name=='sxdg':state.apply_x(axes[0],-.5)
            elif name in ('rx','ry','rz'):
                exponent=_steps(ins.operation.params[0])/2
                getattr(state,{'rx':'apply_x','ry':'apply_y','rz':'apply_z'}[name])(axes[0],exponent,-.5)
            elif name in ('p','u1'):
                state.apply_z(axes[0],_steps(ins.operation.params[0])/2)
            elif name in ('u','u3','u2'):
                theta,phi,lam=(ins.operation.params if name!='u2' else (math.pi/2,*ins.operation.params))
                state.apply_z(axes[0],_steps(lam)/2)
                state.apply_y(axes[0],_steps(theta)/2,-.5)
                state.apply_z(axes[0],_steps(phi)/2)
            elif name=='cx':state.apply_cx(*axes)
            elif name=='cz':state.apply_cz(*axes)
            elif name=='swap':
                a,b=axes;state.apply_cx(a,b);state.apply_cx(b,a);state.apply_cx(a,b)
            elif name=='cy':
                a,b=axes;state.apply_z(b,-.5);state.apply_cx(a,b);state.apply_z(b,.5)
            elif name=='gphase':
                state.apply_global_phase(cmath.exp(1j*float(ins.operation.params[0])))
            elif ins.operation.definition is not None:
                apply(ins.operation.definition,physical)
            else:
                raise ValueError('Unsupported Clifford instruction; no tableau-to-circuit or dense fallback: '+name)
    apply(circuit,list(range(circuit.num_qubits)))
    memory=sum(getattr(state,key).nbytes for key in ('F','G','M','gamma','v','s'))
    return state,dict(qubits=state.n,numerical_state_storage_bytes=memory,gate_counts=dict(count),
                      statevector_allocated=False,phase_preserving_omega=True,source='Official Cirq v1.6.1 minimal CH numerical methods')
def scalar_identity(circuit):
    assert not circuit.parameters
    assert Clifford(circuit)==Clifford(QuantumCircuit(circuit.num_qubits)),'Non-scalar zero-angle Clifford'
    state,metrics=simulate(circuit)
    amplitude=complex(state.inner_product_of_state_and_x(0))
    k=round(cmath.phase(amplitude)*4/math.pi)%8
    root=cmath.exp(1j*k*math.pi/4)
    error=abs(amplitude-root)
    if abs(abs(amplitude)-1)>2e-9 or error>2e-9:
        raise ValueError(('Not the certified eighth-root scalar',amplitude,error))
    return dict(status='strict_CH_eighth_root_recovered',eighth_root_k=k,
                phase_correction_multiple_pi_over4=-k,amplitude0=[amplitude.real,amplitude.imag],
                calibration_error=error,tableau_identity_exact=True,all_hilbert_scalar=True,
                no_angle_dependent_alignment=True,**metrics)

