"""Lossless QPY I/O v3: complete historical replay lookup tables.
Only serialization copies receive additional historical Parameter objects.
Current QC expressions, current parameters, instructions and f64 values are
never rewritten. QPY's original v15 reader replays the same original history.
"""
import base64,hashlib,struct
from qiskit import QuantumCircuit,qpy as _native_qpy
from qiskit.circuit import Parameter,ParameterExpression
from qiskit.circuit.parameterexpression import _INSTRUCTION,_SUBS
from qiskit.qpy.binary_io import value as _value
from qiskit.qpy.common import QPY_VERSION
KEY='__project_lossless_qpy_io_v3__'
OLD_KEY='__project_lossless_qpy_io_v2__'
FORMAT='qpy-complete-history-global-phase-v3'
SAFE_NAME='lossless_qpy_v3'

def historical_parameters(obj):
    seen=set();found={}
    def visit(x):
        if isinstance(x,Parameter):found[x.uuid]=x;return
        if id(x) in seen:return
        seen.add(id(x))
        if isinstance(x,ParameterExpression):
            for key,val in x._parameter_symbols.items():visit(key);visit(val)
            for inst in x._qpy_replay:
                if isinstance(inst,_SUBS):visit(inst.binds)
                else:visit(inst.lhs);visit(inst.rhs)
        elif isinstance(x,dict):
            for key,val in x.items():visit(key);visit(val)
        elif isinstance(x,(tuple,list)):
            for item in x:visit(item)
    visit(obj);return found

def serialization_copy(obj,memo=None):
    if isinstance(obj,Parameter) or not isinstance(obj,ParameterExpression):return obj
    if memo is None:memo={}
    if id(obj) in memo:return memo[id(obj)]
    out=object.__new__(ParameterExpression);memo[id(obj)]=out
    for name in ParameterExpression.__slots__:
        if hasattr(obj,name):setattr(out,name,getattr(obj,name))
    symbols=dict(obj._parameter_symbols)
    for param in historical_parameters(obj).values():symbols.setdefault(param,param._symbol_expr)
    out._parameter_symbols=dict(sorted(symbols.items(),key=lambda item:str(item[0].uuid)))
    replay=[]
    for inst in obj._qpy_replay:
        if isinstance(inst,_SUBS):replay.append(_SUBS({key:serialization_copy(val,memo) for key,val in inst.binds.items()}))
        else:replay.append(_INSTRUCTION(inst.op,serialization_copy(inst.lhs,memo),serialization_copy(inst.rhs,memo)))
    out._qpy_replay=replay;return out

def current_parameters(obj):
    return sorted((p.name,str(p.uuid)) for p in getattr(obj,'parameters',()))

def typed_phase_bytes(phase):
    tag,payload=_value.dumps_value(serialization_copy(phase),version=QPY_VERSION)
    return struct.pack('!I',QPY_VERSION)+tag+payload

def load_phase(data):
    version=struct.unpack('!I',data[:4])[0]
    return _value.loads_value(data[4:5],data[5:],version=version,vectors={})

def _pack(circuit):
    if not isinstance(circuit,QuantumCircuit):raise TypeError('Only project QuantumCircuit artifacts are supported')
    original=circuit.global_phase;phase=typed_phase_bytes(original)
    # Validate self-contained reconstruction before a worker receives the file.
    restored=load_phase(phase)
    if current_parameters(restored)!=current_parameters(original):raise ValueError('Current phase Parameter UUIDs changed in serialization')
    if str(restored)!=str(original):raise ValueError('Current phase symbolic expression changed in serialization')
    packet=dict(format=FORMAT,typed_phase_b64=base64.b64encode(phase).decode('ascii'),typed_phase_sha256=hashlib.sha256(phase).hexdigest(),
        current_parameters=current_parameters(original),symbolic_expression_sha256=hashlib.sha256(str(original).encode('utf8')).hexdigest(),
        historical_parameters=sorted((p.name,str(p.uuid)) for p in historical_parameters(original).values()),
        original_name=circuit.name,original_metadata=circuit.metadata)
    safe=circuit.copy();safe.global_phase=0;safe.name=SAFE_NAME;safe.metadata={KEY:packet};return safe

def restore(circuit):
    metadata=circuit.metadata
    if not isinstance(metadata,dict):return circuit
    if KEY in metadata:packet=metadata[KEY];new=True
    elif OLD_KEY in metadata:packet=metadata[OLD_KEY];new=False
    else:return circuit
    expected=FORMAT if new else 'qpy-value-global-phase-v2'
    if not isinstance(packet,dict) or packet.get('format')!=expected:raise ValueError('Invalid project QPY envelope')
    phase=base64.b64decode(packet['typed_phase_b64'],validate=True)
    if len(phase)<5 or hashlib.sha256(phase).hexdigest()!=packet['typed_phase_sha256']:raise ValueError('Project QPY phase hash mismatch')
    original=load_phase(phase)
    if new:
        if current_parameters(original)!=[tuple(x) for x in packet['current_parameters']]:raise ValueError('Restored current UUID table differs')
        if hashlib.sha256(str(original).encode('utf8')).hexdigest()!=packet['symbolic_expression_sha256']:raise ValueError('Restored current symbol expression differs')
    circuit.global_phase=original;circuit.name=packet['original_name'];circuit.metadata=packet['original_metadata'];return circuit

def dump(programs,file_obj,**kwargs):
    items=[programs] if isinstance(programs,QuantumCircuit) else list(programs)
    return _native_qpy.dump([_pack(circuit) for circuit in items],file_obj,**kwargs)

def load(file_obj,**kwargs):return [restore(circuit) for circuit in _native_qpy.load(file_obj,**kwargs)]
