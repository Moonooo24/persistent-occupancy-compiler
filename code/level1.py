"""Apply the paper's fixed Qiskit level-1 check to one compiled circuit."""
import argparse
from pathlib import Path
import json

from runtime import core, io, G
from qiskit import transpile
from phase_safe import normalize_owned_phase, reject_embedded_numeric_blocks

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("input",type=Path)
parser.add_argument("output",type=Path)
args=parser.parse_args()
with args.input.open("rb") as stream:
    circuit=io.load(stream)[0]
reject_embedded_numeric_blocks(circuit)
circuit,_=normalize_owned_phase(circuit)
def guard(**state):
    assert type(state["pass_"]).__name__!="ConsolidateBlocks"
    assert not state["dag"].count_ops().get("unitary",0)
with core.phase_policy([]):
    out=transpile(circuit,basis_gates=core.BASIS,optimization_level=1,
                  seed_transpiler=core.SEED,qubits_initially_zero=False,callback=guard)
out,_=normalize_owned_phase(out)
n=out.num_qubits
assert n==circuit.num_qubits and set(out.count_ops()).issubset(set(core.BASIS))
assert {p.name for p in circuit.parameters}=={p.name for p in out.parameters}
assert out.layout is None or out.layout.final_index_layout(filter_ancillas=True)==list(range(n))
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open("xb") as stream:
    io.dump(out,stream)
with args.output.open("rb") as stream:
    assert G.fingerprint(io.load(stream)[0])==G.fingerprint(out)
print(json.dumps(core.costs(out)))