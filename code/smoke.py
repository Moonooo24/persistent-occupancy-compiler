"""Verify compiled and routed circuits on the paper's seven-vertex graph."""
from pathlib import Path
import json
from runtime import ROOT, core, io
from journal_api_v3 import compile
from verify_auto import checks

case = json.loads((ROOT / "examples/seven_vertex.json").read_text(encoding="utf-8"))
folder = ROOT / "results/smoke"
folder.mkdir(parents=True, exist_ok=True)
for kind in ("native", "grid"):
    q, cert, alternatives = compile(case, target=None if kind == "native" else "grid", include_shared=False)
    assert q.num_qubits == case["n"] and alternatives
    qpy = folder / (kind+".qpy")
    with qpy.open("wb") as f:
        io.dump(q,f)
    with qpy.open("rb") as f:
        restored=io.load(f)[0]
    evidence = checks(restored, case, all_columns=True)
    error = max(row["max_complex_error"] for row in evidence)
    assert error < 1e-10
    result={"target":kind,"metrics":core.costs(restored),"max_complex_error":error,"candidate_count":len(alternatives)}
    (folder/(kind+".json")).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (folder/(kind+"_certificate.json")).write_text(json.dumps(cert,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)