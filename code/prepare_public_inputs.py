"""Reconstruct the public cases from byte-verified upstream DIMACS inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def normalize(blob, expected_n):
    edges = set()
    n = None
    for line in blob.decode("ascii", errors="replace").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "p":
            n = int(tokens[2])
        elif tokens[0] == "e":
            u, v = sorted(int(t) - 1 for t in tokens[1:3])
            if not 0 <= u < v < expected_n:
                raise ValueError(f"Invalid DIMACS edge: {line}")
            edges.add((u, v))
    if n != expected_n:
        raise ValueError(f"Expected {expected_n} vertices, found {n}")
    return sorted(edges)

def prepare(local_dir):
    manifest = json.loads((ROOT / "data/public_inputs.json").read_text(encoding="utf-8"))
    edges = {}
    for graph in manifest["graphs"]:
        if local_dir is None:
            with urlopen(graph["url"], timeout=30) as stream:
                blob = stream.read()
        else:
            blob = (local_dir / graph["name"]).read_bytes()
        if digest(blob) != graph["sha256"]:
            raise ValueError(f"Upstream file mismatch: {graph['name']}")
        edges[graph["name"]] = normalize(blob, graph["n"])
    for row in manifest["cases"]:
        normal = edges[row["graph"]]
        if digest(json.dumps(normal, separators=(",", ":")).encode()) != row["normalized_edge_sha256"]:
            raise ValueError(f"Normalized edge set differs for {row['file']}")
        path = ROOT / row["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(row["spec"], graph_edges=normal), indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(edges)} upstream graphs and generated {len(manifest['cases'])} cases.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, help="Directory containing the original CMU .col files.")
    prepare(parser.parse_args().local_dir)