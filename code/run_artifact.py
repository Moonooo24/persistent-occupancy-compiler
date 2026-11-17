"""Run archived benchmark input/method combinations using the published compiler."""
import argparse
import json
from pathlib import Path
from subprocess import run
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "data/run_catalog.json").read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=CATALOG, required=True)
    parser.add_argument("--case", help="Run only this case ID from the chosen group.")
    parser.add_argument("--method", help="Run one method (for example ours_all).")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cases = [r for r in CATALOG[args.group] if args.case is None or r["id"] == args.case]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    if not cases:
        parser.error("No matching cases")
    missing = [r["file"] for r in cases if not (ROOT / r["file"]).exists()]
    if missing:
        parser.error(f"Input missing: {missing[0]}. Prepare the public inputs first.")
    routed = args.group == "routing"
    script = "code/routing_v2.py" if routed else "code/experiments_v2.py"
    for case in cases:
        for method in ([args.method] if args.method else case["methods"]):
            argv = [sys.executable, "-B", "-X", "utf8", script, "--case", case["file"], "--method", method]
            if not routed:
                argv += ["--out", f"results/reproduction/{args.group}"]
            run(argv, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()