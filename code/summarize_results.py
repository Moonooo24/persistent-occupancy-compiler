"""Reconstruct the paper's primary paired CX totals from archived measurements."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
E = json.loads((ROOT/"data/evaluation.json").read_text(encoding="utf-8"))
CANDIDATES = ("ours_none", "ours_all", "ours_selected")
PRIMARY = ("ref_t3_original","ref_t3_clique","ref_cpg_open","ref_cpg_closed",
           "ref_matched_exact_original","ref_matched_exact_clique")

def aggregate(rows, routed=False):
    total_ref = total_out = total_none = wins = ties = 0
    for row in rows:
        candidates=row["all_candidate_metrics"]
        references=row["all_method_metrics"] if routed else row["all_reference_metrics"]
        names=PRIMARY if routed else tuple(references)
        best=min(candidates, key=lambda n:(candidates[n]["cx"],candidates[n]["parameterized_rz"],
                                           candidates[n]["depth"],n))
        ref=min(references[n]["cx"] for n in names)
        got=candidates[best]["cx"]
        assert ref==row["reference_metric_envelope"]["cx"] and got==row["ours_metrics"]["cx"]
        total_ref+=ref
        total_out+=got
        total_none+=candidates["ours_none"]["cx"]
        wins+=got<ref
        ties+=got==ref
    return dict(programs=len(rows),reference_cx=total_ref,compiled_cx=total_out,
                reduction_pct=round(100*(1-total_out/total_ref),2), wins=wins,ties=ties,
                no_cache_cx=total_none,increment_over_no_cache_pct=round(100*(1-total_out/total_none),2))

def main():
    totals={}
    for name,cohort in E["cohorts"].items():
        programs=cohort["programs"]
        if name=="main144":
            programs=list({row["semantic_program_sha256"]:row for row in programs}.values())
        totals[name]=aggregate(programs)
    for topology in ("line","grid"):
        totals["routing_"+topology]=aggregate([x for x in E["routing_final"] if x["topology"]==topology],routed=True)
    expected={
        "main144":(134,611250,473016,22.61), "new_confirmation72":(72,442568,354650,19.87),
        "sweep120":(120,972132,825400,15.09), "public24":(24,205334,166620,18.85),
        "routing_line":(66,1960198,1781104,9.14), "routing_grid":(66,684282,598736,12.50),
    }
    for group,reference in expected.items():
        x=totals[group]
        assert (x["programs"],x["reference_cx"],x["compiled_cx"],x["reduction_pct"])==reference, (group,x)
    for group,pct in (("main144",4.10),("new_confirmation72",9.13)):
        assert totals[group]["increment_over_no_cache_pct"]==pct
    print(json.dumps(totals,indent=2))

if __name__=="__main__":
    main()