"""Collect results/*.json into markdown tables (test split)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = ["L9_k1", "L5_k5", "L1_k9", "L19_k1", "L15_k5", "L11_k9"]


def load(name):
    p = os.path.join(HERE, "results", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def collect():
    rows = {}  # method -> pair -> {"all":..., "nobos":...}
    for exp in ("exp0_baseline", "exp1_linear", "exp4_final"):
        for pair, d in load(exp).items():
            for m, r in d.items():
                rows.setdefault(m, {})[pair] = r
    for p in glob.glob(os.path.join(HERE, "results", "exp3_*.json")):
        d = json.load(open(p))
        a = d["args"]
        pair = f"L{a['L']}_k{a['k']}"
        tag = os.path.basename(p)[5:-5].rsplit("_L", 1)[0]
        rows.setdefault(f"nl:{tag}", {})[pair] = {"nobos": d["nonlinear"]}
    for pair, d in load("exp2_context").items():
        for m, r in d.items():
            rows.setdefault(f"ctx:{m}", {})[pair] = r
    return rows


def table(rows, metric, proto="nobos", methods=None):
    methods = methods or list(rows)
    hdr = "| method | " + " | ".join(p.replace("_", " ") for p in PAIRS) + " | mean |"
    out = [hdr, "|" + "---|" * (len(PAIRS) + 2)]
    for m in methods:
        vals = []
        for p in PAIRS:
            r = rows.get(m, {}).get(p, {}).get(proto)
            vals.append(None if r is None else r.get(metric))
        got = [v for v in vals if v is not None]
        mean = f"{sum(got) / len(got):.4f}" if len(got) == len(PAIRS) else ""
        out.append(f"| {m} | " + " | ".join("" if v is None else f"{v:.4f}" for v in vals)
                   + f" | {mean} |")
    return "\n".join(out)


if __name__ == "__main__":
    rows = collect()
    metrics = sys.argv[1:] or ["fve", "feat_r2_median", "feat_frac_r2>0.9"]
    for proto in ("nobos", "all"):
        for met in metrics:
            print(f"\n**{met} ({proto})**\n")
            print(table(rows, met, proto))
