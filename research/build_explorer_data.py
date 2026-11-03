"""Build compact per-pair data for the feature-transport explorer.

For each downstream feature sampled from each class (transported /
assembled / synthesized / other), attach:
  * its best empirical upstream partner (oracle, train corr) and test corr,
  * top-3 upstream contributors through the ATO (cos(A_eff d_i^L, d_j^T)),
  * linear / nonlinear feature R^2,
  * Neuronpedia explanations for all features involved.
"""
import argparse
import json
import os
import numpy as np
import torch
from lab import SAE, D, save_result
from fetch_explanations import explain, save as save_expl

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--per_class", type=int, default=40)
    a = ap.parse_args()
    L, k = a.L, a.k
    T = L + k
    d = json.load(open(os.path.join(HERE, "results", f"exp5_features_L{L}_k{k}.json")))
    rows = d["rows"]
    lin = np.load(os.path.join(HERE, "results", f"W_normaware_L{L}_k{k}.npz"))
    W = lin["W"]
    A_eff = torch.from_numpy(W[:D] + W[D:] / d["rms_typ"])
    saeL, saeT = SAE(L), SAE(T)
    pushed = saeL.Dn @ A_eff
    pushed = pushed / pushed.norm(dim=1, keepdim=True).clamp_min(1e-9)

    rng = np.random.default_rng(0)
    picked = []
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["class"], []).append(r)
    for cls, rs in by_cls.items():
        if cls == "transported":  # most confident first, plus a random tail
            rs = sorted(rs, key=lambda r: -r["corr_oracle_test"])
            take = rs[: a.per_class // 2] + list(rng.choice(rs[a.per_class // 2:], size=min(
                a.per_class // 2, max(0, len(rs) - a.per_class // 2)), replace=False))
        elif cls == "synthesized":
            rs = sorted(rs, key=lambda r: -((r["r2_nl"] or 0) - r["r2_lin"]))
            take = rs[: a.per_class]
        else:
            take = list(rng.choice(rs, size=min(a.per_class, len(rs)), replace=False))
        picked += take
    out = []
    for r in picked:
        j = r["down"]
        sc = pushed @ saeT.Dn[j]
        top = torch.topk(sc, 3)
        contrib = [{"up": int(i), "cos": float(v)} for v, i in zip(top.values, top.indices)]
        e_down = explain(T, j)
        e_up = explain(L, r["up_oracle"])
        for c in contrib:
            c["expl"] = explain(L, c["up"]).get("explanation", "")
        out.append({**r, "down_expl": e_down.get("explanation", ""),
                    "down_logits": e_down.get("top_logits", []),
                    "up_oracle_expl": e_up.get("explanation", ""), "ato_top3": contrib})
        if len(out) % 20 == 0:
            save_expl(); print(f"{len(out)}/{len(picked)}", flush=True)
    save_expl()
    meta = {"L": L, "k": k, "T": T, "taxonomy": d["taxonomy"], "matching": d["matching"],
            "n_rows_total": len(rows)}
    path = os.path.join(HERE, "results_final", f"explorer_L{L}_k{k}.json")
    json.dump({"meta": meta, "rows": out}, open(path, "w"))
    print("wrote", path, len(out))


if __name__ == "__main__":
    main()
