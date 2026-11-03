"""Render research/report/feature_transport.html from the result JSONs.

python research/build_page.py   (re-run whenever results change)
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIN = os.path.join(HERE, "results_final")
OUT = os.path.join(HERE, "report", "feature_transport.html")
PAIRS = [(9, 1), (5, 5), (1, 9), (19, 1), (15, 5), (11, 9)]


def j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def first(*paths):
    for p in paths:
        d = j(p)
        if d is not None:
            return d
    return None


def scorecard():
    e0 = first(os.path.join(RES, "exp0_baseline.json"), os.path.join(FIN, "exp0_baseline.json")) or {}
    e1 = first(os.path.join(RES, "exp1_linear.json"), os.path.join(FIN, "exp1_linear.json")) or {}
    rows = {"Identity (y = x)": {}, "Repo ATO (ridge, CV α)": {},
            "Norm-aware linear ATO": {}, "+ sparse correction (top-k 8192)": {}}
    keys = ("fve", "feat_r2_median", "feat_frac_r2>0.9")
    ckeys = ("code_f1", "code_fve", "code_precision", "code_recall")
    for L, k in PAIRS:
        p = f"L{L}_k{k}"
        if p in e0:
            rows["Identity (y = x)"][p] = {m: e0[p]["identity"]["nobos"][m] for m in keys}
            rows["Repo ATO (ridge, CV α)"][p] = {m: e0[p]["repo_ridge"]["nobos"][m] for m in keys}
        if p in e1:
            rows["Norm-aware linear ATO"][p] = {m: e1[p]["normaware_I_nobos"]["nobos"][m] for m in keys}
        d = first(os.path.join(RES, f"exp3_topk_h8192_k64_L{L}_k{k}.json"),
                  os.path.join(FIN, f"exp3_topk_h8192_k64_L{L}_k{k}.json"))
        if d:
            rows["+ sparse correction (top-k 8192)"][p] = {m: d["nonlinear"][m] for m in keys}
        c = first(os.path.join(RES, f"exp6_codeeval_L{L}_k{k}.json"),
                  os.path.join(FIN, f"exp6_codeeval_L{L}_k{k}.json"))
        if c:
            for name, src in (("Identity (y = x)", "identity"), ("Norm-aware linear ATO", "linear"),
                              ("+ sparse correction (top-k 8192)", "topk_h8192_k64")):
                if src in c and p in rows[name]:
                    rows[name][p].update({m: c[src][m] for m in ckeys})
    return rows


def feat_loss():
    """5->10 comparison of training objectives (exp6)."""
    c = first(os.path.join(RES, "exp6_codeeval_L5_k5.json"), os.path.join(FIN, "exp6_codeeval_L5_k5.json"))
    if not c:
        return []
    names = [("linear", "Linear operator"), ("topk_h2048_k64", "+ top-k 2048, MSE loss"),
             ("topk_h8192_k64", "+ top-k 8192, MSE loss"),
             ("topk_h2048_k64_fl0.3", "+ top-k 2048, feature loss λ=0.3"),
             ("topk_h2048_k64_fl1", "+ top-k 2048, feature loss λ=1"),
             ("topk_h8192_k64_fl1", "+ top-k 8192, feature loss λ=1"),
             ("topk_h2048_k64_fl3", "+ top-k 2048, feature loss λ=3")]
    keys = ("fve", "feat_r2_median", "code_f1", "code_precision", "code_recall", "code_fve")
    return [{"name": n, **{m: c[t][m] for m in keys}} for t, n in names if t in c]


def features():
    out = {}
    for L, k in PAIRS:
        p = f"L{L}_k{k}"
        ex = j(os.path.join(FIN, f"explorer_L{L}_k{k}.json"))
        full = first(os.path.join(RES, f"exp5_features_L{L}_k{k}.json"))
        if ex is None and full is None:
            continue
        meta = (ex or {}).get("meta", {})
        if full is not None:  # newest run wins for summary numbers
            meta = {**meta, "taxonomy": full["taxonomy"], "matching": full["matching"],
                    "n_rows_total": len(full["rows"]), "L": L, "k": k, "T": L + k}
        rows = []
        for r in (ex or {}).get("rows", []):
            rows.append({
                "d": r["down"], "de": r.get("down_expl", ""), "u": r["up_oracle"],
                "ue": r.get("up_oracle_expl", ""), "c": round(r["corr_oracle_test"], 3),
                "rl": round(r["r2_lin"], 3), "rn": None if r.get("r2_nl") is None else round(r["r2_nl"], 3),
                "cls": r["class"], "n": r.get("down_count_test"),
                "top": [{"u": t["up"], "s": round(t["cos"], 3), "e": t.get("expl", "")}
                        for t in r.get("ato_top3", [])],
            })
        out[p] = {"meta": meta, "rows": rows}
    return out


def main():
    data = {"pairs": [f"L{L}_k{k}" for L, k in PAIRS], "score": scorecard(), "feat": features(),
            "floss": feat_loss()}
    tpl = open(os.path.join(HERE, "report", "template.html")).read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(html)
    print("wrote", OUT, f"{len(html) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
