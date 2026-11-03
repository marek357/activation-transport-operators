"""Iteration 1b: does alpha selected by validation *feature* R^2 (instead of
validation FVE) help the feature metric?

Hypothesis: ridge shrinkage pulls predictions toward the unconditional mean;
the feature metric is computed only on tokens where a feature fires (where
the true projection is far from that mean), so it prefers less shrinkage
than FVE does. Norm-aware features + identity prior (best of iteration 1)."""
import sys
import numpy as np
from lab import *

CANDS = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def main(pairs):
    out = {}
    for L, k in pairs:
        T = L + k
        sae = SAE(T)
        Xtr, _, _ = layer_tokens(L, "train", drop_bos=True)
        Ytr, _, _ = layer_tokens(T, "train", drop_bos=True)
        st = compute_stats_phi(Xtr, Ytr, phi_norm)
        del Xtr, Ytr
        Xva, _, _ = layer_tokens(L, "val", drop_bos=True)
        Yva, _, _ = layer_tokens(T, "val", drop_bos=True)
        Fva = phi_norm(Xva); del Xva
        eig = Eig(st)
        scale = np.trace(st.Cxx) / st.Cxx.shape[0]
        P = np.zeros((2 * D, D)); P[:D] = np.eye(D)
        curve, sols = {}, {}
        for c in CANDS:
            W = eig.solve(c * scale, P); b = st.my - st.mx @ W
            Yh = Fva @ W.astype(np.float32) + b.astype(np.float32)
            f, _ = feature_metrics(Yva, Yh, sae)
            r = residual_metrics(Yva, Yh)
            curve[c] = {"val_fmed": f["feat_r2_median"], "val_f>0.9": f["feat_frac_r2>0.9"],
                        "val_fve": r["fve"]}
            sols[c] = (W.astype(np.float32), b.astype(np.float32))
            print(f"[L={L},k={k}] c={c:g} val fve={r['fve']:.4f} fmed={f['feat_r2_median']:.4f} "
                  f"f>.9={f['feat_frac_r2>0.9']:.4f}", flush=True)
        del Fva, Yva
        c_feat = max(curve, key=lambda c: curve[c]["val_fmed"])
        c_fve = max(curve, key=lambda c: curve[c]["val_fve"])
        Xte, _, pte = layer_tokens(L, "test")
        Yte, _, _ = layer_tokens(T, "test")
        bos_y = bos_vector(T).astype(np.float32)
        res = {"curve": curve}
        for tag, c in (("normaware_I_featsel", c_feat), ("normaware_I_fvesel", c_fve)):
            W, b = sols[c]
            Yh = phi_norm(Xte) @ W + b
            Yh[pte == 0] = bos_y
            res[tag] = evaluate(Yte, Yh, pte, sae)
            res[tag]["rel_alpha"] = c
            print(f"[L={L},k={k}] {tag}(c={c:g}) {fmt(res[tag])}", flush=True)
            np.savez(f"results/W_{tag}_L{L}_k{k}.npz", W=W, b=b)
        out[f"L{L}_k{k}"] = res
        save_result("exp1b_featsel", out)


if __name__ == "__main__":
    pairs = [(9, 1), (5, 5), (1, 9), (19, 1), (15, 5), (11, 9)]
    if len(sys.argv) > 1:
        pairs = [tuple(map(int, p.split(","))) for p in sys.argv[1:]]
    main(pairs)
