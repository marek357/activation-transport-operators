"""Iteration 1: linear hypotheses H1 (BOS lookup), H2 (identity prior),
H3 (norm-aware features [x, x/rms(x)]). alpha selected on the *val* split by
variance-weighted R^2 (FVE) on non-BOS tokens; final numbers on test."""
import sys
import numpy as np
from lab import *

REL = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def select(st_tr, st_va, prior_fn):
    eig = Eig(st_tr)
    scale = np.trace(st_tr.Cxx) / st_tr.Cxx.shape[0]
    best = None
    curve = {}
    for c in REL:
        a = c * scale
        prior = prior_fn()
        W = eig.solve(a, prior)
        b = st_tr.my - st_tr.mx @ W
        _, ss = per_dim_r2_from_stats(W, b, st_va)
        fve = 1 - ss.sum() / st_va.cyy.sum()
        curve[c] = float(fve)
        if best is None or fve > best[0]:
            best = (fve, c, W, b)
    return best, curve


def predict(X, pos, W, b, bos_y, phi=None):
    Xf = X if phi is None else phi(X)
    Yh = Xf @ W.astype(np.float32) + b.astype(np.float32)
    if bos_y is not None:
        Yh[pos == 0] = bos_y
    return Yh


def main(pairs):
    out = {}
    for L, k in pairs:
        T = L + k
        sae = SAE(T)
        Xtr, _, _ = layer_tokens(L, "train", drop_bos=True)
        Ytr, _, _ = layer_tokens(T, "train", drop_bos=True)
        Xva, _, _ = layer_tokens(L, "val", drop_bos=True)
        Yva, _, _ = layer_tokens(T, "val", drop_bos=True)
        Xte, _, pte = layer_tokens(L, "test")
        Yte, _, _ = layer_tokens(T, "test")
        bos_y = bos_vector(T).astype(np.float32)
        res = {}
        st_tr, st_va = compute_stats(Xtr, Ytr), compute_stats(Xva, Yva)
        I = np.eye(D)
        for name, pf in (("ridge0_nobos", lambda: None), ("ridgeI_nobos", lambda: I)):
            (fve, c, W, b), curve = select(st_tr, st_va, pf)
            res[name] = evaluate(Yte, predict(Xte, pte, W, b, bos_y), pte, sae)
            res[name].update(rel_alpha=c, val_curve=curve)
            print(f"[L={L},k={k}] {name}(c={c}) {fmt(res[name])}", flush=True)
        # H3: norm-aware
        st_tr2 = compute_stats_phi(Xtr, Ytr, phi_norm)
        st_va2 = compute_stats_phi(Xva, Yva, phi_norm)
        P = np.zeros((2 * D, D)); P[:D] = I
        (fve, c, W, b), curve = select(st_tr2, st_va2, lambda: P)
        name = "normaware_I_nobos"
        res[name] = evaluate(Yte, predict(Xte, pte, W, b, bos_y, phi_norm), pte, sae)
        res[name].update(rel_alpha=c, val_curve=curve)
        print(f"[L={L},k={k}] {name}(c={c}) {fmt(res[name])}", flush=True)
        np.savez(f"results/W_normaware_L{L}_k{k}.npz", W=W.astype(np.float32), b=b.astype(np.float32))
        out[f"L{L}_k{k}"] = res
        save_result("exp1_linear", out)


if __name__ == "__main__":
    pairs = [(9, 1), (5, 5), (1, 9), (19, 1), (15, 5), (11, 9)]
    if len(sys.argv) > 1:
        pairs = [tuple(map(int, p.split(","))) for p in sys.argv[1:]]
    main(pairs)
