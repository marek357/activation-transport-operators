"""Iteration 0: reproduce the repo's ATO (sklearn Ridge, fit_intercept, alpha by
5-fold contiguous CV with r2 scoring over the repo grid), trained on *all*
positions of the train split, evaluated on the test split. Also the identity
baseline (y_hat = x)."""
import sys
import numpy as np
from lab import *

GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 2000.0, 5000.0, 10000.0]  # repo grid


def kfold_stats(X, Y, k=5):
    n = X.shape[0]
    bounds = np.linspace(0, n, k + 1).astype(int)  # sklearn KFold(shuffle=False)
    return [compute_stats(X[a:b], Y[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]


def cv_alpha(full: Stats, folds, grid):
    scores = {a: [] for a in grid}
    for f in folds:
        tr = full.minus(f)
        eig = Eig(tr)
        for a in grid:
            W, b = ridge(tr, a, eig=eig)
            r2, _ = per_dim_r2_from_stats(W, b, f)
            scores[a].append(r2.mean())
    mean = {a: float(np.mean(v)) for a, v in scores.items()}
    return max(mean, key=mean.get), mean


def main(pairs):
    out = {}
    for L, k in pairs:
        T = L + k
        sae = SAE(T)
        Xtr, _, _ = layer_tokens(L, "train")
        Ytr, _, _ = layer_tokens(T, "train")
        Xte, _, pte = layer_tokens(L, "test")
        Yte, _, _ = layer_tokens(T, "test")
        res = {}
        res["identity"] = evaluate(Yte, Xte, pte, sae)
        print(f"[L={L},k={k}] identity  {fmt(res['identity'])}", flush=True)
        with Timer() as t:
            full = compute_stats(Xtr, Ytr)
            folds = kfold_stats(Xtr, Ytr)
            a_best, cv = cv_alpha(full, folds, GRID)
            W, b = ridge(full, a_best)
        Yh = Xte @ W.astype(np.float32) + b.astype(np.float32)
        res["repo_ridge"] = evaluate(Yte, Yh, pte, sae)
        res["repo_ridge"]["alpha"] = a_best
        res["repo_ridge"]["cv"] = cv
        print(f"[L={L},k={k}] repo_ridge(alpha={a_best}, {t.dt:.0f}s) {fmt(res['repo_ridge'])}",
              flush=True)
        out[f"L{L}_k{k}"] = res
        save_result("exp0_baseline", out)


if __name__ == "__main__":
    pairs = [(9, 1), (5, 5), (1, 9), (19, 1), (15, 5), (11, 9)]
    if len(sys.argv) > 1:
        pairs = [tuple(map(int, p.split(","))) for p in sys.argv[1:]]
    main(pairs)
