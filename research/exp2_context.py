"""Iteration 2 (diagnostic): how much of what a same-token linear ATO misses
is *cross-token* information written by attention?

Context features at the *source* layer only (still an upstream->downstream
map, but with a richer j-policy, cf. the paper's future work):
  x_t, x_hat_t, x_{t-1}, causal mean_{1<=s<=t} x_s
Closed-form ridge with identity prior on the x_t block, alpha chosen on val.
Memory-lean: one split resident at a time, features built per chunk."""
import gc
import sys
import numpy as np
import torch
from lab import *
from exp1_linear import REL


def ctx_arrays(X, pos, bos_x):
    P = np.empty_like(X)
    P[1:] = X[:-1]
    P[pos == 1] = bos_x  # token before the first real token is BOS
    C = np.empty_like(X)
    starts = np.nonzero(pos == 1)[0]
    ends = np.append(starts[1:], len(X))
    for a, b in zip(starts, ends):
        cs = np.cumsum(X[a:b], axis=0, dtype=np.float64)
        C[a:b] = (cs / np.arange(1, b - a + 1)[:, None]).astype(np.float32)
    return P, C


def feats(X, P, C, sl, parts):
    x = X[sl]
    out = []
    if "x" in parts: out.append(x)
    if "xhat" in parts: out.append(x / (np.sqrt((x ** 2).mean(1, keepdims=True)) + 1e-6))
    if "prev" in parts: out.append(P[sl])
    if "cmean" in parts: out.append(C[sl])
    return np.concatenate(out, 1)


def stats_for(L, T, sp, parts_list, bos_x):
    X, _, pos = layer_tokens(L, sp, drop_bos=True)
    Y, _, _ = layer_tokens(T, sp, drop_bos=True)
    P, C = ctx_arrays(X, pos, bos_x)
    sts = {}
    for parts in parts_list:
        n, ch = len(X), 8192
        mx = sum(feats(X, P, C, slice(i, i + ch), parts).sum(0, dtype=np.float64)
                 for i in range(0, n, ch)) / n
        my = Y.mean(0, dtype=np.float64)
        dx = mx.shape[0]
        Cxx = torch.zeros(dx, dx, dtype=torch.float64)
        Cxy = torch.zeros(dx, D, dtype=torch.float64)
        cyy = torch.zeros(D, dtype=torch.float64)
        mxt, myt = torch.from_numpy(mx.astype(np.float32)), torch.from_numpy(my.astype(np.float32))
        for i in range(0, n, ch):
            xc = torch.from_numpy(feats(X, P, C, slice(i, i + ch), parts)) - mxt
            yc = torch.from_numpy(Y[i:i + ch]) - myt
            Cxx += (xc.T @ xc).double(); Cxy += (xc.T @ yc).double()
            cyy += (yc * yc).sum(0).double()
        sts["+".join(parts)] = Stats(n, mx, my, Cxx.numpy(), Cxy.numpy(), cyy.numpy())
    return sts


def run(L, k, parts_list):
    T = L + k
    bos_x = bos_vector(L).astype(np.float32)
    st_tr = stats_for(L, T, "train", parts_list, bos_x); gc.collect()
    st_va = stats_for(L, T, "val", parts_list, bos_x); gc.collect()
    sae = SAE(T)
    X, _, pos = layer_tokens(L, "test", drop_bos=True)
    Y, _, _ = layer_tokens(T, "test", drop_bos=True)
    P, C = ctx_arrays(X, pos, bos_x)
    res = {}
    for parts in parts_list:
        name = "+".join(parts)
        tr, va = st_tr[name], st_va[name]
        eig = Eig(tr)
        scale = np.trace(tr.Cxx) / tr.Cxx.shape[0]
        Pr = np.zeros((tr.Cxx.shape[0], D)); Pr[:D] = np.eye(D)
        best = None
        for c in REL:
            W = eig.solve(c * scale, Pr); b = tr.my - tr.mx @ W
            _, ss = per_dim_r2_from_stats(W, b, va)
            fve = 1 - ss.sum() / va.cyy.sum()
            if best is None or fve > best[0]:
                best = (fve, c, W, b)
        fve, c, W, b = best
        Yh = np.concatenate([feats(X, P, C, slice(i, i + 8192), parts) @ W.astype(np.float32)
                             for i in range(0, len(X), 8192)]) + b.astype(np.float32)
        r = residual_metrics(Y, Yh)
        f, _ = feature_metrics(Y, Yh, sae)
        res[name] = {"nobos": {**r, **f}, "rel_alpha": c, "val_fve": float(fve)}
        print(f"[L={L},k={k}] {name:22s} c={c} val_fve={fve:.4f} | {fmt(res[name])}", flush=True)
        del eig, W
        gc.collect()
    return res


if __name__ == "__main__":
    pairs = [(9, 1), (15, 5)]
    if len(sys.argv) > 1:
        pairs = [tuple(map(int, p.split(","))) for p in sys.argv[1:]]
    out = {}
    for L, k in pairs:
        out[f"L{L}_k{k}"] = run(L, k, [("x", "xhat"), ("x", "xhat", "prev"),
                                       ("x", "xhat", "prev", "cmean")])
        save_result("exp2_context", out)
