"""Iteration 6: re-score operators with SAE-code metrics on test (nobos):
do predictions activate the right downstream SAE features (precision/recall/F1),
with the right magnitudes (code FVE)? Guards against Goodharting the paper's
active-tokens-only feature R^2."""
import json
import sys
import numpy as np
import torch
from lab import *
import exp3_nonlinear as e3


def load_corr(tag, L, k):
    kind = tag.split("_h")[0]
    h = int(tag.split("_h")[1].split("_")[0])
    tk = int(tag.split("_k")[1].split("_")[0]) if "_k" in tag else 64
    m = e3.Corr(D, h, kind, tk)
    m.load_state_dict(torch.load(f"results/corr_{tag}_L{L}_k{k}.pt"))
    info = json.load(open(f"results/exp3_{tag}_L{L}_k{k}.json"))
    rs = info.get("resid_scale")
    if rs is None:  # runs before resid_scale was saved: recompute on a train subsample
        lin = np.load(f"results/W_normaware_L{L}_k{k}.npz")
        Xt, _, _ = layer_tokens(L, "train", drop_bos=True)
        Yt, _, _ = layer_tokens(L + k, "train", drop_bos=True)
        Xt, Yt = Xt[::3].copy(), Yt[::3].copy()
        rs = float((Yt - (phi_norm(Xt) @ lin["W"] + lin["b"])).std())
    return m.eval(), rs


def main(L, k, tags):
    T = L + k
    sae = SAE(T)
    lin = np.load(f"results/W_normaware_L{L}_k{k}.npz")
    X, _, _ = layer_tokens(L, "test", drop_bos=True)
    Y, _, _ = layer_tokens(T, "test", drop_bos=True)
    Ylin = phi_norm(X) @ lin["W"] + lin["b"]
    Xh = e3.rmsn(X).astype(np.float32)
    out = {}
    for name, Yh in (("identity", X), ("linear", Ylin)):
        out[name] = {**residual_metrics(Y, Yh), **feature_metrics(Y, Yh, sae)[0], **code_metrics(Y, Yh, sae)}
        print(f"[L={L},k={k}] {name:28s} " + " ".join(f"{m}={out[name][m]:.4f}" for m in
              ("fve", "feat_r2_median", "code_f1", "code_precision", "code_recall", "code_fve", "pred_l0")), flush=True)
    for tag in tags:
        try:
            m, rs = load_corr(tag, L, k)
        except FileNotFoundError:
            continue
        with torch.no_grad():
            P = torch.cat([m(torch.from_numpy(Xh[i:i + 4096])) for i in range(0, len(Xh), 4096)]).numpy() * rs
        Yh = Ylin + P
        out[tag] = {**residual_metrics(Y, Yh), **feature_metrics(Y, Yh, sae)[0], **code_metrics(Y, Yh, sae)}
        print(f"[L={L},k={k}] {tag:28s} " + " ".join(f"{mm}={out[tag][mm]:.4f}" for mm in
              ("fve", "feat_r2_median", "code_f1", "code_precision", "code_recall", "code_fve", "pred_l0")), flush=True)
    save_result(f"exp6_codeeval_L{L}_k{k}", out)


if __name__ == "__main__":
    L, k = int(sys.argv[1]), int(sys.argv[2])
    main(L, k, sys.argv[3:])
