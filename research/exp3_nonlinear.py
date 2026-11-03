"""Iteration 3: transport + synthesis.

y_hat = [A B] [x; x_hat] + b                      (closed-form linear ATO, frozen)
      + W_dec sigma(W_enc x_hat + b_enc) + b_dec   (learned nonlinear correction)

The nonlinear correction is trained on the residual of the linear ATO, so
the linear part keeps its meaning ("what is linearly transported") and the
correction captures what the intermediate MLPs/attention *synthesise*.
sigma variants follow the neuron/transcoder literature:
  gelu   : dense GELU MLP
  geglu  : gated GELU (Gemma's own MLP form)
  topk   : ReLU + per-token top-k (transcoder-style sparse code; Shabalin &
           Piotrowski 2026 recommend k ~ 64 for neuron-like units)
"""
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
from lab import *


class Corr(nn.Module):
    """kind in {gelu, geglu, topk, topk_ss}. topk_ss = sign-split top-k: each
    encoder unit has two poles (relu(u), relu(-u)) with separate decoders,
    the 'neuron pole' view of Cammarata / Shabalin & Piotrowski."""

    def __init__(self, d, h, kind="gelu", topk=64):
        super().__init__()
        self.kind, self.topk = kind, topk
        self.enc = nn.Linear(d, h)
        if kind == "geglu":
            self.gate = nn.Linear(d, h)
        n_dec = 2 * h if kind == "topk_ss" else h
        if kind.startswith("topk"):
            # sparse decoder: rows of an embedding table, summed with weights
            self.dec_w = nn.Parameter(torch.randn(n_dec, d) * (1e-3 / math.sqrt(topk)))
            self.dec_b = nn.Parameter(torch.zeros(d))
        else:
            self.dec = nn.Linear(h, d)
            nn.init.normal_(self.dec.weight, std=1e-3 / math.sqrt(h))
            nn.init.zeros_(self.dec.bias)

    def forward(self, xh):
        u = self.enc(xh)
        if self.kind == "gelu":
            return self.dec(nn.functional.gelu(u))
        if self.kind == "geglu":
            return self.dec(nn.functional.gelu(self.gate(xh)) * u)
        if self.kind == "topk_ss":
            u = torch.cat([u, -u], 1)
        u = torch.relu(u)
        v, i = u.topk(self.topk, dim=1)
        return nn.functional.embedding_bag(i, self.dec_w, per_sample_weights=v,
                                           mode="sum") + self.dec_b


def rmsn(X):
    return X / (np.sqrt((X ** 2).mean(1, keepdims=True)) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=9)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--kind", default="gelu")
    ap.add_argument("--h", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--feat_lambda", type=float, default=0.0,
                    help="weight of the SAE-decoder-space loss on active features (0 = plain MSE)")
    a = ap.parse_args()
    torch.manual_seed(0)
    L, k = a.L, a.k
    T = L + k
    lin = np.load(f"results/W_normaware_L{L}_k{k}.npz")
    W, b = lin["W"], lin["b"]

    def prep(sp):
        X, _, p = layer_tokens(L, sp, drop_bos=True)
        Y, _, _ = layer_tokens(T, sp, drop_bos=True)
        Ylin = phi_norm(X) @ W + b
        R = (Y - Ylin).astype(np.float32)
        Xh = rmsn(X).astype(np.float32)
        del X
        return Xh, R, Y, Ylin, p

    Xtr, Rtr, Ytr, _, _ = prep("train")
    if a.max_train:
        Xtr, Rtr, Ytr = Xtr[: a.max_train], Rtr[: a.max_train], Ytr[: a.max_train]
    sae = SAE(T)
    if a.feat_lambda > 0:
        # Feature-aware loss: per token, squared error of the projections onto the
        # unit decoder directions of the SAE features active on the *true* y,
        # each normalised by that feature's active-token variance (so the loss is
        # a sum of per-feature (1 - R^2)-like terms, as in the paper's metric).
        F = sae.Dn.shape[0]
        packed = np.zeros((len(Ytr), F // 8), np.uint8)
        cnt = torch.zeros(F, dtype=torch.float64); s1 = torch.zeros(F, dtype=torch.float64)
        s2 = torch.zeros(F, dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, len(Ytr), 2048):
                yb = torch.from_numpy(Ytr[i:i + 2048])
                act = sae.active(yb)
                packed[i:i + 2048] = np.packbits(act.numpy(), axis=1)
                at = (yb @ sae.Dn.T) * act
                cnt += act.sum(0).double(); s1 += at.sum(0).double(); s2 += (at * at).sum(0).double()
        var_f = (s2 / cnt.clamp_min(1) - (s1 / cnt.clamp_min(1)) ** 2).clamp_min(1e-6)
        rs_ = float(Rtr.std())
        inv_v = torch.where(cnt >= 10, rs_ ** 2 / var_f, torch.zeros_like(var_f)).float()
        avg_l0 = float(cnt.sum() / len(Ytr))
        print(f"feature loss: lambda={a.feat_lambda}, avg L0={avg_l0:.1f}", flush=True)
    del Ytr
    Xva, Rva, Yva, Ylva, _ = prep("val")
    rs = float(Rtr.std())  # scale residual targets to ~unit variance
    sst_va = float(((Yva - Yva.mean(0)) ** 2).sum())
    model = Corr(D, a.h, a.kind, a.topk)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n = Xtr.shape[0]
    steps = a.epochs * (n // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.05)
    Xtr_t, Rtr_t = torch.from_numpy(Xtr), torch.from_numpy(Rtr / rs)
    Xva_t = torch.from_numpy(Xva)

    def val_fve():
        model.eval()
        with torch.no_grad():
            P = torch.cat([model(Xva_t[i:i + 8192]) for i in range(0, len(Xva_t), 8192)]) * rs
        model.train()
        e = Rva - P.numpy()
        return 1 - float((e ** 2).sum()) / sst_va

    base_va = 1 - float((Rva ** 2).sum()) / sst_va
    print(f"[L={L},k={k}] {a.kind} h={a.h} linear-only val FVE={base_va:.4f}", flush=True)
    best, best_state = -1e9, None
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(n // a.bs):
            idx = perm[s * a.bs:(s + 1) * a.bs]
            e = model(Xtr_t[idx]) - Rtr_t[idx]
            if a.feat_lambda > 0:
                m = torch.from_numpy(np.unpackbits(packed[idx.numpy()], axis=1)).float()
                p = e @ sae.Dn.T
                feat_term = (p * p * m * inv_v).sum(1).mean() / avg_l0
                loss = (e * e).sum(1).mean() / D + a.feat_lambda * feat_term
            else:
                loss = (e * e).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        v = val_fve()
        msg = f"  ep {ep} train_loss={tot / (n // a.bs):.4f} val FVE={v:.4f}"
        if a.feat_lambda > 0:  # select on the metric we target
            model.eval()
            with torch.no_grad():
                Pv = torch.cat([model(Xva_t[i:i + 8192]) for i in range(0, len(Xva_t), 8192)]).numpy() * rs
            model.train()
            fv = feature_metrics(Yva, Ylva + Pv, sae)[0]["feat_r2_median"]
            msg += f" val fmed={fv:.4f}"
            v = fv
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
        if v > best:
            best, best_state = v, {kk: vv.clone() for kk, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    del Xtr, Rtr, Xtr_t, Rtr_t
    # test
    Xte, Rte, Yte, Ylte, pte = prep("test")
    model.eval()
    with torch.no_grad():
        Pte = torch.cat([model(torch.from_numpy(Xte[i:i + 8192])) for i in range(0, len(Xte), 8192)]).numpy() * rs
    Yh = Ylte + Pte
    r_lin = {**residual_metrics(Yte, Ylte), **feature_metrics(Yte, Ylte, sae)[0]}
    r_nl = {**residual_metrics(Yte, Yh), **feature_metrics(Yte, Yh, sae)[0]}
    tag = f"{a.kind}_h{a.h}" + (f"_k{a.topk}" if a.kind == "topk" else "") \
        + (f"_fl{a.feat_lambda:g}" if a.feat_lambda > 0 else "")
    print(f"[L={L},k={k}] linear   {fmt({'nobos': r_lin})}")
    print(f"[L={L},k={k}] {tag} {fmt({'nobos': r_nl})}")
    save_result(f"exp3_{tag}_L{L}_k{k}", {"args": vars(a), "val_best_fve": best, "resid_scale": rs,
                                          "linear": r_lin, "nonlinear": r_nl})
    torch.save(model.state_dict(), f"results/corr_{tag}_L{L}_k{k}.pt")


if __name__ == "__main__":
    main()
