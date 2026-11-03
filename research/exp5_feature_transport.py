"""Iteration 5: which SAE features move between layers?

For a pair (L -> T=L+k) with Gemma Scope SAEs at both ends:

1. Codes a^L(t) = SAE_L(x_t), a^T(t) = SAE_T(y_t) on train and test tokens.
2. Empirical co-activation: Pearson corr(a_i^L, a_j^T) for all 16k x 16k pairs
   (sparse accumulation). The train-split argmax is the *oracle* upstream
   partner of each downstream feature; test corr is used for scoring.
3. Matching rules for "which upstream feature becomes downstream feature j":
     dec-cos : argmax_i cos(d_i^L, d_j^T)            (identity-map matching,
                                                        as in prior work)
     ato-dec : argmax_i cos(A_eff d_i^L, d_j^T)        (push through the ATO)
     ato-enc : argmax_i  e_j^T A_eff d_i^L * E[a_i^L | a_i^L > 0]
                                       (predicted contribution to j's pre-act)
   where A_eff = A + B / rms_typ linearises the norm-aware ATO y = A x + B x_hat + b.
   Scored by the test correlation of the matched pair and agreement with
   the oracle.
4. Per-downstream-feature taxonomy:
     transported : a single upstream feature explains it (test corr >= 0.8)
                   and the linear ATO predicts it (feature R^2 >= 0.8)
     assembled   : linear ATO predicts it (R^2 >= 0.8), no single upstream
                   partner does (corr < 0.8)
     synthesized : linear ATO fails (R^2 < 0.5) but the nonlinear correction
                   lifts R^2 by >= 0.2
     other       : everything else
"""
import argparse
import json
import numpy as np
import torch
from lab import *

torch.set_grad_enabled(False)


def codes_sparse(X, sae, chunk=2048):
    """Return list of (rows, cols, vals) per chunk as torch sparse CSR blocks."""
    out = []
    for i in range(0, len(X), chunk):
        a = sae.encode(torch.from_numpy(X[i:i + chunk]))
        out.append(a.to_sparse_csr())
    return out


def comoments(CL, CT, F=16384):
    S = torch.zeros(F, F)  # float32 accumulator of sum_t a_i^L a_j^T
    sL = torch.zeros(F, dtype=torch.float64); qL = torch.zeros(F, dtype=torch.float64)
    sT = torch.zeros(F, dtype=torch.float64); qT = torch.zeros(F, dtype=torch.float64)
    nL = torch.zeros(F); nT = torch.zeros(F)
    n = 0
    for aL, aT in zip(CL, CT):
        dT = aT.to_dense()
        dL = aL.to_dense()
        S.addmm_(dL.T.to_sparse_csr(), dT)  # sparse(F_L x n) @ dense(n x F_T)
        sL += dL.sum(0).double(); qL += (dL * dL).sum(0).double(); nL += (dL > 0).sum(0)
        sT += dT.sum(0).double(); qT += (dT * dT).sum(0).double(); nT += (dT > 0).sum(0)
        n += dT.shape[0]
    return S, sL, qL, sT, qT, nL, nT, n


def corr_matrix(S, sL, qL, sT, qT, n):
    """In-place Pearson correlation (float32, ~1 GB per 16k x 16k matrix)."""
    mL, mT = (sL / n).float(), (sT / n).float()
    sdL = (qL / n - (sL / n) ** 2).clamp_min(1e-12).sqrt().float()
    sdT = (qT / n - (sT / n) ** 2).clamp_min(1e-12).sqrt().float()
    S.div_(n).sub_(torch.outer(mL, mT)).div_(torch.outer(sdL, sdT))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=9)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--corr", default="", help="nonlinear correction tag, e.g. gelu_h2048")
    ap.add_argument("--min_count", type=int, default=20)
    a = ap.parse_args()
    L, k = a.L, a.k
    T = L + k
    saeL, saeT = SAE(L), SAE(T)
    lin = np.load(f"results/W_normaware_L{L}_k{k}.npz")
    W, b = lin["W"], lin["b"]
    A, B = W[:D], W[D:]

    # --- codes + co-moments (train for oracle, test for scoring)
    stats = {}
    for sp in ("train", "test"):
        X, _, _ = layer_tokens(L, sp, drop_bos=True)
        Y, _, _ = layer_tokens(T, sp, drop_bos=True)
        if sp == "train":
            rms_typ = float(np.median(np.sqrt((X ** 2).mean(1))))
            X, Y = X[::2], Y[::2]  # 75k tokens is plenty for co-activation stats
        CL, CT = codes_sparse(X, saeL), codes_sparse(Y, saeT)
        stats[sp] = comoments(CL, CT)
        if sp == "test":
            Xte, Yte = X, Y
        del CL, CT
        print(f"co-moments {sp} done", flush=True)

    S, sL, qL, sT, qT, nL_tr, nT_tr, n = stats.pop("train")
    act_mean_L = (sL / nL_tr.double().clamp_min(1)).float()  # E[a_i^L | active], train
    C_tr = corr_matrix(S, sL, qL, sT, qT, n)
    S, sL, qL, sT, qT, nL_te, nT_te, n_te = stats.pop("test")
    C_te = corr_matrix(S, sL, qL, sT, qT, n_te)
    del S

    # downstream features active often enough on both splits; live upstream features
    eligible = (nT_te >= a.min_count) & (nT_tr >= a.min_count)
    j_idx = torch.nonzero(eligible).squeeze(1)
    up_ok = (nL_tr >= a.min_count) & (nL_te >= 1)

    # --- matching rules
    DL = saeL.Dn  # unit upstream decoder rows [F, D]
    DT = saeT.Dn
    A_eff = torch.from_numpy(A + B / rms_typ)
    pushed = DL @ A_eff  # [F_L, D]  (row-vector convention: y = x A)
    pushed_n = pushed / pushed.norm(dim=1, keepdim=True).clamp_min(1e-9)
    big = torch.finfo(torch.float32).min

    def argmax_over_up(score):  # score [F_L, |j|]
        score = score.clone()
        score[~up_ok] = big
        return score.argmax(0)

    def csls(U):
        """Cross-domain similarity local scaling (Conneau et al. 2018): penalise
        'hub' features that are close to everything. U: unit rows [F_L, D]."""
        S = U @ DT.T                                   # [F_L, F_T] cosines
        rL = S.topk(10, dim=1).values.mean(1)          # hubness of upstream i
        rT = S.topk(10, dim=0).values.mean(0)          # hubness of downstream j
        out = 2 * S[:, j_idx] - rL[:, None] - rT[j_idx][None, :]
        del S
        return out

    res = {}
    oracle = argmax_over_up(C_tr[:, j_idx])
    best_te = C_te[oracle, j_idx]
    oracle_tr_corr = C_tr[oracle, j_idx]
    del C_tr
    rules = {
        "oracle(train-corr)": oracle,
        "dec-cos": argmax_over_up(DL @ DT[j_idx].T),
        "ato-dec": argmax_over_up(pushed_n @ DT[j_idx].T),
        "dec-cos-csls": argmax_over_up(csls(DL)),
        "ato-dec-csls": argmax_over_up(csls(pushed_n)),
        "ato-enc": argmax_over_up((pushed * (saeL.W_dec.norm(dim=1, keepdim=True)
                                             * act_mean_L[:, None])) @ saeT.W_enc[:, j_idx]),
    }
    for name, m in rules.items():
        c = C_te[m, j_idx]
        res[name] = {
            "test_corr_mean": float(c.mean()), "test_corr_median": float(c.median()),
            "frac_corr>0.8": float((c > 0.8).float().mean()),
            "frac_corr>0.5": float((c > 0.5).float().mean()),
            "agree_with_oracle": float((m == oracle).float().mean()),
        }
        print(f"[L={L},k={k}] match {name:20s} " + " ".join(f"{kk}={v:.3f}" for kk, v in res[name].items()),
              flush=True)

    # --- per-downstream-feature R^2 for linear ATO (+ optional nonlinear)
    Yh_lin = phi_norm(Xte) @ W + b
    _, pf_lin = feature_metrics(Yte, Yh_lin, saeT, min_count=a.min_count)
    r2_lin = np.full(16384, np.nan); r2_lin[pf_lin["idx"]] = pf_lin["r2"]
    r2_nl = None
    if a.corr:
        import exp3_nonlinear as e3
        kind, h = a.corr.split("_h")[0], int(a.corr.split("_h")[1].split("_")[0])
        tk = int(a.corr.split("_k")[1]) if "_k" in a.corr else 64
        model = e3.Corr(D, h, kind, tk)
        model.load_state_dict(torch.load(f"results/corr_{a.corr}_L{L}_k{k}.pt"))
        model.eval()
        info = json.load(open(f"results/exp3_{a.corr}_L{L}_k{k}.json"))
        rs = info.get("resid_scale")
        if rs is None:  # older runs: recompute residual scale on a train subsample
            Xtr_, _, _ = layer_tokens(L, "train", drop_bos=True)
            Ytr_, _, _ = layer_tokens(T, "train", drop_bos=True)
            Xtr_, Ytr_ = Xtr_[::3].copy(), Ytr_[::3].copy()
            rs = float((Ytr_ - (phi_norm(Xtr_) @ W + b)).std()); del Xtr_, Ytr_
        Xh = e3.rmsn(Xte).astype(np.float32)
        P = torch.cat([model(torch.from_numpy(Xh[i:i + 4096])) for i in range(0, len(Xh), 4096)]).numpy() * rs
        _, pf_nl = feature_metrics(Yte, Yh_lin + P, saeT, min_count=a.min_count)
        r2_nl = np.full(16384, np.nan); r2_nl[pf_nl["idx"]] = pf_nl["r2"]

    # --- taxonomy over eligible downstream features
    jj = j_idx.numpy()
    corr_best = best_te.numpy()
    rl = r2_lin[jj]
    rn = r2_nl[jj] if r2_nl is not None else np.full_like(rl, np.nan)
    cls = np.full(len(jj), "other", dtype=object)
    cls[(rl >= 0.8) & (corr_best < 0.8)] = "assembled"
    cls[(rl >= 0.8) & (corr_best >= 0.8)] = "transported"
    if r2_nl is not None:
        cls[(rl < 0.5) & (rn - rl >= 0.2)] = "synthesized"
    counts = {c: int((cls == c).sum()) for c in ("transported", "assembled", "synthesized", "other")}
    print(f"[L={L},k={k}] taxonomy over {len(jj)} downstream features: {counts}", flush=True)

    # --- table of feature pairs for inspection
    ato_m = rules["ato-dec-csls"].numpy()
    dec_m = rules["dec-cos"].numpy()
    rows = []
    for t, j in enumerate(jj):
        rows.append({
            "down": int(j), "up_oracle": int(oracle[t]), "up_ato": int(ato_m[t]), "up_deccos": int(dec_m[t]),
            "corr_oracle_test": float(corr_best[t]),
            "corr_ato_test": float(C_te[ato_m[t], j]), "corr_deccos_test": float(C_te[dec_m[t], j]),
            "r2_lin": float(rl[t]), "r2_nl": None if np.isnan(rn[t]) else float(rn[t]),
            "class": cls[t], "down_count_test": int(nT_te[j]),
        })
    save_result(f"exp5_features_L{L}_k{k}", {"matching": res, "taxonomy": counts, "rows": rows,
                                             "rms_typ": rms_typ})


if __name__ == "__main__":
    main()
