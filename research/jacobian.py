"""Average Jacobian of a learned transport operator (a data-driven "J-lens").

Operator (row-vector convention, y = x W):
    f(x) = x A + x_hat B + b + rs * sum_{u in S(x)} relu(x_hat . enc_u + b_u) dec_u
    x_hat = x / rms(x),   rms(x) = ||x|| / sqrt(d),   S(x) = top-k active units.

d x_hat = d x P / rms with P = I - x_hat x_hat^T / d (symmetric), so

    J(x) = A + (P / rms) (B + rs * sum_{u in S(x)} enc_u (x) dec_u)

and the average over tokens is

    J_bar = A + Q B + rs * G^T Dec,
    Q     = E[P / rms]                               (d x d)
    G[u]  = E[1{u in S} P enc_u / rms]               (h x d)

This is the operator-level analogue of the Jacobian lens (the average
linearised effect of an upstream activation on a downstream one), computed
from our fitted map instead of the network. Under Gaussian inputs it would
coincide with OLS (Stein's lemma).
"""
import numpy as np
import torch


@torch.no_grad()
def average_jacobian(X: np.ndarray, A: np.ndarray, B: np.ndarray, corr=None, rs: float = 1.0,
                     chunk: int = 256):
    d = X.shape[1]
    Xt = torch.from_numpy(X)
    n = Xt.shape[0]
    inv_rms_sum = 0.0
    S = torch.zeros(d, d, dtype=torch.float64)          # sum x_hat x_hat^T / rms
    G = None
    if corr is not None:
        enc_w = corr.enc.weight                          # [h, d]
        G = torch.zeros_like(enc_w, dtype=torch.float64)
    for i in range(0, n, chunk):
        x = Xt[i:i + chunk]
        rms = x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        xh = x / rms
        inv_rms_sum += float((1 / rms).sum())
        S += ((xh / rms).T @ xh).double()
        if corr is not None:
            _, idx = corr.codes(xh) if hasattr(corr, "codes") else _codes(corr, xh)
            E = enc_w[idx]                               # [c, k, d]
            proj = (E * xh[:, None, :]).sum(-1, keepdim=True) / d   # x_hat . enc / d
            g = (E - proj * xh[:, None, :]) / rms[:, :, None]       # P enc / rms
            G.index_add_(0, idx.reshape(-1), g.reshape(-1, d).double())
    Q = torch.eye(d, dtype=torch.float64) * (inv_rms_sum / n) - S / (d * n)
    J_lin = torch.from_numpy(A).double() + Q @ torch.from_numpy(B).double()
    if corr is None:
        return J_lin.float().numpy(), None
    dec = corr.dec_w.double()                            # [h, d]
    J_full = J_lin + rs * (G / n).T @ dec
    return J_lin.float().numpy(), J_full.float().numpy()


def _codes(corr, xh):
    v, i = torch.relu(corr.enc(xh)).topk(corr.topk, dim=1)
    return v, i
