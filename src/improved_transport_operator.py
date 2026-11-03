"""Improved activation transport operators (see research/RESEARCH_LOG.md).

`NormAwareTransportOperator` is a drop-in replacement for
`TransportOperator` (same `fit(dataset)` / `predict(X)` /
`get_transport_matrix()` / `get_bias()` interface). It changes four things,
each validated on held-out data in research/:

1. **BOS special-casing.** In Gemma-2 the <bos> residual is a constant vector
   (norm ~10x a normal token). The repo's ridge fit treats its ~700 identical
   copies as extreme-leverage points. We drop BOS rows from the fit, detect
   them at predict time (cosine > 0.95 to the stored BOS activation; ordinary
   tokens reach <= 0.79), and output the stored downstream BOS activation.
2. **Identity prior.** Ridge shrinks toward W = I (the residual stream
   carries x forward), not toward 0.
3. **Norm-aware features.** Gemma-2 sublayers read RMSNorm(x) and write
   through a post-norm, so the update y - x depends (almost) only on
   x_hat = x / rms(x). We regress on [x, x_hat]:
       y_hat = A x + B x_hat + b.
4. **Alpha chosen on the validation split** from a relative grid (alpha =
   c * mean eigenvalue of the train covariance) using exact closed-form
   validation FVE, instead of 5-fold CV over a fixed absolute grid.

An optional nonlinear "synthesis" correction (research/exp3_nonlinear.py)
can be attached with `attach_correction`.
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REL_ALPHAS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _rms_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.sqrt((X.astype(np.float32) ** 2).mean(1, keepdims=True)) + 1e-6)


class _Moments:
    """Streaming centred second moments in float64."""

    def __init__(self, dx: int, dy: int):
        self.n = 0
        self.sx = np.zeros(dx)
        self.sy = np.zeros(dy)
        self.Sxx = torch.zeros(dx, dx, dtype=torch.float64)
        self.Sxy = torch.zeros(dx, dy, dtype=torch.float64)
        self.syy = np.zeros(dy)
        self.shift_x = None
        self.shift_y = None

    def add(self, F: np.ndarray, Y: np.ndarray):
        if self.shift_x is None:  # shift by first batch mean for stability
            self.shift_x = F.mean(0)
            self.shift_y = Y.mean(0)
        f = torch.from_numpy((F - self.shift_x).astype(np.float32))
        y = torch.from_numpy((Y - self.shift_y).astype(np.float32))
        self.n += F.shape[0]
        self.sx += f.sum(0).double().numpy()
        self.sy += y.sum(0).double().numpy()
        self.Sxx += (f.T @ f).double()
        self.Sxy += (f.T @ y).double()
        self.syy += (y * y).sum(0).double().numpy()

    def finalize(self):
        n = self.n
        mx_s, my_s = self.sx / n, self.sy / n
        Cxx = self.Sxx.numpy() - n * np.outer(mx_s, mx_s)
        Cxy = self.Sxy.numpy() - n * np.outer(mx_s, my_s)
        cyy = self.syy - n * my_s ** 2
        return n, mx_s + self.shift_x, my_s + self.shift_y, Cxx, Cxy, cyy


class NormAwareTransportOperator:
    def __init__(
        self,
        L: int,
        k: int,
        norm_aware: bool = True,
        identity_prior: bool = True,
        special_case_bos: bool = True,
        bos_cos_threshold: float = 0.95,
        rel_alphas=REL_ALPHAS,
        batch_size: int = 4096,
    ):
        self.L, self.k = L, k
        self.norm_aware = norm_aware
        self.identity_prior = identity_prior
        self.special_case_bos = special_case_bos
        self.bos_cos_threshold = bos_cos_threshold
        self.rel_alphas = tuple(rel_alphas)
        self.batch_size = batch_size
        self.W = None  # [d_feat, d_out]
        self.b = None
        self.bos_x = None
        self.bos_y = None
        self.alpha_rel_ = None
        self.val_curve_ = None
        self.correction = None
        self.is_fitted_ = False
        self.method = "normaware_identity_ridge"

    # ------------------------------------------------------------------
    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.norm_aware:
            return np.concatenate([X, _rms_normalize(X)], axis=1)
        return X

    def _is_bos(self, X: np.ndarray) -> np.ndarray:
        if not self.special_case_bos or self.bos_x is None:
            return np.zeros(X.shape[0], dtype=bool)
        cos = (X @ self.bos_x) / (np.linalg.norm(X, axis=1) * np.linalg.norm(self.bos_x) + 1e-9)
        return cos > self.bos_cos_threshold

    def _stream_moments(self, dataset, find_bos: bool):
        mom = None
        bos_x, bos_y, n_bos = None, None, 0
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0)
        first = True
        rms_samples = []
        for xb, yb in loader:
            X = xb.numpy().astype(np.float32)
            Y = yb.numpy().astype(np.float32)
            if first and find_bos and self.special_case_bos and self.bos_x is None:
                # datasets iterate sequences from position 0, i.e. row 0 is <bos>
                self.bos_x = X[0].copy()
            first = False
            is_bos = self._is_bos(X)
            if is_bos.any():
                bos_y = Y[is_bos].sum(0) + (0 if bos_y is None else bos_y)
                n_bos += int(is_bos.sum())
                X, Y = X[~is_bos], Y[~is_bos]
            if X.shape[0] == 0:
                continue
            if find_bos:
                rms_samples.append(np.sqrt((X[::16] ** 2).mean(1)))
            F = self._features(X)
            if mom is None:
                mom = _Moments(F.shape[1], Y.shape[1])
            mom.add(F, Y)
        if find_bos and n_bos:
            self.bos_y = (bos_y / n_bos).astype(np.float32)
        if find_bos and rms_samples:
            self.typical_rms_ = float(np.median(np.concatenate(rms_samples)))
        return mom.finalize()

    def fit(self, dataset, val_dataset=None) -> "NormAwareTransportOperator":
        t0 = time.time()
        n, mx, my, Cxx, Cxy, cyy = self._stream_moments(dataset, find_bos=True)
        d_in, d_out = Cxx.shape[0], Cxy.shape[1]
        prior = np.zeros((d_in, d_out))
        if self.identity_prior:
            prior[: d_out, :] = np.eye(d_out)
        w, V = np.linalg.eigh(Cxx)
        w = np.clip(w, 0, None)
        VtC, VtP = V.T @ Cxy, V.T @ prior
        scale = float(w.mean())

        def solve(c):
            a = c * scale
            W = V @ ((VtC + a * VtP) / (w + a)[:, None])
            return W, my - mx @ W

        if val_dataset is None or len(self.rel_alphas) == 1:
            c_best = self.rel_alphas[len(self.rel_alphas) // 2] if val_dataset is None \
                else self.rel_alphas[0]
            self.W, self.b = solve(c_best)
        else:
            vn, vmx, vmy, vCxx, vCxy, vcyy = self._stream_moments(val_dataset, find_bos=False)
            best, self.val_curve_ = None, {}
            for c in self.rel_alphas:
                W, b = solve(c)
                off = vmy - vmx @ W - b
                ss = (vcyy - 2 * np.einsum("ij,ij->j", W, vCxy)
                      + np.einsum("ij,ij->j", W, vCxx @ W) + vn * off ** 2)
                fve = 1 - ss.sum() / vcyy.sum()
                self.val_curve_[c] = float(fve)
                if best is None or fve > best[0]:
                    best = (fve, c, W, b)
            _, c_best, self.W, self.b = best
        self.alpha_rel_ = c_best
        self.W = self.W.astype(np.float32)
        self.b = self.b.astype(np.float32)
        self.is_fitted_ = True
        print(f"NormAwareTransportOperator L={self.L} k={self.k}: fitted on {n} tokens "
              f"(alpha_rel={c_best}) in {time.time() - t0:.1f}s")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("operator must be fitted first")
        X = np.asarray(X, dtype=np.float32)
        Y = self._features(X) @ self.W + self.b
        if self.correction is not None:
            with torch.no_grad():
                Xh = torch.from_numpy(_rms_normalize(X).astype(np.float32))
                Y = Y + np.concatenate([self.correction(Xh[i:i + 8192]).numpy()
                                        for i in range(0, len(Xh), 8192)]) * self.correction_scale
        bos = self._is_bos(X)
        if bos.any() and self.bos_y is not None:
            Y[bos] = self.bos_y
        return Y

    transform = predict

    def attach_correction(self, module: torch.nn.Module, scale: float = 1.0):
        """Attach a nonlinear correction r(x_hat) (trained on this operator's residual)."""
        self.correction = module.eval()
        self.correction_scale = scale
        return self

    def get_transport_matrix(self) -> np.ndarray:
        """Effective linear map [d_in, d_out] at a typical token.

        The operator is A x + B x_hat; around a token with rms(x) = r the x_hat
        block contributes approximately B / r. We return A + B / r_typ with
        r_typ the median train rms, so rank/efficiency analyses still apply.
        """
        d = self.W.shape[1]
        if not self.norm_aware:
            return self.W
        return self.W[:d] + self.W[d:] / getattr(self, "typical_rms_", 1.0)

    def get_bias(self) -> np.ndarray:
        return self.b

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "NormAwareTransportOperator":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj


# ----------------------------------------------------------------------------
# Optional nonlinear "synthesis" correction
# ----------------------------------------------------------------------------
class SparseCorrection(torch.nn.Module):
    """r(x_hat) = W_dec topk(ReLU(W_enc x_hat + b_enc)) + b_dec.

    A transcoder-like sparse code over the RMS-normalised upstream vector,
    trained on the residual of the linear operator. In research/ this beat
    dense GELU / GeGLU MLPs at matched size; k=64 follows Shabalin &
    Piotrowski (arXiv:2610.24521)."""

    def __init__(self, d: int = 2304, h: int = 8192, k: int = 64):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(d, h)
        self.dec_w = torch.nn.Parameter(torch.randn(h, d) * (1e-3 / k ** 0.5))
        self.dec_b = torch.nn.Parameter(torch.zeros(d))

    def codes(self, xh: torch.Tensor):
        v, i = torch.relu(self.enc(xh)).topk(self.k, dim=1)
        return v, i

    def forward(self, xh: torch.Tensor) -> torch.Tensor:
        v, i = self.codes(xh)
        return torch.nn.functional.embedding_bag(i, self.dec_w, per_sample_weights=v,
                                                 mode="sum") + self.dec_b


def _materialize(op: NormAwareTransportOperator, dataset):
    """(x_hat, residual) arrays over non-BOS rows of a dataset."""
    xs, rs = [], []
    for xb, yb in DataLoader(dataset, batch_size=op.batch_size, num_workers=0):
        X, Y = xb.numpy().astype(np.float32), yb.numpy().astype(np.float32)
        keep = ~op._is_bos(X)
        X, Y = X[keep], Y[keep]
        xs.append(_rms_normalize(X).astype(np.float32))
        rs.append(Y - (op._features(X) @ op.W + op.b))
    return np.concatenate(xs), np.concatenate(rs)


def fit_sparse_correction(op: NormAwareTransportOperator, train_dataset, val_dataset,
                          h: int = 8192, k: int = 64, epochs: int = 3, lr: float = 1e-3,
                          batch_size: int = 1024, seed: int = 0) -> NormAwareTransportOperator:
    """Train a SparseCorrection on the residual of a fitted linear operator,
    keep the epoch with the best validation FVE, and attach it to `op`."""
    torch.manual_seed(seed)
    Xtr, Rtr = _materialize(op, train_dataset)
    Xva, Rva = _materialize(op, val_dataset)
    scale = float(Rtr.std())
    model = SparseCorrection(Xtr.shape[1], h, k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = max(1, epochs * (len(Xtr) // batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    Xt, Rt = torch.from_numpy(Xtr), torch.from_numpy(Rtr / scale)
    Xv = torch.from_numpy(Xva)
    best, best_state = np.inf, None
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        model.train()
        for s in range(len(Xt) // batch_size):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            loss = ((model(Xt[idx]) - Rt[idx]) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            P = torch.cat([model(Xv[i:i + 8192]) for i in range(0, len(Xv), 8192)]).numpy() * scale
        err = float(((Rva - P) ** 2).sum())
        print(f"  correction epoch {ep}: val residual SSE {err:.4g} (linear only {float((Rva ** 2).sum()):.4g})")
        if err < best:
            best, best_state = err, {kk: vv.clone() for kk, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    return op.attach_correction(model, scale)
