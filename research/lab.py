"""Shared research harness for hill-climbing activation transport operators.

Everything here mirrors the repo's protocol (src/activation_loader.py +
eval.py) so numbers are comparable:
  * 60/20/20 split by *sequence index* (partition_loader),
  * same-token policy j == i, all attention-masked positions,
  * residual R^2 = mean over dims of per-dim R^2 (eval.score_residual_default),
  * feature R^2 = R^2 of the projection onto the unit-norm SAE decoder
    direction, restricted to tokens where the downstream JumpReLU SAE fires
    (eval.compute_all_feature_metrics with filter_inactive_features=True).

Additionally we report variants that drop the BOS position, because the BOS
residual is a constant vector (std ~1e-2, norm ~1600 vs ~130 for other
tokens) and dominates per-dimension variance.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

DATA = os.environ.get("ATO_DATA", "/home/user/ato_data")
D = 2304
torch.set_num_threads(os.cpu_count() or 4)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def split_indices(n_seq: int = 1136):
    tr = int(n_seq * 0.6)
    va = int(n_seq * 0.2)
    return {"train": np.arange(0, tr), "val": np.arange(tr, tr + va),
            "test": np.arange(tr + va, n_seq)}


_mask_cache = {}


def _mask():
    if "am" not in _mask_cache:
        _mask_cache["am"] = np.load(os.path.join(DATA, "attention_mask.npy")).astype(bool)
        _mask_cache["ids"] = np.load(os.path.join(DATA, "input_ids.npy"))
    return _mask_cache["am"], _mask_cache["ids"]


def layer_tokens(layer: int, split: str, drop_bos: bool = False):
    """Return float32 [n_tok, D] activations of `layer` for a split, plus
    (seq_idx, pos) for every row. Row order is identical across layers."""
    am, _ = _mask()
    seqs = split_indices(am.shape[0])[split]
    mm = np.load(os.path.join(DATA, f"layer_{layer}.f16"), mmap_mode="r")
    m = am[seqs].copy()
    if drop_bos:
        m[:, 0] = False
    s_loc, pos = np.nonzero(m)
    out = np.empty((len(s_loc), D), np.float32)
    # gather per sequence (contiguous reads)
    o = 0
    for i, s in enumerate(seqs):
        ps = np.nonzero(m[i])[0]
        if len(ps) == 0:
            continue
        blk = np.asarray(mm[s, : ps[-1] + 1])
        out[o:o + len(ps)] = blk[ps]
        o += len(ps)
    if not drop_bos:  # restore exact fp32 BOS rows
        bos = np.load(os.path.join(DATA, f"layer_{layer}_bos.npy"))
        rows = np.nonzero(pos == 0)[0]
        out[rows] = bos[seqs[s_loc[rows]]]
    return out, seqs[s_loc], pos


def token_ids(split: str, drop_bos: bool = False):
    am, ids = _mask()
    seqs = split_indices(am.shape[0])[split]
    m = am[seqs].copy()
    if drop_bos:
        m[:, 0] = False
    return ids[seqs][m]


def bos_vector(layer: int) -> np.ndarray:
    return np.load(os.path.join(DATA, f"layer_{layer}_bos.npy")).mean(0)


# --------------------------------------------------------------------------
# sufficient statistics + closed-form linear solvers
# --------------------------------------------------------------------------
@dataclass
class Stats:
    n: float
    mx: np.ndarray        # [dx]
    my: np.ndarray        # [dy]
    Cxx: np.ndarray       # centered sums  [dx, dx]
    Cxy: np.ndarray       # [dx, dy]
    cyy: np.ndarray       # diag of centered Y^T Y [dy]
    Cyy: np.ndarray | None = None

    def minus(self, o: "Stats") -> "Stats":
        """Stats of (self \\ o) for o a subset (used for exact K-fold CV)."""
        n = self.n - o.n
        sx = self.mx * self.n - o.mx * o.n
        sy = self.my * self.n - o.my * o.n
        Sxx = _raw(self.Cxx, self.mx, self.mx, self.n) - _raw(o.Cxx, o.mx, o.mx, o.n)
        Sxy = _raw(self.Cxy, self.mx, self.my, self.n) - _raw(o.Cxy, o.mx, o.my, o.n)
        syy = (self.cyy + self.n * self.my ** 2) - (o.cyy + o.n * o.my ** 2)
        mx, my = sx / n, sy / n
        return Stats(n, mx, my, Sxx - n * np.outer(mx, mx), Sxy - n * np.outer(mx, my),
                     syy - n * my ** 2)


def _raw(C, ma, mb, n):
    return C + n * np.outer(ma, mb)


def phi_norm(X: np.ndarray) -> np.ndarray:
    """[x, x_hat] with x_hat = x / rms(x) (what RMSNorm feeds every sublayer)."""
    r = np.sqrt((X.astype(np.float32) ** 2).mean(1, keepdims=True)) + 1e-6
    return np.concatenate([X, X / r], axis=1)


def compute_stats_phi(X, Y, phi, chunk: int = 16384) -> Stats:
    """Like compute_stats but with features phi(X) built chunk by chunk."""
    n = X.shape[0]
    mx = sum(phi(X[i:i + chunk]).sum(0, dtype=np.float64)
             for i in range(0, n, chunk)) / n
    my = Y.mean(0, dtype=np.float64)
    dx = mx.shape[0]
    Cxx = torch.zeros(dx, dx, dtype=torch.float64)
    Cxy = torch.zeros(dx, Y.shape[1], dtype=torch.float64)
    cyy = torch.zeros(Y.shape[1], dtype=torch.float64)
    mxt = torch.from_numpy(mx.astype(np.float32)); myt = torch.from_numpy(my.astype(np.float32))
    for i in range(0, n, chunk):
        xc = torch.from_numpy(phi(X[i:i + chunk])) - mxt
        yc = torch.from_numpy(Y[i:i + chunk]) - myt
        Cxx += (xc.T @ xc).double(); Cxy += (xc.T @ yc).double(); cyy += (yc * yc).sum(0).double()
    return Stats(n, mx, my, Cxx.numpy(), Cxy.numpy(), cyy.numpy())


def compute_stats(X: np.ndarray, Y: np.ndarray, full_yy: bool = False,
                  chunk: int = 16384, w: np.ndarray | None = None) -> Stats:
    """Centered second moments in float64 (fp32 matmuls on centered chunks)."""
    if w is None:
        n = X.shape[0]
        mx = X.mean(0, dtype=np.float64)
        my = Y.mean(0, dtype=np.float64)
    else:
        n = float(w.sum())
        mx = (w[:, None] * X).sum(0, dtype=np.float64) / n
        my = (w[:, None] * Y).sum(0, dtype=np.float64) / n
    dx, dy = X.shape[1], Y.shape[1]
    Cxx = torch.zeros(dx, dx, dtype=torch.float64)
    Cxy = torch.zeros(dx, dy, dtype=torch.float64)
    cyy = torch.zeros(dy, dtype=torch.float64)
    Cyy = torch.zeros(dy, dy, dtype=torch.float64) if full_yy else None
    mxt = torch.from_numpy(mx.astype(np.float32))
    myt = torch.from_numpy(my.astype(np.float32))
    for i in range(0, X.shape[0], chunk):
        xc = torch.from_numpy(X[i:i + chunk]) - mxt
        yc = torch.from_numpy(Y[i:i + chunk]) - myt
        if w is not None:
            sw = torch.from_numpy(np.sqrt(w[i:i + chunk]).astype(np.float32))[:, None]
            xc, yc = xc * sw, yc * sw
        Cxx += (xc.T @ xc).double()
        Cxy += (xc.T @ yc).double()
        cyy += (yc * yc).sum(0).double()
        if full_yy:
            Cyy += (yc.T @ yc).double()
    return Stats(n, mx, my, Cxx.numpy(), Cxy.numpy(), cyy.numpy(),
                 None if Cyy is None else Cyy.numpy())


class Eig:
    """Cache eigendecomposition of Cxx for fast multi-alpha ridge."""

    def __init__(self, st: Stats):
        w, V = np.linalg.eigh(st.Cxx)
        self.w, self.V = np.clip(w, 0, None), V
        self.VtCxy = V.T @ st.Cxy

    def solve(self, alpha: float, prior: np.ndarray | None = None,
              st: Stats | None = None) -> np.ndarray:
        """argmin ||Yc - Xc W||^2 + alpha ||W - prior||^2."""
        if prior is None:
            return self.V @ (self.VtCxy / (self.w + alpha)[:, None])
        rhs = self.VtCxy + alpha * (self.V.T @ prior)
        return self.V @ (rhs / (self.w + alpha)[:, None])


def ridge(st: Stats, alpha: float, prior=None, eig: Eig | None = None):
    eig = eig or Eig(st)
    W = eig.solve(alpha, prior)
    b = st.my - st.mx @ W
    return W, b


def per_dim_r2_from_stats(W, b, st: Stats):
    """Per-output R^2 of (W,b) on a dataset summarised by st (exact)."""
    # residual e = (y - my) - (x - mx) W + (my - mx W - b)
    off = st.my - st.mx @ W - b
    ss = (st.cyy - 2 * np.einsum("ij,ij->j", W, st.Cxy)
          + np.einsum("ij,ij->j", W, st.Cxx @ W) + st.n * off ** 2)
    return 1 - ss / np.maximum(st.cyy, 1e-12), ss


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
class SAE:
    def __init__(self, layer: int):
        p = np.load(os.path.join(DATA, "sae", f"layer_{layer}.npz"))
        self.W_enc = torch.from_numpy(p["W_enc"])
        self.b_enc = torch.from_numpy(p["b_enc"])
        self.thr = torch.from_numpy(p["threshold"])
        self.b_dec = torch.from_numpy(p["b_dec"])
        Wd = torch.from_numpy(p["W_dec"])
        self.W_dec = Wd
        self.Dn = Wd / (Wd.norm(dim=1, keepdim=True) + 1e-12)  # unit decoder rows

    def active(self, Y: torch.Tensor) -> torch.Tensor:
        pre = Y @ self.W_enc + self.b_enc
        return (pre > self.thr) & (pre > 1e-6)

    def encode(self, Y: torch.Tensor) -> torch.Tensor:
        pre = Y @ self.W_enc + self.b_enc
        return pre * (pre > self.thr)


def residual_metrics(Y: np.ndarray, Yh: np.ndarray):
    yt = torch.from_numpy(Y).double()
    yp = torch.from_numpy(Yh).double()
    d = yt - yp
    ss_res = (d ** 2).sum(0)
    ss_tot = ((yt - yt.mean(0)) ** 2).sum(0)
    r2_dim = 1 - ss_res / (ss_tot + 1e-12)
    cos = torch.nn.functional.cosine_similarity(yt, yp, dim=1)
    return {
        "r2_res": float(r2_dim.mean()),               # repo definition
        "fve": float(1 - ss_res.sum() / ss_tot.sum()),  # variance-weighted R^2
        "cos": float(cos.mean()),
        "rel_err": float((d.norm(dim=1) / yt.norm(dim=1).clamp_min(1e-9)).mean()),
    }


def feature_metrics(Y: np.ndarray, Yh: np.ndarray, sae: SAE, min_count: int = 10,
                    chunk_rows: int = 2048, feat_chunk: int = 4096):
    """Per-feature decoder-projection R^2 over tokens where the SAE fires on
    the true Y. Returns summary dict + per-feature arrays.
    (chunk_rows=2048 keeps the peak at ~1 GB: rows x 16k features x few temps)"""
    F = sae.Dn.shape[0]
    n = torch.zeros(F, dtype=torch.float64)
    s1 = torch.zeros(F, dtype=torch.float64)
    s2 = torch.zeros(F, dtype=torch.float64)
    sr = torch.zeros(F, dtype=torch.float64)
    for i in range(0, Y.shape[0], chunk_rows):
        yt = torch.from_numpy(Y[i:i + chunk_rows])
        yp = torch.from_numpy(Yh[i:i + chunk_rows].astype(np.float32))
        m = sae.active(yt).float()
        at = yt @ sae.Dn.T
        ap = yp @ sae.Dn.T
        ap.sub_(at).pow_(2).mul_(m)          # masked squared error
        n += m.sum(0).double()
        at.mul_(m)                           # m is 0/1 so (m*at)^2 == m*at^2
        s1 += at.sum(0).double()
        s2 += (at * at).sum(0).double()
        sr += ap.sum(0).double()
        del m, at, ap
    ok = n >= min_count
    sst = s2 - s1 ** 2 / n.clamp_min(1)
    r2 = torch.where(sst > 1e-12, 1 - sr / sst.clamp_min(1e-12), torch.zeros_like(sst))
    r2 = r2[ok].numpy()
    keep = r2 > -1  # paper's filter
    summ = {
        "n_feats": int(ok.sum()),
        "feat_r2_median": float(np.median(r2)),
        "feat_r2_mean_gt-1": float(r2[keep].mean()),
        "feat_frac_r2>0.9": float((r2 > 0.9).mean()),
        "feat_frac_r2>0.5": float((r2 > 0.5).mean()),
        # pooled "fraction of active-feature variance unexplained"
        "feat_fvu_pooled": float(sr[ok].sum() / sst[ok].sum()),
    }
    return summ, {"idx": np.nonzero(ok.numpy())[0], "r2": r2, "count": n[ok].numpy()}


def evaluate(Y, Yh, pos, sae: SAE, tag: str = ""):
    """Report metrics on (a) all positions [repo protocol] and (b) no BOS."""
    out = {}
    nb = pos > 0
    for name, sel in (("all", slice(None)), ("nobos", nb)):
        Ys, Yhs = Y[sel], Yh[sel]
        r = residual_metrics(Ys, Yhs)
        f, _ = feature_metrics(Ys, Yhs, sae)
        out[name] = {**r, **f}
    return out


def fmt(res: dict, keys=("r2_res", "fve", "feat_r2_median", "feat_r2_mean_gt-1",
                         "feat_frac_r2>0.9")):
    return " | ".join(
        f"{p}: " + " ".join(f"{k}={res[p][k]:.4f}" for k in keys)
        for p in ("all", "nobos") if p in res)


def save_result(name: str, payload: dict):
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "results", f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    return path


class Timer:
    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t


def code_metrics(Y: np.ndarray, Yh: np.ndarray, sae: SAE, chunk_rows: int = 2048):
    """Does the predicted vector light up the same downstream SAE features?
    Encode both y and y_hat with the target SAE and compare codes. Unlike the
    paper's feature R^2 (scored only where a feature truly fires), this also
    penalises features switched on where they should be off."""
    tp = fp = fn = 0
    ss_res = ss_tot = 0.0
    s1 = torch.zeros(sae.thr.shape[0], dtype=torch.float64); n = 0
    s2 = torch.zeros_like(s1)
    cos_sum = 0.0
    for i in range(0, Y.shape[0], chunk_rows):
        a = sae.encode(torch.from_numpy(Y[i:i + chunk_rows]))
        ah = sae.encode(torch.from_numpy(Yh[i:i + chunk_rows].astype(np.float32)))
        pa, ph = a > 0, ah > 0
        tp += int((pa & ph).sum()); fp += int((~pa & ph).sum()); fn += int((pa & ~ph).sum())
        ss_res += float(((a - ah) ** 2).sum())
        s1 += a.sum(0).double(); s2 += (a * a).sum(0).double(); n += a.shape[0]
        cos_sum += float(torch.nn.functional.cosine_similarity(a, ah, dim=1).nan_to_num(0).sum())
    ss_tot = float((s2 - s1 ** 2 / n).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"code_precision": prec, "code_recall": rec,
            "code_f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "code_fve": 1 - ss_res / ss_tot, "code_cos": cos_sum / n,
            "pred_l0": (tp + fp) / n, "true_l0": (tp + fn) / n}
