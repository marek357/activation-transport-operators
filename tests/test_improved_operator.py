"""Integration test: NormAwareTransportOperator on the repo's own data classes.

Uses a local copy of one (small) dataset part, e.g.
  ATO_TEST_ACTS=/path/with/activations_part_0007.zarr.zip  pytest tests/
and checks the streamed, chunked fit against a direct closed-form reference.
"""
import os
import numpy as np
import pytest

ACTS = os.environ.get("ATO_TEST_ACTS", "/home/user/ato_zarr_test/acts")
pytestmark = pytest.mark.skipif(not os.path.isdir(ACTS), reason="no local activations")


def _arrays(ds):
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in ds])
    return np.stack(xs), np.stack(ys)


def test_matches_closed_form_reference():
    from src.activation_loader import ActivationLoader, get_train_val_test_datasets
    from src.improved_transport_operator import NormAwareTransportOperator, _rms_normalize

    loader = ActivationLoader(ACTS)
    tr, va, te = get_train_val_test_datasets(5, 2, loader)
    op = NormAwareTransportOperator(5, 2, rel_alphas=(1e-3,), batch_size=500).fit(tr, va)

    X, Y = _arrays(tr)
    bos = np.zeros(len(X), bool)
    bos[0] = True
    cos = X @ X[0] / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[0]))
    bos |= cos > 0.95
    Xf, Yf = X[~bos].astype(np.float64), Y[~bos].astype(np.float64)
    F = np.concatenate([Xf, _rms_normalize(Xf.astype(np.float32)).astype(np.float64)], 1)
    mx, my = F.mean(0), Yf.mean(0)
    C = (F - mx).T @ (F - mx)
    a = 1e-3 * np.linalg.eigvalsh(C).clip(0).mean()
    P = np.zeros((F.shape[1], Y.shape[1])); P[: Y.shape[1]] = np.eye(Y.shape[1])
    Wr = np.linalg.solve(C + a * np.eye(len(C)), (F - mx).T @ (Yf - my) + a * P)
    br = my - mx @ Wr

    Xt, Yt = _arrays(te)
    pred = op.predict(Xt)
    Ft = np.concatenate([Xt, _rms_normalize(Xt)], 1)
    ref = Ft @ Wr + br
    tbos = op._is_bos(Xt)
    assert tbos.sum() == len(te.idx_list)  # exactly one <bos> per test sequence
    ref[tbos] = Y[bos].mean(0)
    rel = np.abs(pred - ref).max() / np.abs(ref).max()
    assert rel < 1e-3, rel


def test_save_load_roundtrip(tmp_path):
    from src.activation_loader import ActivationLoader, get_train_val_test_datasets
    from src.improved_transport_operator import NormAwareTransportOperator

    loader = ActivationLoader(ACTS)
    tr, va, te = get_train_val_test_datasets(5, 1, loader)
    op = NormAwareTransportOperator(5, 1).fit(tr, va)
    p = tmp_path / "op.pkl"
    op.save(str(p))
    op2 = NormAwareTransportOperator.load(str(p))
    X, Y = _arrays(te)
    pred = op.predict(X)
    np.testing.assert_allclose(pred, op2.predict(X))
    assert op.get_transport_matrix().shape == (2304, 2304)
    # sanity: with alpha chosen on val, the operator beats identity on held-out
    # non-BOS tokens even with this tiny (~2k token) training set
    m = ~op._is_bos(X)
    sst = ((Y[m] - Y[m].mean(0)) ** 2).sum()
    assert 1 - ((Y[m] - pred[m]) ** 2).sum() / sst > 1 - ((Y[m] - X[m]) ** 2).sum() / sst
