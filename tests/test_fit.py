import numpy as np
import pytest

from cc_compact_tuning import fit

TRUE = np.array([2, 0.03, 6]) / 8e6  # budget share per token: write group, read, output


def windows(rng, n, sonnet=1.5, fable=None):
    """Windows that each used exactly one budget, with Fable in every other one if `fable`."""
    ws = []
    for i in range(n):
        X = np.zeros((3, 5))
        X[0] = [5e3, 0, rng.uniform(1e6, 3e6), rng.uniform(2e7, 6e7), rng.uniform(2e5, 6e5)]
        X[1] = [0, 0, rng.uniform(0, 3e5), rng.uniform(0, 5e6), rng.uniform(0, 1e5)]
        if fable and i % 2:
            X[2] = [0, 0, rng.uniform(2e5, 6e5), rng.uniform(5e6, 2e7), rng.uniform(5e4, 2e5)]
        w = dict(X=X, fable=0.3 if X[2].any() else 0.0)
        w["X"] = X / (fit.design(w, sonnet, fable or 1) @ TRUE)
        ws.append(w)
    return ws


def test_fit_recovers_weights_that_explain_every_window():
    f = fit.fit(windows(np.random.default_rng(1), 12))

    _, read, out = fit.rel(f)
    assert read == pytest.approx(0.03, rel=1e-3)
    assert out == pytest.approx(6, rel=1e-3)
    assert f["sonnet"] == pytest.approx(1.5, rel=1e-3)
    assert 2 / f["write"] == pytest.approx(8e6, rel=1e-3)
    assert not f["free_fable"] and f["fable"] == 1


def test_fable_factor_is_fitted_once_windows_use_fable():
    f = fit.fit(windows(np.random.default_rng(2), 12, fable=2.3))

    assert f["free_fable"]
    assert f["fable"] == pytest.approx(2.3, rel=1e-3)
    assert fit.rel(f)[1] == pytest.approx(0.03, rel=1e-3)


def test_fixed_sonnet_is_its_api_price():
    ws = windows(np.random.default_rng(3), 12, sonnet=1.0)
    assert fit.fit(ws, free_sonnet=False)["rms"] == pytest.approx(0, abs=1e-6)
