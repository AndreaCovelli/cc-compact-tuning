import numpy as np
import pytest
from conftest import request

from cc_compact_tuning import fit, replay, store

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


def test_fixed_out_ratio_pins_output_to_input_without_overflow():
    ws = windows(np.random.default_rng(4), 12)
    for ratio in (1.5, 5.0, 15.0):
        f = fit.fit(ws, fixed_out_ratio=ratio)
        _, _, out = fit.rel(f)
        assert out == pytest.approx(ratio, rel=1e-4)


def test_leverage_loo_and_date_step_check():
    ws = windows(np.random.default_rng(5), 14, fable=2.0)
    for i, w in enumerate(ws):
        w["hit"] = 1_000.0 * (i + 1)
    f = fit.fit(ws)
    lev = fit.leverage(f)
    assert len(lev) == len(ws)
    assert lev.sum() == pytest.approx(f["p"], abs=1e-2)
    loo_rms, preds = fit.loo(ws)
    assert loo_rms == pytest.approx(0.0, abs=1e-2)
    assert all(p == pytest.approx(1.0, rel=1e-2) for p in preds)
    ds = fit.date_step_check(ws)
    assert ds is not None and ds["both_rms"] <= ds["step_rms"] + 1e-6


def test_with_summaries_adds_measured_compaction_tokens():
    S = store.empty()
    for k in range(10):
        sid = f"s{k}"
        t0 = replay.SWITCH + 3600 * k
        rs = [request(t0 + 60 * i, cr=50e3 + 2e3 * i, cw1=2e3, out=500) for i in range(4)]
        rs.append(request(t0 + 60 * 5, cr=10e3, cw1=35e3, out=500))
        for i, r in enumerate(rs):
            S["requests"][f"{sid}-{i}"] = r | dict(session=sid)
        pre = 160e3
        S["compactions"][sid] = dict(t=t0 + 60 * 4 + 30, session=sid, agent=None, trigger="auto", pre=pre)
        S["cost_states"][sid] = dict(
            modelUsage={
                replay.OPUS: dict(
                    inputTokens=sum(r["inp"] for r in rs) + 3500,
                    cacheCreationInputTokens=sum(r["cw1"] for r in rs) + 1000,
                    cacheReadInputTokens=sum(r["cr"] for r in rs) + 0.97 * pre,
                    outputTokens=sum(r["out"] for r in rs) + 6000,
                )
            }
        )
    w = dict(
        X=np.zeros((3, 5)),
        fable=0.0,
        compaction_events=[
            dict(t=replay.SWITCH + 270, session="s0", agent=None, pre=160e3, model=replay.OPUS)
        ],
    )
    out_ws, ok = fit.with_summaries([w], S)
    assert ok
    assert out_ws[0]["X"][0] == pytest.approx([3500, 0, 1000, 0.97 * 160e3, 6000])
