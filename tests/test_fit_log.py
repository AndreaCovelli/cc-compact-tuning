import numpy as np
import pytest
from conftest import request

from cc_compact_tuning import fit_log
from cc_compact_tuning.pricing import CATS, model_factor


def test_segments_split_at_login_and_drop_single_lines():
    rows = [(1, 10, 100, "a"), (2, 12, 100, "a"), (3, 14, 100, "a"), (5, 1, 100, "a"), (6, 3, 200, "a")]
    segs = fit_log.segments(rows, logins=[4])
    assert segs == [(100, "a", [(1, 10), (2, 12), (3, 14)])]


def test_read_log_merges_saved_lines_and_skips_incomplete(tmp_path, empty_store):
    saved = "\t".join(["10", "s", "m", "40", "500", "10", "900"])
    empty_store["rate_limits"][saved] = 1
    log = tmp_path / "rate.tsv"
    log.write_text(
        "header\n" + "\t".join(["20", "s", "m", "42", "500", "10", "900"]) + "\n20\ts\tm\t\t\t\t\n"
    )
    assert fit_log.read_log(log, empty_store) == [(10.0, 40.0, 500, "900"), (20.0, 42.0, 500, "900")]


def test_fit_recovers_weights_from_percentage_rises(empty_store):
    rng = np.random.default_rng(2)
    v = np.array([2, 0.03, 6]) / 9e6
    R = []
    for i in range(400):
        model = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2", "claude-opus-5-5")[i % 4]
        r = request(60.0 * i, model, cw1=rng.uniform(0, 4e4))
        r |= dict(cr=rng.uniform(5e4, 1.5e5), out=rng.uniform(100, 3e3))
        R.append(r)
    empty_store["requests"] = {str(i): r for i, r in enumerate(R)}
    T, F = fit_log.features(empty_store)
    # Exact percentages after every request, in 4 windows of 100 requests.
    rows = []
    for k in range(4):
        pct = 0.0
        for r in R[100 * k : 100 * (k + 1)]:
            x = np.array([r[c] for c in CATS]) * model_factor(r)
            x *= {"claude-sonnet-5-5": 1.8, "claude-fable-5-2": 2.3}.get(r["model"], 1)
            pct += 100 * np.array([x @ fit_log.TIE, x[3], x[4]]) @ v
            rows.append((r["t"] + 1, pct, k, "acct"))

    f = fit_log.fit(fit_log.points(fit_log.segments(rows, []), T, F))

    assert f["read"] == pytest.approx(0.03, rel=1e-2)
    assert f["out"] == pytest.approx(6, rel=1e-2)
    assert f["sonnet"] == pytest.approx(1.8, rel=1e-2)
    assert f["fable"] == pytest.approx(2.3, rel=1e-2)
    assert f["budget"] == pytest.approx(9e6, rel=1e-2)
