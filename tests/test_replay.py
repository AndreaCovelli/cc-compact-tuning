import itertools
import re

import numpy as np
import pytest
from conftest import request

from cc_compact_tuning import replay, store

M = dict(g=2e3, post=40e3, postw=30e3, inp=3.5e3, cw=1e3, rf=0.97, out=6e3)


def growing_session(n=100, start=20e3):
    """One session whose context grows 2K per request, cache hits all the way."""
    return {"s": [request(60.0 * i, cr=start + 2e3 * i, cw1=2e3, out=500) for i in range(n)]}


def test_replay_compacts_when_the_context_crosses_the_trigger():
    reqs, done = replay.simulate(growing_session(), {}, 100e3, M)
    assert len(done) == 2
    assert max(cw + cr for _, cw, cr, _, _ in reqs) <= 100e3 + M["g"]


def test_never_compacting_matches_the_transcript():
    s = growing_session()
    reqs, done = replay.simulate(s, {}, 1e12, M)
    assert done == [] and [(cw, cr) for _, cw, cr, _, _ in reqs] == [(r["cw1"], r["cr"]) for r in s["s"]]


def test_real_compactions_are_replaced_by_continued_growth():
    s = {"s": [request(0, cr=150e3, cw1=2e3), request(120, cr=40e3, cw1=30e3)]}
    comps = {"s": [dict(t=60)]}
    reqs, done = replay.simulate(s, comps, 1e12, M)
    assert done == [] and reqs[1][1] + reqs[1][2] == 152e3 + M["g"]


def test_optimum_trigger_falls_as_cache_reads_get_dearer():
    cheap, _ = replay.optimum(M, [1, 1.25, 2, 0.01, 5])
    dear, _ = replay.optimum(M, [1, 1.25, 2, 0.1, 5])
    assert M["post"] < dear < cheap


HAIKU, FABLE = "claude-haiku-5-5", "claude-fable-5-2"


def test_haiku_prompts_above_100k_cost_five_times_as_much():
    small = replay.totals([(0, 0, 90e3, 0, 0)], [], M, HAIKU)
    big = replay.totals([(0, 0, 110e3, 0, 0)], [], M, HAIKU)
    assert small[2] == pytest.approx(90e3 * 0.05) and big[2] == pytest.approx(110e3 * 0.25)


def test_summary_requests_are_priced_by_the_context_they_read():
    t = replay.totals([], [150e3], M, HAIKU)  # reads 145.5K: the long rate
    assert t == pytest.approx([M["inp"] * 0.125, M["cw"] * 0.125, 0.97 * 150e3 * 0.25, M["out"] * 0.125])


def test_fable_compacts_later_than_opus_as_its_reads_are_relatively_cheaper():
    w = [1, 1.25, 2, 0.05, 5]
    opus, _ = replay.optimum(M, w)
    fable, _ = replay.optimum(M, w, FABLE)
    assert fable > opus


def compacting_store(model, n=3):
    """`n` sessions of `model` that grow to 150K, compact and go on, with matching cost-state."""
    S = store.empty()
    for k in range(n):
        sid, t0 = f"s{k}", replay.SWITCH + 3600 * k
        rs = [request(t0 + 60 * i, model, cr=50e3 + 2e3 * i, cw1=2e3, out=500) for i in range(51)]
        rs.append(request(t0 + 60 * 51, model, cr=10e3, cw1=35e3, out=500))
        for i, r in enumerate(rs):
            S["requests"][f"{sid}-{i}"] = r | dict(session=sid)
        pre = 152e3
        S["compactions"][sid] = dict(t=t0 + 60 * 50 + 30, session=sid, agent=None, trigger="auto", pre=pre)
        S["cost_states"][sid] = dict(
            modelUsage={
                model: dict(
                    inputTokens=sum(r["inp"] for r in rs) + 3500,
                    cacheCreationInputTokens=sum(r["cw1"] for r in rs) + 1000,
                    cacheReadInputTokens=sum(r["cr"] for r in rs) + 0.97 * pre,
                    outputTokens=sum(r["out"] for r in rs) + 6000,
                )
            }
        )
    return S


def test_a_model_with_enough_compactions_replays_its_own_sessions(monkeypatch):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    S = compacting_store(FABLE)
    source, m = replay.workload(S, FABLE)
    assert source == FABLE and m["n"] == 3 and m["inp"] == pytest.approx(3500)
    assert m["rf"] == pytest.approx(0.97) and m["post"] == 45e3


def test_a_model_without_compactions_falls_back_to_the_opus_sessions(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    S = compacting_store(replay.OPUS)
    assert replay.workload(S, HAIKU)[0] == replay.OPUS
    replay.report(S, tmp_path, model=HAIKU)
    out = capsys.readouterr().out
    assert "Opus sessions are replayed as if they ran on Haiku" in out and "x5 that above 100K" in out


def test_nothing_to_replay_without_compactions():
    with pytest.raises(SystemExit):
        replay.workload(store.empty(), FABLE)


def test_exact_optimum_formula_multiplies_rf_by_post():
    w = [1, 1.25, 2, 0.05, 5]
    T, K0 = replay.optimum(M, w)
    expected = M["post"] + (2 * M["g"] * (M["rf"] * M["post"] + K0 / 0.05)) ** 0.5
    assert T == pytest.approx(expected)


def test_simulate_clamps_negative_cf_and_shrinks_cold_cache_cw():
    # Simulate a large manual compaction / rewind making c + off < 0, plus a 2h gap triggering cold cache
    s = {
        "s": [
            request(0, cr=150e3, cw1=2e3),
            request(
                100, cr=10e3, cw1=35e3
            ),  # triggers counterfactual compaction at >100K -> off = post - 45K
            request(200, cr=5e3, cw1=1e3),  # c + off = 6K + (40K - 45K) = 1K < SYSTEM
            request(5000, cr=10e3, cw1=30e3),  # gap > 3600 & cw > 20e3: cold cache, cf must stay >= SYSTEM
        ]
    }
    reqs, _ = replay.simulate(s, {}, 100e3, M)
    for inp, cw, cr, _, _ in reqs:
        assert inp + cw + cr >= 0
        assert cw >= 0 and cr >= 0


def test_measure_includes_negative_non_compaction_deltas_in_g(monkeypatch):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    S = compacting_store(replay.OPUS, n=3)
    # Add a microcompaction drop of -10K inside s0 (no compaction boundary between idx 10 and 11)
    for i in range(11, 51):
        S["requests"][f"s0-{i}"]["cr"] -= 10_000
    S["cost_states"]["s0"]["modelUsage"][replay.OPUS]["cacheReadInputTokens"] -= 40 * 10_000
    m = replay.measure(S, replay.OPUS)
    assert m is not None and m["g"] < 2000


def test_discrete_pct_does_not_interpolate_never():
    vals = [180e3] * 85 + [1e12] * 15
    assert replay.pct(vals)[2] == 1e12


def test_report_uses_fitted_model_weights_when_configured(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    monkeypatch.setitem(replay.FITTED_MODEL_W, FABLE, [1.0, 1.9, 2.0, 0.04, 7.0])
    S = compacting_store(FABLE)
    replay.report(S, tmp_path, fitted=None, model=FABLE)
    out = capsys.readouterr().out
    assert "fitted weights are per Fable token from fit-log" in out
    assert "fitted weights per Fable token" in out


def test_weekly_loop_covers_more_than_three_weeks(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    S = compacting_store(replay.OPUS, n=3)
    # Add a session 4 weeks after SWITCH
    t_late = replay.SWITCH + 29 * 86400
    S["requests"]["late-0"] = request(t_late, cr=50e3, cw1=2e3, out=500) | dict(session="late")
    S["requests"]["late-1"] = request(t_late + 60, cr=52e3, cw1=2e3, out=500) | dict(session="late")
    replay.report(S, tmp_path, fitted=[1, 1.25, 2, 0.03, 5], model=replay.OPUS)
    out = capsys.readouterr().out
    assert out.count("sessions): best") >= 2


def test_shrink_pulls_sessions_with_few_compactions_toward_the_center():
    by = {"a": [60e3, 62e3, 58e3, 61e3], "b": [40e3, 41e3, 39e3, 40e3], "one": [80e3], "c": [50e3, 51e3]}
    s, k = replay.shrink(by, 50e3)
    assert 0 < k < 1
    assert abs(s["a"] - 60.5e3) < abs(s["one"] - 80e3)  # four compactions are trusted more than one
    assert 50e3 < s["one"] < 80e3 and 40e3 <= s["b"] < 41e3


def test_shrink_falls_back_to_the_center_without_between_session_variance():
    assert replay.shrink({"a": [40e3, 50e3], "b": [50e3, 40e3]}, 45e3) == ({}, np.inf)
    assert replay.shrink({"a": [40e3]}, 40e3) == ({}, np.inf)


def test_simulate_restarts_each_session_at_its_own_post_compaction_context():
    s = growing_session() | {"t": growing_session()["s"]}
    m = M | dict(post_s={"s": 70e3}, postw_s={"s": 90e3})  # a write above the context is capped by it
    reqs, _ = replay.simulate(s, {}, 100e3, m)

    def restarts(rr):  # (context, write) of each request right after a simulated compaction
        return [(b[1] + b[2], b[1]) for a, b in itertools.pairwise(rr) if b[1] + b[2] < a[1] + a[2]]

    own, other = restarts(reqs[:100]), restarts(reqs[100:])
    assert set(own) == {(70e3, 70e3)} and set(other) == {(M["post"], M["postw"])}
    assert len(own) > len(other)  # restarting higher, it reaches the trigger again sooner


def test_measure_skips_per_session_values_without_repeat_compactions(monkeypatch):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    S = compacting_store(replay.OPUS, n=4)
    for k, after in enumerate((20e3, 30e3, 60e3, 70e3)):
        S["requests"][f"s{k}-51"]["cr"] = after - 35e3 if after > 35e3 else 0
        S["requests"][f"s{k}-51"]["cw1"] = min(after, 35e3)
    m = replay.measure(S, replay.OPUS)
    assert m["post"] == pytest.approx(45e3)
    assert m["post_s"] == {} and m["post_k"] == np.inf  # one compaction per session: no within variance


SONNET = "claude-sonnet-5-5"


def test_a_fitted_factor_scales_the_compaction_cost_but_not_the_optimum():
    w = [1, 1.25, 2, 0.05, 5]
    T1, K1 = replay.optimum(M, w, SONNET)
    T2, K2 = replay.optimum(M, w, SONNET, factor=1.5)
    assert T2 == pytest.approx(T1) and K2 == pytest.approx(1.5 * K1)


def test_report_prices_sonnet_with_its_fitted_factor_in_the_fitted_scenario(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    monkeypatch.setattr(replay, "FITTED_W", list(replay.API_W))  # same weights in both scenarios
    monkeypatch.setitem(replay.FITTED_FACTOR, SONNET, 2.0)
    replay.report(compacting_store(replay.OPUS), tmp_path, model=SONNET)
    out = capsys.readouterr().out
    assert "fitted: x2.00 that (fit-log), 1 x Opus per input token" in out
    fit, api = (float(x) for x in re.findall(r"one compaction costs (\d+)K", out))
    assert fit == pytest.approx(2 * api, abs=1)
    fit_best, api_best = re.findall(r"vs best \((\w+)\)", out)[:: len(replay.SHOWN)]
    assert fit_best == api_best


def test_explicit_weights_do_not_apply_the_fitted_factor(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(replay, "MIN_COMPACTIONS", 3)
    replay.report(compacting_store(replay.OPUS), tmp_path, fitted=list(replay.API_W), model=SONNET)
    out = capsys.readouterr().out
    assert "fitted: x" not in out
    fit, api = (float(x) for x in re.findall(r"one compaction costs (\d+)K", out))
    assert fit == api
