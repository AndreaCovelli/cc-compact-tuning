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
