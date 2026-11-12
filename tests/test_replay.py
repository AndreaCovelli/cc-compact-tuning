from conftest import request

from cc_compact_tuning import replay

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
