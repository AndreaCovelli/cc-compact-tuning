"""Measure what a compaction costs and replay the main-thread Opus sessions since the switch to a
200K window at other trigger points.

The summary request that writes a compaction is not in the transcript; the session's cost-state
record counts it, so per-session cost-state minus transcript totals gives its tokens.
"""
import collections
import datetime as dt
import statistics as st
import sys

import numpy as np

from common import load

SWITCH = dt.datetime(2026, 10, 27, 6, 41, 1, tzinfo=dt.timezone.utc).timestamp()
OPUS = "claude-opus-5-5"
SYSTEM = 14.3e3          # system prompt + tools: what a cold resume after compaction still reads


def ctx(r):
    return r["inp"] + r["cw5"] + r["cw1"] + r["cr"]


def sessions(S, since=SWITCH):
    main, side = collections.defaultdict(list), []
    for r in S["requests"].values():
        if r["t"] < since or r["model"] != OPUS:
            continue
        (side.append(r) if r["side"] else main[r["session"]].append(r))
    comps = collections.defaultdict(list)
    for c in S["compactions"].values():
        if c["t"] >= since and c["trigger"] == "auto" and not c["agent"]:
            comps[c["session"]].append(c)
    return ({k: sorted(v, key=lambda r: r["t"]) for k, v in main.items()}, side,
            {k: sorted(v, key=lambda c: c["t"]) for k, v in comps.items()})


def measure(S):
    main, side, comps = sessions(S)
    allreq = collections.defaultdict(list)
    for r in S["requests"].values():
        if r["model"] == OPUS:
            allreq[r["session"]].append(r)
    early = {c["session"] for c in S["compactions"].values() if c["t"] < SWITCH}
    rows = []
    for sid, cs in comps.items():
        st_ = S["cost_states"].get(sid)
        mu = st_ and st_["modelUsage"].get(OPUS)
        if not mu or sid in early:
            continue
        rs = allreq[sid]
        tr = [sum(r["inp"] for r in rs), sum(r["cw5"] + r["cw1"] for r in rs),
              sum(r["cr"] for r in rs), sum(r["out"] for r in rs)]
        cv = [mu.get(k, 0) for k in ("inputTokens", "cacheCreationInputTokens",
                                      "cacheReadInputTokens", "outputTokens")]
        d = [a - b for a, b in zip(cv, tr)]
        pre = sum(c["pre"] for c in cs)
        if min(d) < -1000 or not 0.85 <= d[2] / pre <= 1.1:
            continue        # resumed or restarted sessions, whose cost-state does not cover the file
        rows.append((len(cs), pre, d))
    n = sum(r[0] for r in rows)
    pre = sum(r[1] for r in rows)
    tot = [sum(r[2][i] for r in rows) for i in range(4)]
    m = dict(sessions=len(rows), n=n, inp=tot[0] / n, cw=tot[1] / n, rf=tot[2] / pre, out=tot[3] / n,
             out_med=st.median(r[2][3] / r[0] for r in rows))
    # First main-thread request after each compaction: what is left, and what it writes.
    post, postw, g = [], [], []
    for sid, rs in main.items():
        bts = [c["t"] for c in comps.get(sid, [])]
        for b in bts:
            nxt = next((r for r in rs if r["t"] > b), None)
            if nxt and nxt["t"] - b < 3600:
                post.append(ctx(nxt))
                postw.append(nxt["cw5"] + nxt["cw1"])
        c = [ctx(r) for r in rs]
        g += [y - x for x, y in zip(c, c[1:]) if 0 <= y - x < 200e3 and x > 40e3]
    m |= dict(post=st.median(post), postw=st.median(postw), g=st.mean(g), g_med=st.median(g),
              ncomp_post=len(post))
    return m


def simulate(main, comps, trigger, M):
    """Replay each session with compaction at `trigger` instead of where it happened."""
    reqs, done = [], []
    for sid, rs in main.items():
        bts = [c["t"] for c in comps.get(sid, [])]
        off, prev, prevcf = 0.0, None, None
        for r in rs:
            c = ctx(r)
            cw = r["cw5"] + r["cw1"]
            if prev is not None and any(prev["t"] < b < r["t"] for b in bts):
                off = prevcf + M["g"] - c          # the real compaction does not happen here
                cw = min(cw, M["g"])
            cf = c + off
            if cf > trigger:
                done.append(cf - M["g"])
                off = M["post"] - c
                cf, cw = M["post"], M["postw"]
            if prev is not None and r["t"] - prev["t"] > 3600 and cw > 20e3:
                cw = max(cw, cf - SYSTEM)          # cold cache: rewrite all of it
            reqs.append((r["inp"], cw, max(cf - cw - r["inp"], 0), r["out"], r["t"]))
            prev, prevcf = r, cf
    return reqs, done


def totals(reqs, done, M):
    """Token totals [input, cache write, cache read, output], summary requests included."""
    t = np.array([sum(x[i] for x in reqs) for i in range(4)], float)
    return t + [len(done) * M["inp"], len(done) * M["cw"], sum(done) * M["rf"], len(done) * M["out"]]


def weigh(t, w):
    win, cw5, cw1, cr, out = w
    return t @ [win, cw1, cr, out]          # main-thread Opus writes are 1h writes


def optimum(M, w):
    """T* = R + sqrt(2 g (R + K0 / r)): K0 is everything a compaction costs except reading the
    context, including the first request after it writing ~34K instead of reading it."""
    win, cw5, cw1, cr, out = w
    K0 = M["inp"] * win + M["cw"] * cw1 + M["out"] * out + M["postw"] * (cw1 - cr)
    return M["post"] + np.sqrt(2 * M["g"] * (M["post"] + K0 / (cr * M["rf"]))), K0


GRID = [100e3 + 10e3 * i for i in range(21)] + [168e3, 400e3, 1e12]


def curve(main_, comps, M):
    return {t: totals(*simulate(main_, comps, t, M), M) for t in GRID}


def label(t):
    return "never" if t > 1e9 else f"{t/1e3:.0f}K"


def main():
    S = load()
    M = measure(S)
    print(f"summary request, from {M['sessions']} sessions / {M['n']} compactions: input {M['inp']/1e3:.1f}K,"
          f" cache write {M['cw']/1e3:.1f}K, reads {100*M['rf']:.0f}% of the context,"
          f" output {M['out']/1e3:.1f}K (session median {M['out_med']/1e3:.1f}K)")
    print(f"first request after compaction (n={M['ncomp_post']}): context {M['post']/1e3:.1f}K,"
          f" writes {M['postw']/1e3:.1f}K; growth per request mean {M['g']/1e3:.2f}K"
          f" median {M['g_med']/1e3:.2f}K")
    fitted = [float(x) for x in sys.argv[1:6]] if len(sys.argv) > 5 else [1, 1.25, 2, 0.0258, 5.25]
    W = {"fitted": fitted, "API": [1, 1.25, 2, 0.05, 5]}
    main_, side, comps = sessions(S)
    print(f"\nmain-thread Opus requests {sum(map(len, main_.values()))} in {len(main_)} sessions,"
          f" {sum(map(len, comps.values()))} auto compactions; subagent requests {len(side)} (left out:"
          f" the same in every scenario)")
    actual = [(r["inp"], r["cw5"] + r["cw1"], r["cr"], r["out"], r["t"]) for rs in main_.values() for r in rs]
    tact = totals(actual, [c["pre"] for cs in comps.values() for c in cs], M)
    C = curve(main_, comps, M)
    for name, w in W.items():
        T, K0 = optimum(M, w)
        cost = {t: weigh(v, w) for t, v in C.items()}
        best = min(cost, key=cost.get)
        print(f"\n{name} weights (input, 5m write, 1h write, read, output) = {w}")
        print(f"   one compaction costs {K0/1e3:.0f}K units + reading the context;"
              f" analytic optimum {T/1e3:.0f}K trigger = {(T+33e3)/1e3:.0f}K window")
        print(f"   actual {weigh(tact, w)/1e6:.1f}M units; replay at 168K {cost[168e3]/1e6:.1f}M")
        for t in GRID:
            if t in (100e3, 120e3, 140e3, 150e3, 160e3, 168e3, 180e3, 190e3, 200e3, 220e3, 250e3, 270e3, 300e3, 400e3, 1e12):
                print(f"   trigger {label(t):>6s} (window {label(t+33e3) if t < 1e9 else '-':>5s}):"
                      f" {100*(cost[t]/cost[best]-1):+5.1f}% vs best ({label(best)})")

    # Weight uncertainty: every bootstrap sample of the fit, through the same replay.
    try:
        B = np.load("data/bootstrap.npy")
    except FileNotFoundError:
        B = None
    if B is not None:
        opt, pen, pen2 = [], [], []
        for read, out in B[:, :2]:
            w = [1, 1.25, 2, read, out]
            cost = {t: weigh(v, w) for t, v in C.items()}
            b = min(cost, key=cost.get)
            opt.append(b)
            pen.append(100 * (cost[168e3] / cost[b] - 1))
            pen2.append(100 * (cost[270e3] / cost[b] - 1))
        q = lambda x: np.percentile(x, [10, 50, 90])
        print(f"\nover {len(B)} bootstrap fits: optimum trigger 10/50/90% = "
              + " / ".join(label(x) for x in q(opt))
              + f"; cost of 168K over the optimum {' / '.join(f'{x:.1f}%' for x in q(pen))}"
              + f"; of 270K (300K window) {' / '.join(f'{x:.1f}%' for x in q(pen2))}")

    # Stability: sessions grouped by the week they started.
    w = W["fitted"]
    print("\nby week the session started, fitted weights:")
    week = 7 * 86400
    for k in range(3):
        a, b = SWITCH + k * week, SWITCH + (k + 1) * week
        sub = {s: rs for s, rs in main_.items() if a <= rs[0]["t"] < b}
        if not sub:
            continue
        cost = {t: weigh(v, w) for t, v in curve(sub, comps, M).items()}
        best = min(cost, key=cost.get)
        print(f"   {dt.datetime.fromtimestamp(a):%m-%d} ({len(sub)} sessions): best {label(best)}; "
              + "  ".join(f"{label(t)} {100*(cost[t]/cost[best]-1):+.1f}%" for t in (130e3, 150e3, 168e3, 200e3, 270e3)))


if __name__ == "__main__":
    main()
