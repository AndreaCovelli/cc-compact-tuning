"""Measure what a compaction costs and replay the main-thread sessions of one model (Opus, Sonnet,
Haiku or Fable) since the switch to a 200K window at other trigger points.

The summary request that writes a compaction is not in the transcript; the session's cost-state
record counts it, so per-session cost-state minus transcript totals gives its tokens.

After a simulated compaction a session restarts at its own post-compaction context and cache write
(the first request after its real compactions), each session's median shrunk toward the median over
all sessions by k = within / between-session variance, so a session with one compaction is not
taken at face value. Sessions without a measured compaction use the overall medians. The context
left is not modelled as growing with the trigger. It tracks the triggering turn's overshoot (the
context compacted minus the request before it, correlation ~0.8), not the context before that turn
(~-0.1), and compactions at a ~268K trigger before the switch left less (median 40.6K) than at
~169K after it (47.6K). The apparent within-session slope comes from a few large overshoots.

Tokens are weighed either with per-model fitted weights from `fit-log` (when available in
`FITTED_MODEL_W`, e.g. Fable) or at the model's API price ratios to Opus (Haiku costs 5x above a
100K prompt) followed by Opus usage weights (`FITTED_W`). A model with fewer than MIN_COMPACTIONS
measured compactions is replayed on the Opus sessions instead. The bootstrap interval covers weight
uncertainty only, not session sampling or extrapolation beyond observed triggers.
"""

import collections
import datetime as dt
import itertools
import statistics as st
from pathlib import Path

import numpy as np

from .pricing import API_W, LONG_PROMPT, MODELS, PRICES, price_ratio
from .store import Store

SWITCH = dt.datetime(2026, 10, 27, 6, 41, 1, tzinfo=dt.UTC).timestamp()
OPUS = MODELS["opus"]
SYSTEM = 14.3e3  # system prompt + tools: what a cold resume after compaction still reads
NAMES = {m: k.capitalize() for k, m in MODELS.items()}
MIN_COMPACTIONS = 10  # measured compactions a model needs for its own sessions to be replayed

FITTED_W = [1, 2.06, 2, 0.0366, 7.65]  # input, 5m write, 1h write, read, output (per Opus token, fit-log)
FITTED_MODEL_W: dict[str, list[float]] = {
    MODELS["fable"]: [1, 2.06, 2, 0.0227, 5.74],
}  # per-model fitted weights in that model's own tokens (fit-log)


def ctx(r):
    return r["inp"] + r["cw5"] + r["cw1"] + r["cr"]


def sessions(S, model=OPUS, since=SWITCH):
    main, side = collections.defaultdict(list), []
    for r in S["requests"].values():
        if r["t"] < since or r["model"] != model:
            continue
        if r["side"]:
            side.append(r)
        else:
            main[r["session"]].append(r)
    comps = collections.defaultdict(list)
    for c in S["compactions"].values():
        if c["t"] >= since and c["trigger"] == "auto" and not c["agent"] and c["session"] in main:
            comps[c["session"]].append(c)
    return (
        {k: sorted(v, key=lambda r: r["t"]) for k, v in main.items()},
        side,
        {k: sorted(v, key=lambda c: c["t"]) for k, v in comps.items()},
    )


def measure(S, model=OPUS):
    """Summary request and post-compaction context of `model`'s sessions, or None when fewer than
    MIN_COMPACTIONS compactions can be measured."""
    main, side, comps = sessions(S, model)
    allreq = collections.defaultdict(list)
    for r in S["requests"].values():
        if r["model"] == model:
            allreq[r["session"]].append(r)
    all_main_bts = collections.defaultdict(list)
    for c in S["compactions"].values():
        if not c["agent"]:
            all_main_bts[c["session"]].append(c["t"])
    early = {c["session"] for c in S["compactions"].values() if c["t"] < SWITCH}
    rows = []
    for sid, cs in comps.items():
        st_ = S["cost_states"].get(sid)
        mu = st_ and st_["modelUsage"].get(model)
        if not mu or sid in early:
            continue
        rs = allreq[sid]
        tr = [
            sum(r["inp"] for r in rs),
            sum(r["cw5"] + r["cw1"] for r in rs),
            sum(r["cr"] for r in rs),
            sum(r["out"] for r in rs),
        ]
        cv = [
            mu.get(k, 0)
            for k in ("inputTokens", "cacheCreationInputTokens", "cacheReadInputTokens", "outputTokens")
        ]
        d = [a - b for a, b in zip(cv, tr, strict=True)]
        pre = sum(c["pre"] for c in cs)
        if min(d) < -1000 or not 0.85 <= d[2] / pre <= 1.1:
            continue  # resumed or restarted sessions, whose cost-state does not cover the file
        rows.append((len(cs), pre, d))
    n = sum(r[0] for r in rows)
    if n < MIN_COMPACTIONS:
        return None
    pre = sum(r[1] for r in rows)
    tot = [sum(r[2][i] for r in rows) for i in range(4)]
    m = dict(
        sessions=len(rows),
        n=n,
        inp=tot[0] / n,
        cw=tot[1] / n,
        rf=tot[2] / pre,
        out=tot[3] / n,
        out_med=st.median(r[2][3] / r[0] for r in rows),
    )
    # First main-thread request after each compaction: what is left, and what it writes.
    post, postw, g = [], [], []
    by_post, by_postw = collections.defaultdict(list), collections.defaultdict(list)
    for sid, rs in main.items():
        bts = [c["t"] for c in comps.get(sid, [])]
        for b in bts:
            nxt = next((r for r in rs if r["t"] > b), None)
            if nxt and nxt["t"] - b < 3600:
                post.append(ctx(nxt))
                postw.append(nxt["cw5"] + nxt["cw1"])
                by_post[sid].append(post[-1])
                by_postw[sid].append(postw[-1])
        sbts = all_main_bts.get(sid, ())
        for a, b in itertools.pairwise(rs):
            if any(a["t"] < bt < b["t"] for bt in sbts):
                continue
            ca, cb = ctx(a), ctx(b)
            if ca > 40e3 and -200e3 < cb - ca < 200e3:
                g.append(cb - ca)
    if not post or not g:
        return None
    m |= dict(
        post=st.median(post), postw=st.median(postw), g=st.mean(g), g_med=st.median(g), ncomp_post=len(post)
    )
    m["post_s"], m["post_k"] = shrink(by_post, m["post"])
    m["postw_s"], m["postw_k"] = shrink(by_postw, m["postw"])
    return m


def shrink(by: dict[str, list[float]], center: float) -> tuple[dict[str, float], float]:
    """Each session's median pulled toward `center`: (n * median + k * center) / (n + k) for a
    session with n measurements, where k = within-session / between-session variance (one-way
    ANOVA moments). Returns ({}, inf) when there is no between-session variance to use."""
    multi = [v for v in by.values() if len(v) >= 2]
    dof = sum(len(v) - 1 for v in multi)
    if not dof or len(by) < 2:
        return {}, np.inf
    within = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in multi) / dof
    means = [np.mean(v) for v in by.values()]
    between = np.var(means, ddof=1) - within * np.mean([1 / len(v) for v in by.values()])
    if between <= 0:
        return {}, np.inf
    k = within / between
    return {s: (len(v) * st.median(v) + k * center) / (len(v) + k) for s, v in by.items()}, float(k)


def measure_all(S: Store) -> dict[str, dict]:
    """Return compaction measurements for every model with at least MIN_COMPACTIONS compactions."""
    out = {}
    for m in MODELS.values():
        M = measure(S, m)
        if M is not None:
            out[m] = M
    return out


def simulate(main, comps, trigger, M):
    """Replay each session with compaction at `trigger` instead of where it happened. After a
    compaction the context restarts at the session's own (shrunk) post-compaction context and write
    when measured, else at the medians over all sessions."""
    reqs, done = [], []
    for sid, rs in main.items():
        bts = [c["t"] for c in comps.get(sid, [])]
        post = M.get("post_s", {}).get(sid, M["post"])
        postw = min(M.get("postw_s", {}).get(sid, M["postw"]), post)
        off, prev, prevcf = 0.0, None, None
        for r in rs:
            c = ctx(r)
            cw = r["cw5"] + r["cw1"]
            if prev is not None and any(prev["t"] < b < r["t"] for b in bts):
                off = prevcf + M["g"] - c  # the real compaction does not happen here
                cw = min(cw, M["g"])
            cf = max(c + off, min(c, SYSTEM))
            if cf > trigger:
                done.append(cf - M["g"])
                off = post - c
                cf, cw = post, postw
            if prev is not None and r["t"] - prev["t"] > 3600 and cw > 20e3:
                cw = max(cf - SYSTEM, 0.0)  # cold cache: rewrite all of it
            else:
                cw = min(cw, max(cf - r["inp"], 0.0))
            reqs.append((r["inp"], cw, max(cf - cw - r["inp"], 0.0), r["out"], r["t"]))
            prev, prevcf = r, cf
    return reqs, done


def ratios(model, prompt):
    """Per-request [input, write, read, output] price ratios to Opus for prompts of `prompt`
    tokens. Main-thread writes are 1h writes, whatever the model."""
    lo, hi = (np.array(price_ratio(model, p))[[0, 2, 3, 4]] for p in (0, np.inf))
    return np.where((prompt > LONG_PROMPT.get(model, (np.inf,))[0])[:, None], hi, lo)


def totals(reqs, done, M, model=OPUS):
    """Opus-equivalent token totals [input, cache write, cache read, output] of `model`'s
    requests, summary requests included."""
    a = np.array([x[:4] for x in reqs], float).reshape(-1, 4)
    s = np.array([(M["inp"], M["cw"], M["rf"] * p, M["out"]) for p in done], float).reshape(-1, 4)
    a = np.vstack([a, s])
    return (a * ratios(model, a[:, :3].sum(1))).sum(0)


def weigh(t, w):
    win, cw5, cw1, cr, out = w
    return t @ [win, cw1, cr, out]  # main-thread writes are 1h writes


def optimum(M, w, model=OPUS, use_api_ratio=True):
    """T* = R + sqrt(2 g (rf R + K0 / r)): K0 is everything a compaction costs except reading the
    context, including the first request after it writing ~34K instead of reading it. Uses the
    model's base prices when `use_api_ratio` is True, so it ignores Haiku's dearer prompts above 100K."""
    scale = price_ratio(model) if use_api_ratio else [1.0] * 5
    win, cw5, cw1, cr, out = (a * b for a, b in zip(w, scale, strict=True))
    K0 = M["inp"] * win + M["cw"] * cw1 + M["out"] * out + M["postw"] * (cw1 - cr)
    return M["post"] + np.sqrt(2 * M["g"] * (M["rf"] * M["post"] + K0 / cr)), K0


GRID = sorted([60e3 + 10e3 * i for i in range(25)] + [95e3, 168e3, 400e3, 1e12])  # 95K: below Haiku's step
SHOWN = (60e3, 80e3, 100e3, 120e3, 140e3, 150e3, 160e3, 168e3, 180e3, 190e3, 200e3, 220e3, 250e3, 270e3)
SHOWN += (300e3, 400e3, 1e12)


def curve(main_, comps, M, model=OPUS):
    return {t: totals(*simulate(main_, comps, t, M), M, model) for t in GRID}


def pct(x):
    return np.percentile(x, [10, 50, 90], method="nearest")


def label(t):
    return "never" if t > 1e9 else f"{t / 1e3:.0f}K"


def workload(S: Store, model: str = OPUS):
    """The model whose sessions are replayed for `model`, with its compaction measurements: the
    model itself, or Opus when it has fewer than MIN_COMPACTIONS measured compactions."""
    for m in dict.fromkeys((model, OPUS)):
        M = measure(S, m)
        if M is not None:
            return m, M
    raise SystemExit(
        f"fewer than {MIN_COMPACTIONS} measured auto compactions, even on Opus: nothing to replay"
    )


def _boot_weights(data_dir: Path, model: str, use_model_w: bool) -> tuple[list[list[float]], bool] | None:
    npz = data_dir / "bootstrap_log.npz"
    if npz.exists():
        d = np.load(npz)
        if use_model_w and model == MODELS["fable"] and "fable_w" in d and len(d["fable_w"]) > 0:
            return [[1.0, cw5, 2.0, rd, ot] for cw5, rd, ot in d["fable_w"]], False
        if "opus_w" in d and len(d["opus_w"]) > 0:
            return [[1.0, cw5, 2.0, rd, ot] for cw5, rd, ot in d["opus_w"]], True
    path = data_dir / "bootstrap.npy"
    if path.exists():
        B = np.load(path)
        return [[1.0, 1.25, 2.0, rd, ot] for rd, ot in B[:, :2]], True
    return None


def report(S: Store, data_dir: Path, fitted: list[float] | None = None, model: str = OPUS) -> None:
    """Measure compactions, then replay at every trigger in GRID under fitted and API weights,
    under every bootstrap sample of the fit (if `fit` or `fit-log` has saved one), and week by week."""
    source, M = workload(S, model)
    name, src = NAMES.get(model, model), NAMES.get(source, source)
    if source != model:
        print(
            f"{name}: fewer than {MIN_COMPACTIONS} measured auto compactions of its own, so the Opus"
            f" sessions are replayed as if they ran on {name}\n"
        )
    print(
        f"summary request, from {M['sessions']} sessions / {M['n']} compactions: input {M['inp'] / 1e3:.1f}K,"
        f" cache write {M['cw'] / 1e3:.1f}K, reads {100 * M['rf']:.0f}% of the context,"
        f" output {M['out'] / 1e3:.1f}K (session median {M['out_med'] / 1e3:.1f}K)"
    )
    print(
        f"first request after compaction (n={M['ncomp_post']}): context {M['post'] / 1e3:.1f}K,"
        f" writes {M['postw'] / 1e3:.1f}K; growth per request mean {M['g'] / 1e3:.2f}K"
        f" median {M['g_med'] / 1e3:.2f}K"
    )
    if M["post_s"]:
        p = np.percentile(list(M["post_s"].values()), [10, 90])
        print(
            f"per-session post-compaction context ({len(M['post_s'])} sessions, shrunk with"
            f" k={M['post_k']:.2f}, write k={M['postw_k']:.2f}):"
            f" 10/90% {p[0] / 1e3:.1f}K / {p[1] / 1e3:.1f}K;"
            f" sessions without a measured compaction use the median"
        )
    use_model_w = fitted is None and model in FITTED_MODEL_W
    fitted_w = list(FITTED_MODEL_W[model] if use_model_w else (FITTED_W if fitted is None else fitted))

    main_, side, comps = sessions(S, source)
    print(
        f"\nmain-thread {src} requests {sum(map(len, main_.values()))} in"
        f" {len(main_)} sessions, {sum(map(len, comps.values()))} auto compactions; subagent requests"
        f" {len(side)} (left out: the same in every scenario)"
    )
    ratio = ", ".join(f"{x:.3g}" for x in price_ratio(model))
    long_ = ""
    if model in LONG_PROMPT:
        n, p = LONG_PROMPT[model]
        long_ = f"; x{p[0] / PRICES[model][0]:.0f} that above {n / 1e3:.0f}K prompts"
    if model != OPUS and not use_model_w:
        print(f"priced as {name}: {ratio} x Opus (input, 5m write, 1h write, read, output){long_}")
    elif use_model_w:
        print(f"fitted weights are per {name} token from fit-log (API comparison still uses {name} ratios)")

    actual = [(r["inp"], r["cw5"] + r["cw1"], r["cr"], r["out"], r["t"]) for rs in main_.values() for r in rs]
    c_pres = [c["pre"] for cs in comps.values() for c in cs]
    C_api = curve(main_, comps, M, model)
    C_fit = curve(main_, comps, M, OPUS) if use_model_w else C_api
    tact_api = totals(actual, c_pres, M, model)
    tact_fit = totals(actual, c_pres, M, OPUS) if use_model_w else tact_api

    scenarios = [
        ("fitted", fitted_w, C_fit, tact_fit, not use_model_w, name if use_model_w else "Opus"),
        ("API", API_W, C_api, tact_api, True, "Opus"),
    ]
    for wname, w, C_use, tact_use, use_api, unit in scenarios:
        T, K0 = optimum(M, w, model, use_api_ratio=use_api)
        cost = {t: weigh(v, w) for t, v in C_use.items()}
        best = min(cost, key=cost.get)
        print(f"\n{wname} weights per {unit} token (input, 5m write, 1h write, read, output) = {w}")
        print(
            f"   one compaction costs {K0 / 1e3:.0f}K units + reading the context;"
            f" analytic optimum {T / 1e3:.0f}K trigger = {(T + 33e3) / 1e3:.0f}K window"
            + (
                f" (at the price below {LONG_PROMPT[model][0] / 1e3:.0f}K)"
                if use_api and model in LONG_PROMPT
                else ""
            )
        )
        print(f"   actual {weigh(tact_use, w) / 1e6:.1f}M units; replay at 168K {cost[168e3] / 1e6:.1f}M")
        for t in SHOWN:
            print(
                f"   trigger {label(t):>6s} (window {label(t + 33e3) if t < 1e9 else '-':>5s}):"
                f" {100 * (cost[t] / cost[best] - 1):+5.1f}% vs best ({label(best)})"
            )

    # Weight uncertainty: every bootstrap sample of the fit, through the same replay.
    boot = _boot_weights(data_dir, model, use_model_w) if fitted is None else None
    if boot is not None:
        ws_boot, boot_api = boot
        C_b = C_api if boot_api else C_fit
        opt, pen, pen2 = [], [], []
        for w_b in ws_boot:
            cost = {t: weigh(v, w_b) for t, v in C_b.items()}
            b = min(cost, key=cost.get)
            opt.append(b)
            pen.append(100 * (cost[168e3] / cost[b] - 1))
            pen2.append(100 * (cost[270e3] / cost[b] - 1))
        print(
            f"\nover {len(ws_boot)} bootstrap fits (weight uncertainty only): optimum trigger 10/50/90% = "
            + " / ".join(label(x) for x in pct(opt))
            + f"; cost of 168K over the optimum {' / '.join(f'{x:.1f}%' for x in pct(pen))}"
            + f"; of 270K (300K window) {' / '.join(f'{x:.1f}%' for x in pct(pen2))}"
        )

    # Stability: sessions grouped by the week they started.
    print("\nby week the session started, fitted weights:")
    week = 7 * 86400
    max_t = max((rs[0]["t"] for rs in main_.values()), default=SWITCH)
    k = 0
    while SWITCH + k * week <= max_t:
        a, b = SWITCH + k * week, SWITCH + (k + 1) * week
        k += 1
        sub = {s: rs for s, rs in main_.items() if a <= rs[0]["t"] < b}
        if not sub:
            continue
        C_sub = curve(sub, comps, M, OPUS if use_model_w else model)
        cost = {t: weigh(v, fitted_w) for t, v in C_sub.items()}
        best = min(cost, key=cost.get)
        print(
            f"   {dt.datetime.fromtimestamp(a):%m-%d} ({len(sub)} sessions): best {label(best)}; "
            + "  ".join(
                f"{label(t)} {100 * (cost[t] / cost[best] - 1):+.1f}%"
                for t in (130e3, 150e3, 168e3, 200e3, 270e3)
            )
        )
