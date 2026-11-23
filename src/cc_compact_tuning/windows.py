"""Rebuild the 5-hour windows that ended in a usage-limit hit.

A window that hit the limit spans reset - 5h .. hit. Accounts are switched with /login (or
outside Claude Code transcripts via claude-account), and each account has its own windows, so a
window only counts requests after the last /login before the hit. When the status line's
rate-limit log shows multiple accounts active during a window, requests and compactions are
attributed by (session, timestamp) to the hitting (5h reset, 7d reset) account; single-account
windows keep all requests in the time span so unlogged headless (`claude -p`) runs are preserved.

A window is flagged when the transcripts cannot explain when it started (its account was used more
than 10 minutes before the first logged request), when two unattributed windows overlap, when a
multi-account window has too many unattributed tokens (>5%), or when a large share of Fable
appears in fewer than MIN_FABLE_WINDOWS clean windows.
"""

import collections
import itertools
from collections.abc import Iterable
from typing import Any

import numpy as np

from .pricing import API_W, CATS, MODELS, local, model_factor, row
from .store import Store

SLACK = 600
FABLE_SHARE = 0.1  # a window with at least this share of Fable (at API prices) identifies it
MIN_FABLE_WINDOWS = 3
UNATTR_SHARE_LIMIT = 0.05


def parse_rate_log(lines: Iterable[str]) -> list[tuple[float, str, str, float, int, str]]:
    """Parse rate-limit TSV lines into sorted (t, session8, model, pct5, reset5, acct7)."""
    rows = []
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) < 7 or p[0] == "time" or not p[3] or not p[4] or not p[6]:
            continue
        try:
            rows.append((float(p[0]), p[1], p[2], float(p[3]), int(float(p[4])), p[6]))
        except ValueError:
            continue
    return sorted(rows)


def session_timeline(
    rows: Iterable[tuple[float, str, str, float, int, str]],
) -> dict[str, list[tuple[float, int, str]]]:
    """Map each 8-char session prefix to sorted (t, reset5, acct7) log observations."""
    tl: dict[str, list[tuple[float, int, str]]] = collections.defaultdict(list)
    for t, sess8, _, _, r5, a7 in rows:
        tl[sess8].append((t, r5, a7))
    for v in tl.values():
        v.sort()
    return dict(tl)


def attribute_event(
    t: float,
    session: str,
    timeline: dict[str, list[tuple[float, int, str]]],
    length: int = 5 * 3600,
) -> tuple[int, str] | None:
    """Attribute an event at `t` in `session` to its (reset5, acct7) via the session's timeline."""
    evs = timeline.get(session[:8])
    if not evs:
        return None
    cand = [e for e in evs if e[1] - length - 300 <= t <= e[1] + 120]
    if not cand:
        return None
    chosen = next((e for e in cand if e[0] >= t - 5), cand[-1])
    return chosen[1], chosen[2]


def has_unexplained_jump(
    mono: list[tuple[float, float]],
    R: list[dict[str, Any]],
    T: np.ndarray,
    min_dp: float = 8.0,
    min_api_per_pp: float = 25e3,
) -> tuple[float, float] | None:
    """Detect a >= `min_dp` percentage jump without matching transcript tokens."""
    if not len(T) or len(mono) < 2:
        return None
    for (t0, p0), (t1, p1) in itertools.pairwise(mono):
        dp = p1 - p0
        if dp < min_dp:
            continue
        sub = R[np.searchsorted(T, t0 - 60) : np.searchsorted(T, t1 + 5, side="right")]
        api = float(tokens(sub).sum(0) @ API_W) if sub else 0.0
        per_pp = api / dp
        if per_pp < min_api_per_pp:
            return dp, per_pp
    return None


def tokens(reqs: list[dict[str, Any]]) -> np.ndarray:
    """Opus-equivalent tokens per category at API prices; rows [Opus and Haiku], [Sonnet], [Fable]."""
    X = np.zeros((3, 5))
    for r in reqs:
        f = model_factor(r)
        X[row(r)] += [r[c] * f[k] for k, c in enumerate(CATS)]
    return X


def build(store: Store, kind: str = "5h", length: int = 5 * 3600) -> list[dict[str, Any]]:
    R = sorted(store["requests"].values(), key=lambda r: r["t"])
    T = np.array([r["t"] for r in R])
    logins = sorted(lg["t"] for lg in store["logins"].values())
    log_rows = parse_rate_log(store.get("rate_limits") or ())
    timeline = session_timeline(log_rows)

    thread_model: dict[tuple[str, str | None], str] = {}
    sess_model: dict[str, str] = {}
    for r in R:
        if r.get("model"):
            thread_model[(r["session"], r.get("agent"))] = r["model"]
            if not r.get("side"):
                sess_model[r["session"]] = r["model"]

    hits: dict[float, dict] = {}
    for h in sorted(store["hits"].values(), key=lambda h: h["t"]):
        if h["kind"] == kind and h["reset"]:
            hits.setdefault(h["reset"], h)  # retries repeat the hit; keep the first
    all_comps = sorted(store["compactions"].values(), key=lambda c: c["t"])

    out = []
    for reset, h in sorted(hits.items()):
        start = reset - length
        login = max((t for t in logins if t < h["t"]), default=None)
        eff = max(start, login) if login else start
        w = R[np.searchsorted(T, eff) : np.searchsorted(T, h["t"], side="right")]
        if not w:
            continue
        comps_w = [c for c in all_comps if eff <= c["t"] <= h["t"]]

        flags = []
        attributed = False
        unattr_share = 0.0

        win_log = [row_l for row_l in log_rows if eff - 60 <= row_l[0] <= h["t"] + 60]
        active = {(r5, a7) for _, _, _, _, r5, a7 in win_log if r5 >= eff - 60 and r5 - length <= h["t"] + 60}
        accts = {a7 for _, a7 in active}
        other_r5 = [
            r5 for r5, _ in active if abs(r5 - reset) > 60 and min(h["t"], r5) - max(eff, r5 - length) > SLACK
        ]
        multi_active = len(accts) > 1 or bool(other_r5)
        single_acct_logged = bool(win_log) and not multi_active
        hit_acct = None

        if multi_active:
            hit_accts = collections.Counter(
                a7
                for t_l, _, _, _, r5, a7 in log_rows
                if eff - 60 <= t_l <= h["t"] + 300 and abs(r5 - reset) <= 60
            )
            hit_acct = hit_accts.most_common(1)[0][0] if hit_accts else None
            if hit_acct is not None:
                mine, unattr = [], []
                for r in w:
                    attr = attribute_event(r["t"], r["session"], timeline, length)
                    if attr is not None and attr[1] == hit_acct and abs(attr[0] - reset) <= 60:
                        mine.append(r)
                    elif attr is None:
                        unattr.append(r)
                mine_comps = []
                for c in comps_w:
                    attr = attribute_event(c["t"], c["session"], timeline, length)
                    if attr is not None and attr[1] == hit_acct and abs(attr[0] - reset) <= 60:
                        mine_comps.append(c)
                mine_api = float(tokens(mine).sum(0) @ API_W) if mine else 0.0
                unattr_api = float(tokens(unattr).sum(0) @ API_W) if unattr else 0.0
                tot_api = mine_api + unattr_api
                unattr_share = unattr_api / tot_api if tot_api > 0 else 0.0
                attributed = True
                w = mine
                comps_w = mine_comps
                if not w:
                    continue
                if unattr_share > UNATTR_SHARE_LIMIT:
                    flags.append(f"{unattr_share:.0%} unattributed during multi-account window")
            else:
                flags.append("multiple accounts active in rate-limit log")

        if attributed:
            own_log = sorted(
                (r_l[0], r_l[3]) for r_l in win_log if r_l[5] == hit_acct and abs(r_l[4] - reset) <= 60
            )
        elif single_acct_logged:
            own_log = sorted((r_l[0], r_l[3]) for r_l in win_log)
        else:
            own_log = []
        mono, mx = [], -1.0
        for t_l, p5 in own_log:
            if mx < p5 < 100.0:
                mono.append((t_l, p5))
                mx = p5
        jump = has_unexplained_jump(mono, R, T)
        if jump is not None:
            flags.append(f"unexplained +{jump[0]:.0f}% jump in rate-limit log (external usage)")

        anchor = login if login and login >= start - SLACK else w[0]["t"]
        if anchor - start > SLACK:
            flags.append(f"started {int((anchor - start) / 60)}m before the first request")
        if login and start - SLACK <= login and w[0]["t"] - login > SLACK:
            flags.append("idle after /login")

        inside = [c for c in comps_w if c.get("trigger") == "auto"]
        compaction_events = [
            dict(
                t=c["t"],
                session=c["session"],
                agent=c.get("agent"),
                pre=c.get("pre"),
                trigger=c.get("trigger"),
                model=(
                    thread_model.get((c["session"], c.get("agent")))
                    or sess_model.get(c["session"])
                    or MODELS["opus"]
                ),
            )
            for c in comps_w
        ]
        X = tokens(w)
        tot = float(X.sum(0) @ API_W)
        out.append(
            dict(
                hit=h["t"],
                reset=reset,
                start=start,
                eff=eff,
                n=len(w),
                X=X,
                fable=float(X[2] @ API_W / tot) if tot > 0 else 0.0,
                compactions=len(inside),
                compaction_events=compaction_events,
                attributed=attributed,
                single_acct_logged=single_acct_logged,
                unattributed_share=unattr_share,
                flags=flags,
                label=local(h["t"]),
            )
        )

    for a in out:
        if a["attributed"]:
            continue
        for b in out:
            if a is b or (a["single_acct_logged"] and b["attributed"]):
                continue
            if a["eff"] < b["hit"] and b["eff"] < a["hit"]:
                a["flags"].append(f"overlaps the window to {b['label']}: two accounts in use")
    fable = [w for w in out if w["fable"] >= FABLE_SHARE and not w["flags"]]
    if len(fable) < MIN_FABLE_WINDOWS:
        for w in fable:
            w["flags"].append(
                f"{w['fable']:.0%} Fable: factor needs {MIN_FABLE_WINDOWS} such windows, {len(fable)} so far"
            )
    return out


def report(store: Store) -> None:
    for w in build(store):
        X = w["X"].sum(0)
        tag = " [attr]" if w["attributed"] else ""
        print(
            f"{w['label']}  from {local(w['eff'])[-5:]} n={w['n']:4d} c={w['compactions']}{tag}",
            " ".join(f"{c}={x / 1e6:5.2f}M" for c, x in zip(CATS[1:], X[1:], strict=True)),
            f"sonnet={w['X'][1] @ API_W / 1e6:4.1f}M fable={w['X'][2] @ API_W / 1e6:4.1f}M",
            f"API={X @ API_W / 1e6:5.1f}M",
            "; ".join(w["flags"]),
        )
