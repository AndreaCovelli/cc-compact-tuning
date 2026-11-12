"""Rebuild the 5-hour windows that ended in a usage-limit hit.

A window that hit the limit spans reset - 5h .. hit. Accounts are switched with /login, and each
account has its own windows, so a window only counts requests after the last /login before the
hit. A window is flagged when the transcripts cannot explain when it started: its account must
have been used, outside Claude Code or before a /login, more than 10 minutes before the first
request we see (window starts are rounded down to 10 minutes).

A window with a large share of Fable is also flagged until there are MIN_FABLE_WINDOWS of them:
with fewer, a free Fable factor could absorb each window's error rather than measure Fable.
"""

from typing import Any

import numpy as np

from .pricing import API_W, CATS, local, model_factor, row
from .store import Store

SLACK = 600
FABLE_SHARE = 0.1  # a window with at least this share of Fable (at API prices) identifies it
MIN_FABLE_WINDOWS = 3


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
    hits: dict[float, dict] = {}
    for h in sorted(store["hits"].values(), key=lambda h: h["t"]):
        if h["kind"] == kind and h["reset"]:
            hits.setdefault(h["reset"], h)  # retries repeat the hit; keep the first
    comps = [c for c in store["compactions"].values() if c["trigger"] == "auto"]
    out = []
    for reset, h in sorted(hits.items()):
        start = reset - length
        login = max((t for t in logins if t < h["t"]), default=None)
        eff = max(start, login) if login else start
        w = R[np.searchsorted(T, eff) : np.searchsorted(T, h["t"], side="right")]
        if not w:
            continue
        flags = []
        anchor = login if login and login >= start - SLACK else w[0]["t"]
        if anchor - start > SLACK:
            flags.append(f"started {int((anchor - start) / 60)}m before the first request")
        if login and start - SLACK <= login and w[0]["t"] - login > SLACK:
            flags.append("idle after /login")
        inside = [c for c in comps if eff <= c["t"] <= h["t"]]
        X = tokens(w)
        out.append(
            dict(
                hit=h["t"],
                reset=reset,
                start=start,
                eff=eff,
                n=len(w),
                X=X,
                fable=float(X[2] @ API_W / (X.sum(0) @ API_W)),
                compactions=len(inside),
                flags=flags,
                label=local(h["t"]),
            )
        )
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
        print(
            f"{w['label']}  from {local(w['eff'])[-5:]} n={w['n']:4d} c={w['compactions']}",
            " ".join(f"{c}={x / 1e6:5.2f}M" for c, x in zip(CATS[1:], X[1:], strict=True)),
            f"sonnet={w['X'][1] @ API_W / 1e6:4.1f}M fable={w['X'][2] @ API_W / 1e6:4.1f}M",
            f"API={X @ API_W / 1e6:5.1f}M",
            "; ".join(w["flags"]),
        )
