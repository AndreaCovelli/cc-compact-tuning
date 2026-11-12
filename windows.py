"""Rebuild the 5-hour windows that ended in a usage-limit hit.

A window that hit the limit spans reset - 5h .. hit. Accounts are switched with /login, and each
account has its own windows, so a window only counts requests after the last /login before the
hit. A window is flagged when the transcripts cannot explain when it started: its account must
have been used, outside Claude Code or before a /login, more than 10 minutes before the first
request we see (window starts are rounded down to 10 minutes).
"""
import numpy as np
from common import CATS, load, local, model_factor

SLACK = 600


def build(S=None, kind="5h", length=5 * 3600):
    S = S or load()
    R = sorted(S["requests"].values(), key=lambda r: r["t"])
    T = np.array([r["t"] for r in R])
    logins = sorted(l["t"] for l in S["logins"].values())
    hits = {}
    for h in sorted(S["hits"].values(), key=lambda h: h["t"]):
        if h["kind"] == kind and h["reset"]:
            hits.setdefault(h["reset"], h)          # retries repeat the hit; keep the first
    comps = sorted(S["compactions"].values(), key=lambda c: c["t"])
    out = []
    for reset, h in sorted(hits.items()):
        start = reset - length
        login = max((t for t in logins if t < h["t"]), default=None)
        eff = max(start, login) if login else start
        i, j = np.searchsorted(T, eff), np.searchsorted(T, h["t"], side="right")
        w = R[i:j]
        if not w:
            continue
        flags = []
        anchor = login if login and login >= start - SLACK else w[0]["t"]
        if anchor - start > SLACK:
            flags.append(f"started {int((anchor - start) / 60)}m before the first request")
        if login and start - SLACK <= login and w[0]["t"] - login > SLACK:
            flags.append("idle after /login")
        X = np.zeros((2, 5))       # rows: Opus-equivalent tokens of [Opus+Fable+Haiku, Sonnet]
        for r in w:
            f = model_factor(r)
            row = 1 if (r["model"] or "").startswith("claude-sonnet") else 0
            X[row] += [r[c] * f[k] for k, c in enumerate(CATS)]
        nc = sum(1 for c in comps if eff <= c["t"] <= h["t"] and c["trigger"] == "auto")
        pre = sum(c["pre"] or 0 for c in comps if eff <= c["t"] <= h["t"] and c["trigger"] == "auto")
        out.append(dict(hit=h["t"], reset=reset, start=start, eff=eff, n=len(w), X=X,
                        compactions=nc, compacted=pre, flags=flags, label=local(h["t"])))
    return out


if __name__ == "__main__":
    from common import API_W
    for w in build():
        X = w["X"].sum(0)
        api = float(w["X"].sum(0) @ API_W)
        print(f"{w['label']}  from {local(w['eff'])[-5:]} n={w['n']:4d} c={w['compactions']}",
              " ".join(f"{c}={x/1e6:5.2f}M" for c, x in zip(CATS[1:], X[1:])),
              f"sonnet={w['X'][1] @ API_W / 1e6:4.1f}M API={api/1e6:5.1f}M", "; ".join(w["flags"]))
