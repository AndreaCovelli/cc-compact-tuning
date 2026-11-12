"""Fit the usage weights from the status line's rate-limit log instead of limit hits.

~/.claude/statusline.sh appends a line each time a session's 5-hour percentage changes (it has
come in whole percent so far). Within one 5-hour window and one account, the rise in percentage
between two log lines is the budget share of the requests in between, so every window gives many
points rather than one hit. Segments are cut at each /login, since the next lines may belong to
another account.

    cc-compact-tuning fit-log [LOG]       fit from the log (default: the status line's, plus
                                          what collect has saved of it)
    cc-compact-tuning fit-log --synthetic check the fit on a log made from known weights
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .fit import TIE
from .pricing import CATS, local, model_factor, row
from .store import Store


def read_log(path: Path, S: Store) -> list[tuple[float, float, int, str]]:
    """Lines of the log merged with those collect.py saved before the status line pruned them."""
    lines = set(S["rate_limits"])
    if path.exists():
        with open(path) as f:
            next(f, None)
            lines |= {line.rstrip("\n") for line in f}
    rows = []
    for line in lines:
        p = line.split("\t")
        if len(p) < 7 or not p[3] or not p[4]:
            continue
        rows.append((float(p[0]), float(p[3]), int(float(p[4])), p[6]))
    return sorted(rows)


def segments(rows, logins):
    """[(window reset, account, [(t, pct), ...])] with no /login inside a segment."""
    seg = {}
    for t, pct, reset, acct in rows:
        k = sum(1 for lg in logins if lg < t)  # login epoch
        seg.setdefault((reset, acct, k), []).append((t, pct))
    return [(r, a, sorted(v)) for (r, a, _), v in sorted(seg.items()) if len(v) > 1]


def features(S):
    R = sorted(S["requests"].values(), key=lambda r: r["t"])
    T = np.array([r["t"] for r in R])
    F = np.zeros((len(R) + 1, 3, 5))  # cumulative Opus-equivalent tokens, [other, sonnet, fable]
    for i, r in enumerate(R):
        f = model_factor(r)
        F[i + 1] = F[i]
        F[i + 1, row(r)] += [r[c] * f[k] for k, c in enumerate(CATS)]
    return T, F


def points(segs, T, F):
    """One point per log line after the first of its segment: (rise in %, tokens since)."""
    out = []
    for reset, _acct, v in segs:
        t0, p0 = v[0]
        i0 = np.searchsorted(T, t0, side="right")
        for t, p in v[1:]:
            i = np.searchsorted(T, t, side="right")
            out.append((reset, p - p0, F[i] - F[i0]))
    return out


def fit(P):
    """Weights relative to the 1h write = 2, and the Sonnet and Fable factors on top of their API
    price ratio; a factor stays at 1 when no point has tokens of that model."""
    dp = np.array([p[1] for p in P])
    D = np.array([p[2] for p in P])  # points x [other, sonnet, fable] x categories
    A, B, C = (np.stack([X @ TIE, X[:, 3], X[:, 4]], 1) for X in (D[:, 0], D[:, 1], D[:, 2]))
    free_s, free_f = bool(B.any()), bool(C.any())

    def unpack(q):
        s = np.exp(q[3]) if free_s else 1.0
        f = np.exp(q[3 + free_s]) if free_f else 1.0
        return np.exp(q[:3]), float(s), float(f)

    def res(q):
        v, s, f = unpack(q)
        return 100 * (A + s * B + f * C) @ v - dp

    x0 = np.log([2e-7, 5e-9, 5e-7] + [1.0] * (free_s + free_f))
    r = least_squares(res, x0, loss="soft_l1", f_scale=1.0)
    v, s, f = unpack(r.x)
    k = 2 / v[0]
    return dict(
        read=k * v[1], out=k * v[2], sonnet=s, fable=f, budget=k, rms=float(np.sqrt(np.mean(r.fun**2)))
    )


def synthetic(S, w_true):
    """A log as the status line would write it, from the hit windows and known weights
    (write, read, output, Sonnet factor, Fable factor). Windows flagged only for their Fable
    share are kept: the log is what can measure Fable."""
    from .windows import build

    v = np.array([w_true[0], w_true[1], w_true[2]])
    rows = []
    R = sorted(S["requests"].values(), key=lambda r: r["t"])
    for w in build(S):
        if any("Fable" not in fl for fl in w["flags"]):
            continue
        pct = 0.0
        for r in R:
            if not w["eff"] <= r["t"] <= w["hit"]:
                continue
            f = model_factor(r)
            x = np.array([r[c] * f[k] for k, c in enumerate(CATS)])
            x *= (1, w_true[3], w_true[4])[row(r)]
            pct += 100 * np.array([x @ TIE, x[3], x[4]]) @ v
            rows.append((r["t"] + 2, round(pct, 1), int(w["reset"]), "acct"))
    return rows


def report(S: Store, data_dir: Path, log: Path, synthetic_log: bool = False) -> None:
    """Fit the log (or a synthetic one), bootstrap by window, and save the bootstrap."""
    logins = sorted(lg["t"] for lg in S["logins"].values())
    if synthetic_log:
        budget = 9e6  # input-token units per 5h, write = 2
        true = (2 / budget, 0.025 / budget, 5.5 / budget, 1.8, 2.3)
        rows = synthetic(S, true)
        print(
            f"synthetic log: {len(rows)} lines; true read 0.025, output 5.5, sonnet x1.8, fable x2.3,"
            " budget 9.0M"
        )
    else:
        rows = read_log(log, S)
        print(f"{log}: {len(rows)} lines from {local(rows[0][0]) if rows else '-'}")
    segs = segments(rows, logins)
    T, F = features(S)
    P = points(segs, T, F)
    print(f"{len(segs)} window segments, {len(P)} points")
    if len(segs) < 3:
        print("not enough data yet")
        return
    f = fit(P)
    print(
        f"fit (write = 2): read {f['read']:.4f}, output {f['out']:.2f}, sonnet x{f['sonnet']:.2f},"
        f" fable x{f['fable']:.2f},"
        f" budget {f['budget'] / 1e6:.2f}M per 5h, rms {f['rms']:.2f} points of %"
    )
    # Bootstrap by window segment.
    rng = np.random.default_rng(0)
    by = {}
    for p in P:
        by.setdefault(p[0], []).append(p)
    groups = list(by.values())
    B = []
    for _ in range(200):
        Pb = [p for i in rng.choice(len(groups), len(groups)) for p in groups[i]]
        g = fit(Pb)
        B.append((g["read"], g["out"], g["sonnet"], g["fable"]))
    q = np.percentile(B, [10, 50, 90], axis=0)
    print(
        "bootstrap 10/50/90%: read "
        + " / ".join(f"{x:.4f}" for x in q[:, 0])
        + ", output "
        + " / ".join(f"{x:.2f}" for x in q[:, 1])
        + ", sonnet "
        + " / ".join(f"{x:.2f}" for x in q[:, 2])
        + ", fable "
        + " / ".join(f"{x:.2f}" for x in q[:, 3])
        + " (a resample without a model's tokens leaves its factor at 1)"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "bootstrap_log.npy", np.array([(r, o, 0, 0, s, 0, f) for r, o, s, f in B]))
