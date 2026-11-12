"""Fit subscription usage weights to the 5-hour windows that ended in a limit hit.

Each clean window used exactly one 5-hour budget, so for weights w (budget share per token):
    X_window . w = 1.
Uncached input is a few thousand tokens per window and cannot be identified, so it is tied to
the 1h write (input = write / 2, the API ratio); the 5m write is tied to it the same way
(5m = 1h * 1.25 / 2). Free: 1h write, cache read, output, and factors for Sonnet and Fable.
Their tokens are already converted to Opus at API prices, so a factor of 1 means the
subscription prices the model relative to Opus as the API does. The Fable factor is fitted
only once enough clean windows use Fable (windows.MIN_FABLE_WINDOWS); until then it is 1 and
the windows that use Fable are flagged. Weights are reported relative to the 1h write = 2
(its API weight) and, for comparison with the October fit, relative to output = 5.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .pricing import API_W, OPUS, PRICES
from .store import Store
from .windows import FABLE_SHARE, build

TIE = np.array([0.5, 0.625, 1, 0, 0])  # inp, cw5, cw1 move together
SONNET_API = PRICES["claude-sonnet-5-5"][0] / OPUS[0]  # 0.5: a Sonnet token in Opus tokens at API prices
FABLE_API = PRICES["claude-fable-5-2"][0] / OPUS[0]  # 2, except cache reads (1)


def design(w, sonnet=1.0, fable=1.0):
    X = w["X"][0] + sonnet * w["X"][1]
    if len(w["X"]) > 2:
        X = X + fable * w["X"][2]
    return np.array([X @ TIE, X[3], X[4]])  # columns: write group, read, output


def _unpack(p, free_sonnet, free_fable, fixed):
    lw, lr, lo = p[:3]
    if fixed is not None:
        lo = fixed
    i, s, f = 3, 1.0, 1.0
    if free_sonnet:
        s, i = float(np.exp(p[i])), i + 1
    if free_fable:
        f = float(np.exp(p[i]))
    return np.exp([lw, lr, lo]), s, f


def resid(p, ws, free_sonnet, free_fable, fixed=None):
    v, s, f = _unpack(p, free_sonnet, free_fable, fixed)
    return np.array([np.log(design(w, s, f) @ v) for w in ws])  # log(share of budget used)


def has_fable(ws) -> bool:
    return any(w.get("fable", 0) >= FABLE_SHARE for w in ws)


def fit(ws, free_sonnet=True, fixed=None, free_fable=None):
    """free_fable=None fits the Fable factor when some window has a large share of Fable."""
    if free_fable is None:
        free_fable = has_fable(ws)
    x0 = np.log([2e-7, 5e-9, 5e-7] + [1.0] * (free_sonnet + free_fable))
    r = least_squares(resid, x0, args=(ws, free_sonnet, free_fable, fixed), method="trf")
    v, s, f = _unpack(r.x, free_sonnet, free_fable, fixed)
    return dict(
        write=v[0],
        read=v[1],
        out=v[2],
        sonnet=s,
        fable=f,
        free_fable=free_fable,
        rms=float(np.sqrt(np.mean(r.fun**2))),
        cost=float(r.cost),
    )


def rel(f, anchor="write"):
    k = 2 / f["write"] if anchor == "write" else 5 / f["out"]
    return k * f["write"], k * f["read"], k * f["out"]


def show(name, f):
    a, b, c = rel(f)
    d, e, g = rel(f, "out")
    budget = 2 / f["write"]  # input-token units per 5h with write = 2
    fable = f"fable x{f['fable']:.2f}" if f["free_fable"] else "fable x1 (fixed)"
    print(
        f"{name:30s} write=2: read {b:.4f} out {c:5.2f} | out=5: write {d:.2f} read {e:.4f} | "
        f"sonnet x{f['sonnet']:.2f} = {SONNET_API * f['sonnet']:.2f} Opus | {fable} | "
        f"rms {100 * f['rms']:.1f}% | budget {budget / 1e6:.1f}M"
    )


def report(store: Store, data_dir: Path, n_boot: int = 1000) -> None:
    """Print the fits, profile and bootstrap; save the bootstrap for the replay."""
    all_ws = build(store)
    clean = [w for w in all_ws if not w["flags"]]
    print(f"{len(all_ws)} windows, {len(clean)} clean")
    api = np.array(API_W)
    for name, ws in (("clean", clean), ("all", all_ws)):
        t = np.array([w["X"].sum(0) @ api for w in ws])
        t2 = np.array([(w["X"].sum(0) + w["X"][1]) @ api for w in ws])
        print(
            f"API weights, {name}: CV {100 * t.std() / t.mean():.1f}% (Sonnet at API price), "
            f"{100 * t2.std() / t2.mean():.1f}% (Sonnet at Opus price), mean {t.mean() / 1e6:.2f}M"
        )
    print(
        "\nfactors multiply the API price ratio: Sonnet x1 = 0.5 Opus tokens; "
        "Fable x1 = 2 Opus tokens (1 for cache reads)"
    )
    show("free (clean)", fit(clean))
    show("free, Sonnet at API ratio", fit(clean, free_sonnet=False))
    show("free (all windows)", fit(all_ws))

    # Profile the output weight: how well does each fixed output/write ratio fit?
    base = fit(clean)
    print("\nprofile of output weight (clean windows):")
    for ratio in (1.5, 2.5, 3.5, 5, 7, 10, 15):
        f = fit(clean, fixed=np.log(base["write"] * ratio / 2))
        a, b, c = rel(f)
        print(f"  out/input {c:5.1f}: read {b:.4f}, rms {100 * f['rms']:.2f}%")

    # Bootstrap over windows; the Fable factor is free or fixed as in the full fit.
    rng = np.random.default_rng(0)
    B = []
    for _ in range(n_boot):
        ws = [clean[i] for i in rng.integers(0, len(clean), len(clean))]
        f = fit(ws, free_fable=base["free_fable"])
        B.append((*rel(f)[1:], *rel(f, "out")[:2], f["sonnet"], 2 / f["write"], f["fable"]))
    B = np.array(B)
    print("\nbootstrap 10/50/90%:")
    names = ["read (write=2)", "output (write=2)", "write (out=5)", "read (out=5)", "sonnet factor"]
    names += ["budget M", "fable factor"]
    for i, n in enumerate(names):
        q = np.percentile(B[:, i], [10, 50, 90]) / (1e6 if n == "budget M" else 1)
        print(f"  {n:18s} {q[0]:.4g} {q[1]:.4g} {q[2]:.4g}")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "bootstrap.npy", B)

    print("\nper-window share of a budget, free fit on clean windows:")
    v = np.array([base["write"], base["read"], base["out"]])
    for w in all_ws:
        used = 100 * design(w, base["sonnet"], base["fable"]) @ v
        print(f"  {w['label']}  {used:6.1f}%  {'; '.join(w['flags'])}")
