"""Fit subscription usage weights to the 5-hour windows that ended in a limit hit.

Each clean window used exactly one 5-hour budget, so for weights w (budget share per token):
    X_window . w = 1.
Uncached input is a few thousand tokens per window and cannot be identified, so it is tied to
the 1h write (input = write / 2, the API ratio); the 5m write is tied to it the same way
(5m = 1h * 1.25 / 2). Free: 1h write, cache read, output, and how much a Sonnet token counts
relative to an Opus one. Weights are reported relative to the 1h write = 2 (its API weight) and,
for comparison with the October fit, relative to output = 5.
"""
import sys

import numpy as np
from scipy.optimize import least_squares

from common import API_W
from windows import build

TIE = np.array([0.5, 0.625, 1, 0, 0])    # inp, cw5, cw1 move together


def design(w, sonnet):
    X = w["X"][0] + sonnet * w["X"][1]
    return np.array([X @ TIE, X[3], X[4]])          # columns: write group, read, output


def resid(p, ws, free_sonnet, fixed=None):
    lw, lr, lo = p[:3]
    if fixed is not None:
        lo = fixed
    s = np.exp(p[3]) if free_sonnet else 0.5
    v = np.exp([lw, lr, lo])
    return np.array([np.log(design(w, s) @ v) for w in ws])     # log(share of budget used)


def fit(ws, free_sonnet=True, fixed=None):
    x0 = np.log([2e-7, 5e-9, 5e-7, 1.0])
    r = least_squares(resid, x0, args=(ws, free_sonnet, fixed), method="trf")
    lw, lr, lo = r.x[:3]
    if fixed is not None:
        lo = fixed
    s = float(np.exp(r.x[3])) if free_sonnet else 0.5
    return dict(write=np.exp(lw), read=np.exp(lr), out=np.exp(lo), sonnet=s,
                rms=float(np.sqrt(np.mean(r.fun ** 2))), cost=float(r.cost))


def rel(f, anchor="write"):
    k = 2 / f["write"] if anchor == "write" else 5 / f["out"]
    return k * f["write"], k * f["read"], k * f["out"]


def show(name, f):
    a, b, c = rel(f)
    d, e, g = rel(f, "out")
    budget = 2 / f["write"]          # input-token units per 5h with write = 2
    print(f"{name:34s} write=2: read {b:.4f} out {c:5.2f} | out=5: write {d:.2f} read {e:.4f} | "
          f"sonnet x{f['sonnet']:.2f} | rms {100*f['rms']:.1f}% | budget {budget/1e6:.1f}M")


def main():
    all_ws = build()
    clean = [w for w in all_ws if not w["flags"]]
    print(f"{len(all_ws)} windows, {len(clean)} clean")
    api = np.array(API_W)
    for name, ws in (("clean", clean), ("all", all_ws)):
        t = np.array([(w["X"][0] + 0.5 * w["X"][1]) @ api for w in ws])
        t2 = np.array([(w["X"][0] + w["X"][1]) @ api for w in ws])
        print(f"API weights, {name}: CV {100*t.std()/t.mean():.1f}% (Sonnet at API price), "
              f"{100*t2.std()/t2.mean():.1f}% (Sonnet at Opus price), mean {t2.mean()/1e6:.2f}M")
    print()
    show("free (clean)", fit(clean))
    show("free, Sonnet at API ratio", fit(clean, free_sonnet=False))
    show("free (all windows)", fit(all_ws))

    # Profile the output weight: how well does each fixed output/write ratio fit?
    base = fit(clean)
    print("\nprofile of output weight (clean windows):")
    for ratio in (1.5, 2.5, 3.5, 5, 7, 10, 15):
        f = fit(clean, fixed=np.log(base["write"] * ratio / 2))
        a, b, c = rel(f)
        print(f"  out/input {c:5.1f}: read {b:.4f}, rms {100*f['rms']:.2f}%")

    # Bootstrap over windows.
    rng = np.random.default_rng(0)
    B = []
    for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 1000):
        ws = [clean[i] for i in rng.integers(0, len(clean), len(clean))]
        f = fit(ws)
        B.append((*rel(f)[1:], *rel(f, "out")[:2], f["sonnet"], 2 / f["write"]))
    B = np.array(B)
    print("\nbootstrap 10/50/90%:")
    for i, n in enumerate(["read (write=2)", "output (write=2)", "write (out=5)", "read (out=5)",
                           "sonnet factor", "budget M"]):
        q = np.percentile(B[:, i], [10, 50, 90])
        print(f"  {n:18s} {q[0]:.4g} {q[1]:.4g} {q[2]:.4g}" if i != 5 else
              f"  {n:18s} {q[0]/1e6:.2f} {q[1]/1e6:.2f} {q[2]/1e6:.2f}")
    np.save("data/bootstrap.npy", B)

    print("\nper-window residuals, free fit:")
    f = fit(clean)
    v = np.array([f["write"], f["read"], f["out"]])
    for w in all_ws:
        print(f"  {w['label']}  {100 * design(w, f['sonnet']) @ v:6.1f}%  {'; '.join(w['flags'])}")


if __name__ == "__main__":
    main()
