"""Fit subscription usage weights to the 5-hour windows that ended in a limit hit.

Each clean window used exactly one 5-hour budget, so for weights w (budget share per token):
    X_window . w = 1.
Compaction summary requests are omitted from transcripts; `with_summaries` adds each inside
compaction's measured summary request (priced by the compacting thread's model) when enough
compactions are measured. Uncached input is a few thousand tokens per window and cannot be
identified, so it is tied to the 1h write (input = write / 2, the API ratio); the 5m write is tied
to it the same way (5m = 1h * 1.25 / 2). Free: 1h write, cache read, output, and factors for Sonnet
and Fable. Their tokens are already converted to Opus at API prices, so a factor of 1 means the
subscription prices the model relative to Opus as the API does.

`report` also prints degrees-of-freedom-adjusted RMS, leave-one-out RMS, per-window leverage, and
when Fable windows are present, a date-step diagnostic testing whether a budget step at the first
Fable window explains the data as well as a Fable factor.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import replay
from .pricing import API_W, MODELS, OPUS, PRICES, price_ratio
from .store import Store
from .windows import FABLE_SHARE, build

TIE = np.array([0.5, 0.625, 1, 0, 0])  # inp, cw5, cw1 move together
SONNET_API = PRICES["claude-sonnet-5-5"][0] / OPUS[0]  # 0.5: a Sonnet token in Opus tokens at API prices
FABLE_API = PRICES["claude-fable-5-2"][0] / OPUS[0]  # 2, except cache reads (1)
LOG_W_BOUNDS = (-30.0, 0.0)
LOG_F_BOUNDS = (-4.0, 4.0)


def with_summaries(ws: list[dict], store: Store) -> tuple[list[dict], bool]:
    """Add measured compaction summary requests to each window's token matrix."""
    meas = replay.measure_all(store)
    opus_m = meas.get(MODELS["opus"])
    if opus_m is None:
        return [dict(w, X=w["X"].copy()) for w in ws], False
    out = []
    for w in ws:
        X = w["X"].copy()
        for c in w.get("compaction_events", ()):
            pre = c.get("pre") or 168_000
            model = c.get("model") or MODELS["opus"]
            M = meas.get(model, opus_m)
            pr = price_ratio(model, M["inp"] + M["cw"] + M["rf"] * pre)
            ri = 1 if model.startswith("claude-sonnet") else 2 if model.startswith("claude-fable") else 0
            X[ri] += np.array([M["inp"], 0.0, M["cw"], M["rf"] * pre, M["out"]]) * pr
        tot = float(X.sum(0) @ API_W)
        fable = float(X[2] @ API_W / tot) if tot > 0 else w.get("fable", 0.0)
        out.append(dict(w, X=X, fable=fable))
    return out, True


def design(w, sonnet=1.0, fable=1.0):
    X = w["X"][0] + sonnet * w["X"][1]
    if len(w["X"]) > 2:
        X = X + fable * w["X"][2]
    return np.array([X @ TIE, X[3], X[4]])  # columns: write group, read, output


def _unpack(p, free_sonnet, free_fable, fixed_out_ratio):
    lw, lr = p[:2]
    if fixed_out_ratio is not None:
        lo = lw + np.log(fixed_out_ratio / 2.0)
        i = 2
    else:
        lo = p[2]
        i = 3
    s, f = 1.0, 1.0
    if free_sonnet:
        s, i = float(np.exp(p[i])), i + 1
    if free_fable:
        f = float(np.exp(p[i]))
    return np.exp([lw, lr, lo]), s, f


def resid(p, ws, free_sonnet, free_fable, fixed_out_ratio=None):
    v, s, f = _unpack(p, free_sonnet, free_fable, fixed_out_ratio)
    return np.array([np.log(design(w, s, f) @ v) for w in ws])  # log(share of budget used)


def has_fable(ws) -> bool:
    return any(w.get("fable", 0) >= FABLE_SHARE for w in ws)


def fit(ws, free_sonnet=True, fixed_out_ratio=None, free_fable=None):
    """free_fable=None fits the Fable factor when some window has a large share of Fable."""
    if free_fable is None:
        free_fable = has_fable(ws)
    n_w = 2 if fixed_out_ratio is not None else 3
    n_f = int(free_sonnet) + int(free_fable)
    x0_vals = [2e-7, 5e-9] + ([] if fixed_out_ratio is not None else [5e-7]) + [1.0] * n_f
    x0 = np.log(x0_vals)
    lo_b = [LOG_W_BOUNDS[0]] * n_w + [LOG_F_BOUNDS[0]] * n_f
    hi_b = [LOG_W_BOUNDS[1]] * n_w + [LOG_F_BOUNDS[1]] * n_f
    r = least_squares(
        resid, x0, args=(ws, free_sonnet, free_fable, fixed_out_ratio), bounds=(lo_b, hi_b), method="trf"
    )
    v, s, f = _unpack(r.x, free_sonnet, free_fable, fixed_out_ratio)
    return dict(
        write=v[0],
        read=v[1],
        out=v[2],
        sonnet=s,
        fable=f,
        free_fable=free_fable,
        rms=float(np.sqrt(np.mean(r.fun**2))),
        cost=float(r.cost),
        jac=r.jac,
        p=len(r.x),
    )


def leverage(fit_res: dict) -> np.ndarray:
    """Hat-matrix diagonal h_i from the Jacobian at the nonlinear least-squares solution."""
    J = fit_res["jac"]
    return np.diag(J @ np.linalg.pinv(J.T @ J) @ J.T)


def loo(
    ws: list[dict], free_sonnet: bool = True, free_fable: bool | None = None
) -> tuple[float, list[float]]:
    """Leave-one-out RMS (on log budget share) and per-window predicted budget shares."""
    if free_fable is None:
        free_fable = has_fable(ws)
    log_err, preds = [], []
    for i, w in enumerate(ws):
        sub = ws[:i] + ws[i + 1 :]
        fi = fit(sub, free_sonnet=free_sonnet, free_fable=free_fable)
        vi = np.array([fi["write"], fi["read"], fi["out"]])
        pred = float(design(w, fi["sonnet"], fi["fable"]) @ vi)
        log_err.append(np.log(pred))
        preds.append(pred)
    return float(np.sqrt(np.mean(np.square(log_err)))), preds


def date_step_check(ws: list[dict]) -> dict | None:
    """Compare Fable factor against a budget step starting at the first Fable-heavy window."""
    fable_hits = [w["hit"] for w in ws if w.get("fable", 0) >= FABLE_SHARE]
    if not fable_hits:
        return None
    t_step = min(fable_hits)
    is_late = np.array([w["hit"] >= t_step for w in ws])
    if is_late.all() or not is_late.any():
        return None

    def res_step(q):
        v, s, b_late = np.exp(q[:3]), float(np.exp(q[3])), float(np.exp(q[4]))
        return np.array(
            [
                np.log(design(w, s, 1.0) @ v) - (np.log(b_late) if late else 0.0)
                for w, late in zip(ws, is_late, strict=True)
            ]
        )

    def res_both(q):
        v, s, fab, b_late = np.exp(q[:3]), float(np.exp(q[3])), float(np.exp(q[4])), float(np.exp(q[5]))
        return np.array(
            [
                np.log(design(w, s, fab) @ v) - (np.log(b_late) if late else 0.0)
                for w, late in zip(ws, is_late, strict=True)
            ]
        )

    lo3, hi3 = [LOG_W_BOUNDS[0]] * 3, [LOG_W_BOUNDS[1]] * 3
    r_step = least_squares(
        res_step, np.log([2e-7, 5e-9, 5e-7, 1.0, 1.0]), bounds=(lo3 + [-4.0, -4.0], hi3 + [4.0, 4.0])
    )
    r_both = least_squares(
        res_both,
        np.log([2e-7, 5e-9, 5e-7, 1.0, 1.5, 1.0]),
        bounds=(lo3 + [-4.0, -4.0, -4.0], hi3 + [4.0, 4.0, 4.0]),
    )
    JB = r_both.jac
    cov = np.linalg.pinv(JB.T @ JB)
    corr = float(cov[4, 5] / np.sqrt(cov[4, 4] * cov[5, 5])) if cov[4, 4] > 0 and cov[5, 5] > 0 else 0.0
    return dict(
        t_step=t_step,
        step_budget=float(np.exp(r_step.x[4])),
        step_rms=float(np.sqrt(np.mean(r_step.fun**2))),
        both_fable=float(np.exp(r_both.x[4])),
        both_budget=float(np.exp(r_both.x[5])),
        both_corr=corr,
        both_rms=float(np.sqrt(np.mean(r_both.fun**2))),
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
    """Print the fits, diagnostics, profile and bootstrap; save the bootstrap for the replay."""
    raw_ws = build(store)
    all_ws, has_sum = with_summaries(raw_ws, store)
    clean = [w for w in all_ws if not w["flags"]]
    raw_clean = [w for w in raw_ws if not w["flags"]]
    print(f"{len(all_ws)} windows, {len(clean)} clean")
    if not has_sum:
        print("note: too few measured compactions to add summary requests; fitting transcripts only")
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
    base = fit(clean)
    show("free (clean)", base)
    if has_sum:
        show("free (clean, no summaries)", fit(raw_clean))
    show("free, Sonnet at API ratio", fit(clean, free_sonnet=False))
    show("free (all windows)", fit(all_ws))

    dof_rms = base["rms"] * np.sqrt(len(clean) / max(len(clean) - base["p"], 1))
    loo_rms, loo_preds = loo(clean, free_fable=base["free_fable"])
    print(
        f"\nfit error (clean, n={len(clean)}, p={base['p']}): "
        f"in-sample rms {100 * base['rms']:.2f}%, dof-adj {100 * dof_rms:.2f}%, "
        f"leave-one-out {100 * loo_rms:.2f}%"
    )
    if base["free_fable"]:
        ds = date_step_check(clean)
        if ds is not None:
            print(
                f"date-step check: budget x{ds['step_budget']:.2f} (Fable=1) rms {100 * ds['step_rms']:.2f}% "
                f"| both free: fable x{ds['both_fable']:.2f}, budget x{ds['both_budget']:.2f} "
                f"(corr {ds['both_corr']:+.2f}) rms {100 * ds['both_rms']:.2f}%"
            )

    # Profile the output weight: how well does each fixed output/input ratio fit?
    print("\nprofile of output weight (clean windows):")
    for ratio in (1.5, 2.5, 3.5, 5, 7, 10, 15):
        f = fit(clean, fixed_out_ratio=ratio)
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
    lev = leverage(base)
    clean_diag = {id(w): (lev[i], 100 * loo_preds[i]) for i, w in enumerate(clean)}
    for w in all_ws:
        used = 100 * design(w, base["sonnet"], base["fable"]) @ v
        diag = clean_diag.get(id(w))
        extra = f"lev={diag[0]:.2f} loo={diag[1]:5.1f}%" if diag is not None else "; ".join(w["flags"])
        print(f"  {w['label']}  {used:6.1f}%  {extra}")
