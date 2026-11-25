"""Fit the usage weights from the status line's rate-limit log instead of limit hits.

statusline/rate-limit-log.sh appends a line each time a session's rate limits change (the 5-hour
percentage has come in whole percent so far). Within one (5h reset, 7d reset) window, a
monotonic progression strictly below 100% measures the budget share consumed over that window.
When multiple accounts overlap in the log, requests and compaction summary requests are
attributed by (session, timestamp) to (5h reset, 7d reset); single-account windows keep all
requests in the span so unlogged headless (`claude -p`) runs are counted too.

Compared to `fit.py`, the log has enough data to free the 5-minute cache write (`cw5`) and
Fable's own `(write, read, output)` weights instead of tying them to API ratios.

    cc-compact-tuning fit-log [LOG]       fit from the log (default: the status line's, plus
                                          what collect has saved of it)
    cc-compact-tuning fit-log --synthetic check the fit on a log made from known weights
"""

import collections
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import replay, store
from .fit import LOG_F_BOUNDS, TIE
from .pricing import API_W, CATS, MODELS, local, model_factor, price_ratio, row
from .store import Store
from .windows import (
    UNATTR_SHARE_LIMIT,
    attribute_event,
    has_unexplained_jump,
    parse_rate_log,
    session_timeline,
)

LOG_W_BOUNDS = (-23.0, -5.0)
LOG_R_BOUNDS = (-27.0, -5.0)


def read_log(
    path: Path, S: Store, max_t: float | None = None
) -> list[tuple[float, str, str, float, int, str]]:
    """Lines of the log merged with those collect.py saved, sorted and capped at `max_t`."""
    lines = set(S["rate_limits"])
    if path.exists():
        with open(path) as f:
            next(f, None)
            lines |= {line.rstrip("\n") for line in f}
    rows = parse_rate_log(lines)
    if max_t is None and S["requests"]:
        max_t = max(r["t"] for r in S["requests"].values()) + 60.0
    if max_t is not None:
        rows = [r for r in rows if r[0] <= max_t]
    return rows


def log_windows(
    rows: list[tuple[float, str, str, float, int, str]],
    S: Store,
    add_summaries: bool = True,
    min_dp: float = 15.0,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict], float]:
    """Build whole-window (reset5, acct7) observations with raw per-model token matrices."""
    timeline = session_timeline(rows)
    by_win: dict[tuple[int, str], list[tuple[float, float, str]]] = collections.defaultdict(list)
    for t, sess8, _, pct5, r5, a7 in rows:
        by_win[(r5, a7)].append((t, pct5, sess8))

    R = sorted(S["requests"].values(), key=lambda r: r["t"])
    T = np.array([r["t"] for r in R])
    thread_model: dict[tuple[str, str | None], str] = {}
    sess_model: dict[str, str] = {}
    for r in R:
        if r.get("model"):
            thread_model[(r["session"], r.get("agent"))] = r["model"]
            if not r.get("side"):
                sess_model[r["session"]] = r["model"]

    meas = replay.measure_all(S) if add_summaries else {}
    opus_m = meas.get(MODELS["opus"])
    comps = sorted(S["compactions"].values(), key=lambda c: c["t"])
    CT = np.array([c["t"] for c in comps])

    win_list = []
    tot_mine_api = 0.0
    tot_unattr_api = 0.0
    n_jumps = 0

    for (r5, a7), evs in sorted(by_win.items()):
        evs.sort()
        mono: list[tuple[float, float]] = []
        mx = -1.0
        for t, p5, _ in evs:
            if mx < p5 < 100.0:
                mono.append((t, p5))
                mx = p5
        if len(mono) < 2:
            continue
        p0, p1 = mono[0][1], mono[-1][1]
        from_start = p0 <= 1.0
        t_start = (r5 - 5 * 3600) if from_start else mono[0][0]
        t_end = mono[-1][0]
        dp = p1 if from_start else (p1 - p0)
        if dp < min_dp or not len(T):
            continue
        if has_unexplained_jump(mono, R, T) is not None:
            n_jumps += 1
            continue

        # Did any other (reset5, acct7) overlap [t_start - 60, t_end + 60]?
        other_active = any(
            (r5_o != r5 or a7_o != a7) and t_start - 60 <= t_o <= t_end + 60
            for t_o, _, _, _, r5_o, a7_o in rows
        )

        i_start = np.searchsorted(T, t_start, side="left" if from_start else "right")
        w_reqs = R[i_start : np.searchsorted(T, t_end, side="right")]
        c_start = np.searchsorted(CT, t_start, side="left" if from_start else "right") if len(CT) else 0
        w_comps = comps[c_start : np.searchsorted(CT, t_end, side="right")] if len(CT) else []

        if other_active:
            mine_reqs, unattr_reqs = [], []
            for r in w_reqs:
                attr = attribute_event(r["t"], r["session"], timeline)
                if attr == (r5, a7):
                    mine_reqs.append(r)
                elif attr is None:
                    unattr_reqs.append(r)
            mine_api = sum(
                float(np.array([r[c] for c in CATS]) @ (np.array(model_factor(r)) * API_W)) for r in mine_reqs
            )
            unattr_api = sum(
                float(np.array([r[c] for c in CATS]) @ (np.array(model_factor(r)) * API_W))
                for r in unattr_reqs
            )
            tot_mine_api += mine_api
            tot_unattr_api += unattr_api
            denom = mine_api + unattr_api
            if denom > 0 and unattr_api / denom > UNATTR_SHARE_LIMIT:
                continue
            w_reqs = mine_reqs
            w_comps = [c for c in w_comps if attribute_event(c["t"], c["session"], timeline) == (r5, a7)]

        X = np.zeros((3, 5))
        for r in w_reqs:
            ri = row(r)
            if ri == 0:
                f = model_factor(r)
                X[0] += [r[c] * f[k] for k, c in enumerate(CATS)]
            else:
                sp = 2.0 if r.get("speed") == "fast" else 1.0
                X[ri] += [r[c] * sp for c in CATS]

        if opus_m is not None:
            for c in w_comps:
                pre = c.get("pre") or 168_000
                m = (
                    thread_model.get((c["session"], c.get("agent")))
                    or sess_model.get(c["session"])
                    or MODELS["opus"]
                )
                M = meas.get(m, opus_m)
                vec = np.array([M["inp"], 0.0, M["cw"], M["rf"] * pre, M["out"]])
                ri = 1 if m.startswith("claude-sonnet") else 2 if m.startswith("claude-fable") else 0
                if ri == 0:
                    pr = price_ratio(m, M["inp"] + M["cw"] + M["rf"] * pre)
                    X[0] += vec * pr
                else:
                    X[ri] += vec

        if X.sum() <= 0:
            continue
        win_list.append(dict(reset=r5, acct=a7, dp=dp, X=X, p0=p0, p1=p1))

    tot_api = tot_mine_api + tot_unattr_api
    unattr_share = tot_unattr_api / tot_api if tot_api > 0 else 0.0
    if stats is not None:
        stats["jumps"] = n_jumps
    return win_list, unattr_share


def fit(W: list[dict], free_cw5: bool = True, fable_mode: str = "free") -> dict:
    """Fit Opus weights (normalized to cw1 = 2), Sonnet factor, and Fable weights from `W`.

    `fable_mode`:
      - "api": single factor multiplying Fable's API ratio [2, 2, 2, 1, 2]
      - "flat": single factor multiplying Opus weights across all categories
      - "free": independent Fable (write, read, output) weights (downgraded to "flat" if <3 Fable windows)
    """
    dp = np.array([w["dp"] for w in W], float)
    D = np.array([w["X"] for w in W], float)
    tot_cw1 = D[:, :, 2].sum()
    tot_cw5 = D[:, :, 1].sum()
    has_cw5 = free_cw5 and tot_cw1 > 0 and (tot_cw5 / tot_cw1 > 0.01)
    has_son = bool(D[:, 1].any())
    n_fab_wins = int(sum(1 for w in W if w["X"][2].any()))
    has_fab = n_fab_wins > 0
    eff_mode = "flat" if (fable_mode == "free" and n_fab_wins < 3) else fable_mode

    tie_in = np.array([0.5, 0.0, 1.0, 0.0, 0.0])  # inp tied to cw1

    def build_cols(row_idx: int) -> np.ndarray:
        X = D[:, row_idx]
        if has_cw5:
            return np.stack([X @ tie_in, X[:, 1], X[:, 3], X[:, 4]], 1)
        return np.stack([X @ TIE, X[:, 3], X[:, 4]], 1)

    A, B, C = build_cols(0), build_cols(1), build_cols(2)
    n_base = 4 if has_cw5 else 3
    n_fab_p = (3 if eff_mode == "free" else 1) if has_fab else 0

    def unpack(q):
        v = np.exp(q[:n_base])
        idx = n_base
        s = float(np.exp(q[idx])) if has_son else 1.0
        idx += int(has_son)
        if not has_fab:
            return v, s, None
        if eff_mode == "free":
            lw1_f, lr_f, lo_f = q[idx : idx + 3]
            w1_f = float(np.exp(lw1_f))
            if has_cw5:
                w5_f = w1_f * float(v[1] / v[0])
                vf = np.array([w1_f, w5_f, float(np.exp(lr_f)), float(np.exp(lo_f))])
            else:
                vf = np.array([w1_f, float(np.exp(lr_f)), float(np.exp(lo_f))])
            return v, s, vf
        return v, s, float(np.exp(q[idx]))

    def res(q):
        v, s, fab = unpack(q)
        base = A + (s * 0.5 * B if has_son else 0.0)
        if not has_fab:
            tot = base @ v
        elif eff_mode == "free":
            tot = base @ v + C @ fab
        elif eff_mode == "flat":
            tot = (base + fab * C) @ v
        else:
            api_scale = np.array([2.0, 2.0, 1.0, 2.0]) if has_cw5 else np.array([2.0, 1.0, 2.0])
            tot = (base + fab * (C * api_scale)) @ v
        return 100.0 * tot - dp

    q0 = list(np.log([2e-7, 1.25e-7, 5e-9, 5e-7] if has_cw5 else [2e-7, 5e-9, 5e-7]))
    if has_cw5:
        lo_b = [LOG_W_BOUNDS[0], LOG_W_BOUNDS[0], LOG_R_BOUNDS[0], LOG_W_BOUNDS[0]]
        hi_b = [LOG_W_BOUNDS[1], LOG_W_BOUNDS[1], LOG_R_BOUNDS[1], LOG_W_BOUNDS[1]]
    else:
        lo_b = [LOG_W_BOUNDS[0], LOG_R_BOUNDS[0], LOG_W_BOUNDS[0]]
        hi_b = [LOG_W_BOUNDS[1], LOG_R_BOUNDS[1], LOG_W_BOUNDS[1]]
    if has_son:
        q0.append(0.0)
        lo_b.append(LOG_F_BOUNDS[0])
        hi_b.append(LOG_F_BOUNDS[1])
    if has_fab:
        if eff_mode == "free":
            q0.extend(np.log([6e-7, 2e-8, 2e-6]))
            lo_b.extend([LOG_W_BOUNDS[0], LOG_R_BOUNDS[0], LOG_W_BOUNDS[0]])
            hi_b.extend([LOG_W_BOUNDS[1], LOG_R_BOUNDS[1], LOG_W_BOUNDS[1]])
        else:
            q0.append(np.log(3.5 if eff_mode == "flat" else 1.8))
            lo_b.append(LOG_F_BOUNDS[0])
            hi_b.append(LOG_F_BOUNDS[1])

    r = least_squares(res, q0, bounds=(lo_b, hi_b), loss="soft_l1", f_scale=1.0)
    v, s, fab = unpack(r.x)
    k = 2.0 / float(v[0])
    cw5_w = float(k * v[1]) if has_cw5 else 1.25
    read_w = float(k * v[2 if has_cw5 else 1])
    out_w = float(k * v[3 if has_cw5 else 2])
    raw_err = res(r.x)

    out = dict(
        cw5=cw5_w,
        read=read_w,
        out=out_w,
        budget=k,
        cw5_ratio=cw5_w / 2.0,
        sonnet=s if has_son else float("nan"),
        fable_mode=eff_mode,
        rms=float(np.sqrt(np.mean(raw_err**2))),
        med_abs=float(np.median(np.abs(raw_err))),
        n=len(W),
        p=n_base + int(has_son) + n_fab_p,
    )
    if not has_fab:
        out |= dict(
            fable=float("nan"),
            fable_read=float("nan"),
            fable_out=float("nan"),
            fable_ratios=(float("nan"), float("nan"), float("nan")),
            fable_rw_vs_opus=float("nan"),
        )
    elif eff_mode == "free":
        kf = 2.0 / float(fab[0])
        r_idx, o_idx = (2, 3) if has_cw5 else (1, 2)
        f_read = float(kf * fab[r_idx])
        f_out = float(kf * fab[o_idx])
        fw = float(fab[0] / v[0])
        fr = float(fab[r_idx] / v[r_idx])
        fo = float(fab[o_idx] / v[o_idx])
        out |= dict(
            fable=fw,
            fable_read=f_read,
            fable_out=f_out,
            fable_ratios=(fw, fr, fo),
            fable_rw_vs_opus=(f_read / 2.0) / (read_w / 2.0),
        )
    elif eff_mode == "flat":
        out |= dict(
            fable=float(fab),
            fable_read=read_w,
            fable_out=out_w,
            fable_ratios=(float(fab), float(fab), float(fab)),
            fable_rw_vs_opus=1.0,
        )
    else:
        out |= dict(
            fable=float(fab),
            fable_read=read_w / 2.0,
            fable_out=out_w,
            fable_ratios=(2.0 * float(fab), float(fab), 2.0 * float(fab)),
            fable_rw_vs_opus=0.5,
        )
    return out


def synthetic(
    S: Store, w_true: dict | None = None
) -> tuple[list[tuple[float, str, str, float, int, str]], Store]:
    """Build a 2-account synthetic log and matching store from clean/Fable hit windows.

    Windows are placed onto clean non-colliding timestamps with two concurrent accounts (plus a
    mid-session account switch) and capped below 95% so no window saturates or bleeds across hits.
    """
    from .windows import build

    if w_true is None:
        w_true = dict(budget=9.5e6, cw5=1.9, read=0.036, out=7.0, sonnet=1.6, fable=(7.0, 0.108, 24.5))
    bud = w_true["budget"]
    v_opus = np.array([1.0, w_true["cw5"], 2.0, w_true["read"], w_true["out"]]) / bud
    fw, fr, fo = w_true["fable"]
    v_fab = np.array([0.5 * fw, fw * (w_true["cw5"] / 2.0), fw, fr, fo]) / bud
    s_fac = w_true["sonnet"]

    R = sorted(S["requests"].values(), key=lambda r: r["t"])
    rows: list[tuple[float, str, str, float, int, str]] = []
    S_syn = store.empty()
    req_id = 0
    ws = [w for w in build(S) if not any("Fable" not in fl for fl in w["flags"])]
    for k, w in enumerate(ws):
        acct = "acct-a" if k % 2 == 0 else "acct-b"
        t_base = 100_000.0 + k * 10_000.0 + (1800.0 if k % 2 else 0.0)
        reset_int = int(t_base + 18_000.0)
        w_reqs = [r for r in R if w["eff"] <= r["t"] <= w["hit"]]
        dt_step = min(30.0, 15_000.0 / max(len(w_reqs), 1))
        pct = 0.0
        for j, r in enumerate(w_reqs):
            ri = row(r)
            if ri == 0:
                f = model_factor(r)
                x = np.array([r[c] * f[i] for i, c in enumerate(CATS)])
                delta = 100.0 * float(x @ v_opus)
            elif ri == 1:
                sp = 2.0 if r.get("speed") == "fast" else 1.0
                x = np.array([r[c] * sp for c in CATS])
                delta = 100.0 * s_fac * 0.5 * float(x @ v_opus)
            else:
                sp = 2.0 if r.get("speed") == "fast" else 1.0
                x = np.array([r[c] * sp for c in CATS])
                delta = 100.0 * float(x @ v_fab)
            if pct + delta >= 95.0:
                break
            pct += delta
            t_req = t_base + dt_step * (j + 1)
            sess = f"syn{k:02d}-{j % 3}-sess"
            S_syn["requests"][str(req_id)] = dict(r, t=t_req, session=sess)
            req_id += 1
            rows.append((t_req + 1.0, sess[:8], r["model"] or "opus", round(pct), reset_int, acct))
    if S_syn["requests"]:
        # Simulate a mid-session account switch on syn00-0-sess in a much later window.
        rows.append((10_000_000.0, "syn00-0-", "opus", 12.0, 10_018_000, "acct-other"))
    return sorted(rows), S_syn


def _show_fit(label: str, f: dict) -> None:
    print(
        f"{label:28s} cw5={f['cw5']:.2f} read={f['read']:.4f} out={f['out']:.2f} "
        f"bud={f['budget'] / 1e6:.2f}M | son=x{f['sonnet']:.2f} | "
        f"fable: w={f['fable_ratios'][0]:.2f}x r={f['fable_ratios'][1]:.2f}x o={f['fable_ratios'][2]:.2f}x "
        f"(own read={f['fable_read']:.4f} out={f['fable_out']:.2f}, rw/opus={f['fable_rw_vs_opus']:.2f}) | "
        f"rms={f['rms']:.2f}pp med={f['med_abs']:.2f}pp"
    )


def report(S: Store, data_dir: Path, log: Path, synthetic_log: bool = False) -> None:
    """Fit whole windows from the rate-limit log, bootstrap by window, and save `bootstrap_log.npz`."""
    if synthetic_log:
        w_true = dict(budget=9.5e6, cw5=1.9, read=0.036, out=7.0, sonnet=1.6, fable=(7.0, 0.108, 24.5))
        rows, S_syn = synthetic(S, w_true)
        print(
            f"synthetic log (2 concurrent accounts): {len(rows)} lines; "
            f"true cw5=1.90, read=0.036, out=7.0, son=x1.6, "
            f"fable=(w=3.5x, r=3.0x, o=3.5x, own read=0.0309, own out=7.0), bud=9.5M"
        )
        W, unattr = log_windows(rows, S_syn, add_summaries=False, min_dp=15.0)
    else:
        all_rows = read_log(log, S, max_t=np.inf)
        rows = read_log(log, S)
        skipped = len(all_rows) - len(rows)
        skip_note = f" ({skipped} lines after last collected request skipped)" if skipped else ""
        print(f"{log}: {len(rows)} lines from {local(rows[0][0]) if rows else '-'}{skip_note}")
        st_info: dict[str, int] = {}
        W, unattr = log_windows(rows, S, add_summaries=True, min_dp=15.0, stats=st_info)
        jump_note = f", {st_info['jumps']} skipped (unexplained jump)" if st_info.get("jumps") else ""
        print(
            f"{len(W)} whole windows (dp>=15%){jump_note}, "
            f"unattributed token share in overlaps: {100 * unattr:.2f}%"
        )

    if len(W) < 3:
        print("not enough data yet")
        return

    f_api = fit(W, free_cw5=False, fable_mode="api")
    f_flat = fit(W, free_cw5=True, fable_mode="flat")
    f_free = fit(W, free_cw5=True, fable_mode="free")
    _show_fit("API-ratio Fable (cw5 tied)", f_api)
    _show_fit("flat Fable + free cw5", f_flat)
    _show_fit("free Fable + free cw5", f_free)

    rng = np.random.default_rng(0)
    B_opus, B_fab = [], []
    vals = {
        k: []
        for k in ("cw5", "read", "out", "sonnet", "fable_w", "fable_r", "fable_o", "fable_read", "fable_out")
    }
    for _ in range(500):
        Wb = [W[i] for i in rng.integers(0, len(W), len(W))]
        g = fit(Wb, free_cw5=True, fable_mode="free")
        B_opus.append((g["cw5"], g["read"], g["out"]))
        vals["cw5"].append(g["cw5"])
        vals["read"].append(g["read"])
        vals["out"].append(g["out"])
        vals["sonnet"].append(g["sonnet"])
        vals["fable_w"].append(g["fable_ratios"][0])
        vals["fable_r"].append(g["fable_ratios"][1])
        vals["fable_o"].append(g["fable_ratios"][2])
        vals["fable_read"].append(g["fable_read"])
        vals["fable_out"].append(g["fable_out"])
        if not np.isnan(g["fable_read"]):
            B_fab.append((g["cw5"], g["fable_read"], g["fable_out"]))

    print("\nbootstrap 10/50/90% (free Fable + free cw5, missing-model resamples excluded):")
    for label_k, fmt in (
        ("cw5", ".2f"),
        ("read", ".4f"),
        ("out", ".2f"),
        ("sonnet", ".2f"),
        ("fable_w", ".2f"),
        ("fable_r", ".2f"),
        ("fable_o", ".2f"),
        ("fable_read", ".4f"),
        ("fable_out", ".2f"),
    ):
        arr = np.asarray(vals[label_k], float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        q = np.percentile(valid, [10, 50, 90])
        print(f"  {label_k:12s} " + " / ".join(f"{x:{fmt}}" for x in q))

    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        data_dir / "bootstrap_log.npz",
        opus_w=np.array(B_opus, float),
        fable_w=np.array(B_fab, float) if B_fab else np.empty((0, 3), float),
    )
    # Backward-compatible 7-column npy
    np.save(
        data_dir / "bootstrap_log.npy",
        np.array(
            [
                (r, o, 0.0, 0.0, s, 0.0, fw)
                for r, o, s, fw in zip(
                    vals["read"], vals["out"], vals["sonnet"], vals["fable_w"], strict=True
                )
            ]
        ),
    )
