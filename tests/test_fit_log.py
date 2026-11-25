import numpy as np
import pytest
from conftest import request

from cc_compact_tuning import fit_log
from cc_compact_tuning.pricing import CATS, model_factor


def _line(t: float, sess: str, model: str, pct: float, r5: int, acct: str) -> str:
    return "\t".join([str(t), sess[:8], model, str(pct), str(r5), "10", acct])


def test_read_log_merges_saved_lines_skips_incomplete_and_future(tmp_path, empty_store):
    empty_store["requests"]["r0"] = request(20.0)
    empty_store["rate_limits"][_line(10, "s1", "opus", 40, 500, "900")] = 1
    log = tmp_path / "rate.tsv"
    log.write_text(
        "header\n"
        + _line(20, "s1", "opus", 42, 500, "900")
        + "\n"
        + _line(200, "s1", "opus", 90, 500, "900")  # > max_req_t + 60: skipped
        + "\n20\ts\tm\t\t\t\t\n"
    )
    assert fit_log.read_log(log, empty_store) == [
        (10.0, "s1", "opus", 40.0, 500, "900"),
        (20.0, "s1", "opus", 42.0, 500, "900"),
    ]


def test_fit_recovers_weights_from_two_concurrent_accounts_with_cw5_and_free_fable(empty_store):
    rng = np.random.default_rng(2)
    bud = 9e6
    cw5_w, read_w, out_w = 1.9, 0.036, 7.2
    son_fac = 1.6
    fw, fr, fo = 7.0, 0.108, 25.2  # Fable: 3.5x write, 3.0x read, 3.5x out vs Opus
    v_opus = np.array([1.0, cw5_w, 2.0, read_w, out_w]) / bud
    v_fab = np.array([0.5 * fw, fw * (cw5_w / 2.0), fw, fr, fo]) / bud

    rows = []
    idx = 0
    # 24 concurrent overlapping windows across 2 accounts, including a session switching accounts!
    for w_i in range(24):
        acct = "acct-a" if w_i % 2 == 0 else "acct-b"
        t_base = 100_000.0 + w_i * 10_000.0 + (1800.0 if w_i % 2 else 0.0)
        r5 = int(t_base + 18_000.0)
        pct = 0.0
        # Vary category & model mix independently per window so all 8 parameters are well-identified
        w_cw5 = rng.uniform(0.0, 4.0)
        w_cw1 = rng.uniform(0.3, 3.0)
        w_cr = rng.uniform(0.3, 3.0)
        w_out = rng.uniform(0.3, 3.0)
        for j in range(40):
            if w_i % 4 == 0:
                model = ("claude-opus-5-5", "claude-fable-5-2")[j % 2]
            elif w_i % 4 == 1:
                model = ("claude-sonnet-5-5", "claude-opus-5-5")[j % 2]
            elif w_i % 4 == 2:
                model = "claude-fable-5-2" if j % 3 != 0 else "claude-opus-5-5"
            else:
                model = "claude-opus-5-5"
            use_cw5 = j % 3 == 0
            r = request(
                t_base + 60.0 * (j + 1),
                model,
                cw5=rng.uniform(0, 5e4) * w_cw5 if use_cw5 else 0.0,
                cw1=0.0 if use_cw5 else rng.uniform(0, 4e4) * w_cw1,
                cr=rng.uniform(4e4, 2.5e5) * w_cr,
                out=rng.uniform(300, 5e3) * w_out,
            )
            sess = f"s{w_i:02d}-{j % 3}-long"
            r["session"] = sess
            empty_store["requests"][str(idx)] = r
            idx += 1

            x = np.array([r[c] for c in CATS])
            if model == "claude-opus-5-5":
                pct += 100.0 * float(x @ v_opus)
            elif model == "claude-sonnet-5-5":
                pct += 100.0 * son_fac * 0.5 * float(x @ v_opus)
            else:
                pct += 100.0 * float(x @ v_fab)
            if pct < 99.0:
                rows.append((r["t"] + 1.0, sess[:8], model, round(pct), r5, acct))

        # Add a stale dip and a 100% right-censored row; both must be filtered out cleanly.
        rows.append((t_base + 600.0, f"s{w_i:02d}-0-long"[:8], "opus", 1.0, r5, acct))
        rows.append((t_base + 3000.0, f"s{w_i:02d}-0-long"[:8], "opus", 100.0, r5, acct))

    # Session s00-0-long switches from acct-a in window 0 to acct-other much later.
    rows.append((450_000.0, "s00-0-lo", "opus", 12.0, 468_000, "acct-other"))
    empty_store["rate_limits"] = {_line(*row): 1 for row in rows}
    parsed = fit_log.read_log(fit_log.Path("/nonexistent"), empty_store)
    W, unattr = fit_log.log_windows(parsed, empty_store, add_summaries=False)
    assert len(W) == 24 and unattr == 0.0

    f = fit_log.fit(W, free_cw5=True, fable_mode="free")
    assert f["cw5"] == pytest.approx(cw5_w, rel=0.1)
    assert f["read"] == pytest.approx(read_w, rel=0.1)
    assert f["out"] == pytest.approx(out_w, rel=0.1)
    assert f["sonnet"] == pytest.approx(son_fac, rel=0.1)
    assert f["fable_ratios"] == pytest.approx((3.5, 3.0, 3.5), rel=0.1)
    assert f["fable_read"] == pytest.approx(0.108 / 3.5, rel=0.1)
    assert f["budget"] == pytest.approx(bud, rel=0.1)


def test_single_account_log_window_preserves_headless_requests(empty_store):
    v = np.array([1.0, 1.25, 2.0, 0.05, 5.0]) / 10e6
    r1 = request(100.0, cw1=1e5, cr=1e6, out=1e4) | dict(session="logged00-a")
    r2 = request(200.0, cw1=1e5, cr=1e6, out=1e4) | dict(session="headless-b")
    empty_store["requests"] = {"1": r1, "2": r2}
    p1 = 100.0 * float((np.array([r1[c] for c in CATS]) * model_factor(r1)) @ v)
    p2 = p1 + 100.0 * float((np.array([r2[c] for c in CATS]) * model_factor(r2)) @ v)
    rows = [
        (101.0, "logged00", "opus", 0.0, 18_100, "acct-a"),
        (201.0, "logged00", "opus", round(p2, 2), 18_100, "acct-a"),
    ]
    W, _ = fit_log.log_windows(rows, empty_store, add_summaries=False, min_dp=1.0)
    assert len(W) == 1
    assert W[0]["X"][0, 2] == 2e5  # both logged and headless requests counted


def test_fit_marks_missing_models_as_nan_not_one():
    W = [
        dict(dp=50.0, X=np.array([[0, 0, 2e6, 20e6, 2e5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])),
        dict(dp=60.0, X=np.array([[0, 0, 2.4e6, 24e6, 2.4e5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])),
    ]
    f = fit_log.fit(W)
    assert np.isnan(f["sonnet"]) and np.isnan(f["fable"]) and np.isnan(f["fable_read"])


def test_log_windows_skips_unexplained_jump(empty_store):
    empty_store["requests"] = {
        "1": request(100.0, cw1=10_000) | dict(session="sessaaaa-1"),
        "2": request(200.0, cw1=10_000) | dict(session="sessaaaa-1"),
    }
    rows = [
        (101.0, "sessaaaa", "opus", 0.0, 18_100, "acct-a"),
        (201.0, "sessaaaa", "opus", 20.0, 18_100, "acct-a"),
    ]
    stats: dict[str, int] = {}
    W, _ = fit_log.log_windows(rows, empty_store, add_summaries=False, stats=stats)
    assert W == [] and stats["jumps"] == 1


def test_synthetic_generates_clean_two_account_store(empty_store):
    rng = np.random.default_rng(9)
    for k in range(4):
        t0 = 50_000.0 * (k + 1)
        for j in range(10):
            m = "claude-fable-5-2" if (k % 2 and j % 2) else "claude-opus-5-5"
            empty_store["requests"][f"{k}-{j}"] = request(
                t0 + 60.0 * j,
                m,
                cw5=rng.uniform(2e4, 8e4) if j % 2 == 0 else 0.0,
                cw1=rng.uniform(2e4, 8e4) if j % 2 else 0.0,
                cr=rng.uniform(1e5, 4e5),
                out=rng.uniform(500, 3e3),
            ) | dict(session=f"orig-{k}")
        empty_store["hits"][f"h{k}"] = dict(
            t=t0 + 1000.0, kind="5h", reset=t0 + 18_000.0, text="", session=f"orig-{k}"
        )
    rows, S_syn = fit_log.synthetic(empty_store)
    W, _ = fit_log.log_windows(rows, S_syn, add_summaries=False, min_dp=10.0)
    assert len(W) == 4 and len({w["acct"] for w in W}) == 2
    assert fit_log.fit(W, free_cw5=True, fable_mode="flat")["rms"] < 1.0


def test_weekly_ratio_recovers_7d_rate_and_skips_censored_and_small_windows():
    true = 7.5  # 5h points per 7d point
    lines = ["time\tsession\tmodel\tfive_hour_pct\tfive_hour_resets_at\tseven_day_pct\tseven_day_resets_at"]
    p7 = 0.0
    for w in range(12):
        r5 = 100_000 + w * 18_000
        acct = str(1_000_000 + (w // 6) * 604_800)
        for k, p5 in enumerate(range(0, 90, 3)):
            lines.append("\t".join(map(str, [r5 - 17_000 + 60 * k, "s1", "opus", p5, r5, round(p7), acct])))
            p7 += 3 / true
        p7 += 1 / true  # gap between windows
        lines.append("\t".join(map(str, [r5 - 1000, "s1", "opus", 100, r5, 99, acct])))  # censored
        lines.append("\t".join(map(str, [r5 - 900, "s1", "opus", 2, r5, 0, acct])))  # stale dip
    # A window that rose only 10 points and one with no 7d value are skipped.
    lines += ["\t".join(map(str, [900_000 + i, "s2", "opus", 5 + 5 * i, 905_000, 3, "x"])) for i in range(3)]
    lines.append("\t".join(map(str, [950_000, "s3", "opus", 50, 960_000, "", "y"])))

    ws = fit_log.weekly_windows(lines)
    assert len(ws) == 12 and {w[0] for w in ws} == {"1000000", "1604800"}
    assert all(w[1] == 87 for w in ws)
    wr = fit_log.weekly_ratio(ws)
    assert wr["ratio"] == pytest.approx(true, rel=0.05)
    assert wr["q"][0] <= wr["ratio"] <= wr["q"][2]
    assert set(wr["by_acct"]) == {"1000000", "1604800"}
    assert fit_log.weekly_windows(lines, max_t=0.0) == []
