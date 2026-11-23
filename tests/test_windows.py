import pytest
from conftest import request

from cc_compact_tuning import windows

H = 3600


def store_with(requests, logins=(), hit=4.5 * H, reset=5 * H):
    return dict(
        requests={str(i): r for i, r in enumerate(requests)},
        logins={str(i): dict(t=t, session="s") for i, t in enumerate(logins)},
        hits={"h": dict(t=hit, kind="5h", reset=reset, text="", session="s")},
        compactions={},
    )


def test_window_counts_requests_from_start_to_hit():
    st = store_with(
        [request(-60, out=1), request(60, out=10), request(2 * H, out=20), request(4.6 * H, out=40)]
    )
    (w,) = windows.build(st)
    assert w["n"] == 2 and w["X"][0][4] == 30 and not w["flags"]


def test_window_is_cut_at_the_last_login():
    st = store_with([request(60, out=10), request(2 * H + 60, out=20)], logins=[2 * H])
    (w,) = windows.build(st)
    assert w["eff"] == 2 * H and w["X"][0][4] == 20
    assert w["flags"] == ["started 120m before the first request"]  # the new account was used before


def test_late_first_request_is_flagged():
    (w,) = windows.build(store_with([request(H, out=10)]))
    assert w["flags"] == ["started 60m before the first request"]


def test_sonnet_and_fable_go_in_their_own_rows():
    st = store_with([request(60, "claude-sonnet-5-5", out=10), request(120, "claude-fable-5-2", out=10)])
    (w,) = windows.build(st)
    assert list(w["X"][:, 4]) == [0, 5, 20]


def several(n, fable_out):
    """n windows 5 hours apart, each with 100 output tokens of Opus and `fable_out` of Fable."""
    st = dict(requests={}, logins={}, hits={}, compactions={})
    for k in range(n):
        o = k * 5 * H
        st["requests"] |= {
            f"o{k}": request(o + 60, out=100),
            f"f{k}": request(o + 120, "claude-fable-5-2", out=fable_out),
        }
        st["hits"][f"h{k}"] = dict(t=o + 4.5 * H, kind="5h", reset=o + 5 * H, text="", session="s")
    return st


def test_fable_windows_are_flagged_until_there_are_enough():
    ws = windows.build(several(2, fable_out=50))
    assert all(w["fable"] == pytest.approx(0.5) for w in ws)
    assert all(w["flags"] == ["50% Fable: factor needs 3 such windows, 2 so far"] for w in ws)
    assert not any(w["flags"] for w in windows.build(several(3, fable_out=50)))
    assert not any(w["flags"] for w in windows.build(several(1, fable_out=5)))  # under 10%


def test_overlapping_windows_of_two_accounts_are_flagged():
    st = store_with([request(60, out=10), request(4 * H, out=20)])
    st["hits"]["h2"] = dict(t=4.6 * H, kind="5h", reset=8 * H, text="", session="s")  # starts at 3h
    a, b = windows.build(st)
    assert a["flags"] == [f"overlaps the window to {b['label']}: two accounts in use"]
    assert b["flags"][-1] == f"overlaps the window to {a['label']}: two accounts in use"


def test_consecutive_windows_do_not_overlap():
    assert not any(w["flags"] for w in windows.build(several(3, fable_out=0)))


def _log_line(t: float, sess: str, pct: float, reset5: float, acct: str, model: str = "opus") -> str:
    return f"{t}\t{sess[:8]}\t{model}\t{pct}\t{int(reset5)}\t10\t{acct}"


def test_overlapping_windows_rescued_by_rate_log_attribution():
    st = store_with(
        [
            request(60, cw1=200_000, out=10) | dict(session="sessaaaa-1"),
            request(3 * H + 60, cw1=200_000, out=20) | dict(session="sessbbbb-1"),
            request(3.5 * H, cw1=1_200_000, out=30) | dict(session="sessaaaa-1"),
        ]
    )
    st["hits"]["h2"] = dict(t=4.6 * H, kind="5h", reset=8 * H, text="", session="sessbbbb-1")
    st["rate_limits"] = {
        _log_line(120, "sessaaaa-1", 10, 5 * H, "acct-a"): 1,
        _log_line(3 * H + 120, "sessbbbb-1", 15, 8 * H, "acct-b"): 1,
        _log_line(3.5 * H + 5, "sessaaaa-1", 95, 5 * H, "acct-a"): 1,
        _log_line(4.5 * H, "sessaaaa-1", 100, 5 * H, "acct-a"): 1,
        _log_line(4.6 * H, "sessbbbb-1", 100, 8 * H, "acct-b"): 1,
    }
    a, b = windows.build(st)
    assert a["attributed"] and b["attributed"]
    assert not a["flags"] and not b["flags"]
    assert a["X"][0][4] == 40
    assert b["X"][0][4] == 20


def test_single_hit_multi_account_strips_other_account_and_keeps_compactions():
    st = store_with(
        [
            request(60, out=100) | dict(session="sessaaaa-1"),
            request(3700, out=999) | dict(session="sessbbbb-1"),
        ]
    )
    st["compactions"] = {
        "c1": dict(t=90, session="sessaaaa-1", agent=None, trigger="auto", pre=168_000),
        "c2": dict(t=3750, session="sessbbbb-1", agent=None, trigger="auto", pre=168_000),
        "c3": dict(t=100, session="sessaaaa-1", agent=None, trigger="manual", pre=100_000),
    }
    st["rate_limits"] = {
        _log_line(90, "sessaaaa-1", 50, 5 * H, "acct-a"): 1,
        _log_line(3750, "sessbbbb-1", 30, 6 * H, "acct-b"): 1,
        _log_line(4.5 * H, "sessaaaa-1", 100, 5 * H, "acct-a"): 1,
    }
    (w,) = windows.build(st)
    assert w["attributed"] and not w["flags"]
    assert w["X"][0][4] == 100
    assert w["compactions"] == 1
    assert len(w["compaction_events"]) == 2
    assert {c["trigger"] for c in w["compaction_events"]} == {"auto", "manual"}


def test_multi_account_window_flags_high_unattributed_share():
    st = store_with(
        [
            request(60, out=10) | dict(session="sessaaaa-1"),
            request(120, out=50) | dict(session="unlogged-1"),
        ]
    )
    st["rate_limits"] = {
        _log_line(90, "sessaaaa-1", 50, 5 * H, "acct-a"): 1,
        _log_line(150, "sessbbbb-1", 20, 6 * H, "acct-b"): 1,
        _log_line(4.5 * H, "sessaaaa-1", 100, 5 * H, "acct-a"): 1,
    }
    (w,) = windows.build(st)
    assert any("unattributed" in fl for fl in w["flags"])


def test_single_account_window_preserves_unlogged_headless_requests():
    st = store_with(
        [
            request(60, out=10) | dict(session="sessaaaa-1"),
            request(120, out=50) | dict(session="headless-1"),
        ]
    )
    st["rate_limits"] = {
        _log_line(90, "sessaaaa-1", 50, 5 * H, "acct-a"): 1,
        _log_line(4.5 * H, "sessaaaa-1", 100, 5 * H, "acct-a"): 1,
    }
    (w,) = windows.build(st)
    assert not w["attributed"] and not w["flags"]
    assert w["X"][0][4] == 60


def test_unexplained_jump_flags_external_usage():
    st = store_with(
        [
            request(60, out=10) | dict(session="sessaaaa-1"),
            request(200, out=100) | dict(session="sessaaaa-1"),
        ]
    )
    st["rate_limits"] = {
        _log_line(70, "sessaaaa-1", 5, 5 * H, "acct-a"): 1,
        _log_line(210, "sessaaaa-1", 20, 5 * H, "acct-a"): 1,  # +15% with ~500 API tokens -> flagged
        _log_line(4.5 * H, "sessaaaa-1", 100, 5 * H, "acct-a"): 1,
    }
    (w,) = windows.build(st)
    assert any("unexplained +15% jump" in fl for fl in w["flags"])


def test_single_account_logged_window_overlapping_attributed_window_is_clean():
    st = store_with(
        [
            request(60, cw1=100_000) | dict(session="sessaaaa-1"),
            request(3 * H + 60, cw1=100_000) | dict(session="sessbbbb-1"),
        ]
    )
    st["hits"]["h2"] = dict(t=4.6 * H, kind="5h", reset=8 * H, text="", session="sessbbbb-1")
    st["rate_limits"] = {
        _log_line(70, "sessaaaa-1", 5, 5 * H, "acct-a"): 1,
        _log_line(4.5 * H, "sessaaaa-1", 100, 5 * H, "acct-a"): 1,
        # Only sessbbbb logs after hit 1, but its 5h window started at 3*H (before hit 1)
        _log_line(4.55 * H, "sessbbbb-1", 50, 8 * H, "acct-b"): 1,
        _log_line(4.6 * H, "sessbbbb-1", 100, 8 * H, "acct-b"): 1,
    }
    a, b = windows.build(st)
    assert a["single_acct_logged"] and b["attributed"]
    assert not a["flags"] and not b["flags"]
