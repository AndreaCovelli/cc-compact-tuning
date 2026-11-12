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
