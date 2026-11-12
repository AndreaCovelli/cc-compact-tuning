import datetime as dt
from zoneinfo import ZoneInfo

from conftest import T0, assistant, iso, write_jsonl

from cc_compact_tuning import collect, store

ROME = ZoneInfo("Europe/Rome")


def rome(*a: int) -> float:
    return dt.datetime(*a, tzinfo=ROME).timestamp()


def test_parse_reset_later_today():
    assert collect.parse_reset("9:20pm", "Europe/Rome", rome(2026, 11, 10, 18, 0)) == rome(
        2026, 11, 10, 21, 20
    )


def test_parse_reset_rolls_over_midnight():
    assert collect.parse_reset("1:40am", "Europe/Rome", rome(2026, 11, 10, 23, 0)) == rome(
        2026, 11, 11, 1, 40
    )


def test_parse_reset_with_date_and_new_year():
    assert collect.parse_reset("Nov 14, 2pm", "Europe/Rome", rome(2026, 11, 10, 9)) == rome(2026, 11, 14, 14)
    assert collect.parse_reset("Jan 2, 7am", "Europe/Rome", rome(2026, 12, 30, 9)) == rome(2027, 1, 2, 7)


def test_parse_reset_rejects_other_text():
    assert collect.parse_reset("tomorrow", "Europe/Rome", T0) is None


def transcript(root):
    hit = dict(
        type="assistant",
        uuid="hit-1",
        timestamp=iso(T0 + 60),
        error="rate_limit",
        message=dict(
            model="<synthetic>",
            usage=dict(input_tokens=0, output_tokens=0),
            content=[dict(type="text", text="You've hit your 5-hour limit · resets 4pm (Europe/Rome)")],
        ),
    )
    login = dict(
        type="user",
        uuid="login-1",
        timestamp=iso(T0 - 600),
        message=dict(role="user", content="<local-command-stdout>Login successful</local-command-stdout>"),
    )
    boundary = dict(
        type="system",
        subtype="compact_boundary",
        uuid="cb-1",
        timestamp=iso(T0 + 30),
        compactMetadata=dict(trigger="auto", preTokens=168_000, postTokens=12_000),
    )
    return write_jsonl(
        root / "proj" / "sess-a.jsonl",
        [login, assistant("m1", T0, out=10), assistant("m1", T0 + 5, out=400), boundary, hit],
    )


def test_scan_dedupes_streamed_messages_and_finds_events(tmp_path, empty_store):
    root = tmp_path / "projects"
    collect.scan(transcript(root), root, empty_store)

    (r,) = empty_store["requests"].values()
    assert (r["t"], r["out"], r["session"]) == (T0, 400, "sess-a")  # first time, final output
    assert (r["cw1"], r["cw5"], r["cr"]) == (800, 200, 50_000)
    (h,) = empty_store["hits"].values()
    assert h["kind"] == "5h" and h["reset"] > h["t"]
    assert empty_store["compactions"]["cb-1"]["pre"] == 168_000
    assert empty_store["logins"]["login-1"]["t"] == T0 - 600


def test_subagent_transcripts_are_side_requests(tmp_path, empty_store):
    root = tmp_path / "projects"
    p = write_jsonl(root / "proj" / "sess-a" / "subagents" / "agent-1.jsonl", [assistant("m9", T0, out=5)])
    collect.scan(p, root, empty_store)
    r = empty_store["requests"]["m9"]
    assert r["side"] and r["agent"] == "agent-1" and r["session"] == "sess-a"


def test_collect_keeps_records_of_deleted_transcripts(tmp_path):
    root, log, path = tmp_path / "projects", tmp_path / "rate.tsv", tmp_path / "data" / "store.json.gz"
    t = transcript(root)
    log.write_text("header\n" + "\t".join(["1", "s", "opus", "40", "2", "10", "3"]) + "\nbroken line\n")
    st = store.empty()
    assert collect.collect(root, log, st) == 1
    store.save(st, path)

    t.unlink()
    log.unlink()
    st = store.load(path)
    assert collect.collect(root, log, st) == 0
    assert "m1" in st["requests"] and "hit-1" in st["hits"]
    assert list(st["rate_limits"]) == ["1\ts\topus\t40\t2\t10\t3"]
