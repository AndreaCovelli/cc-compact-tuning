import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cc_compact_tuning import store
from cc_compact_tuning.collect import merge_rate_log
from cc_compact_tuning.windows import parse_rate_log

SCRIPT = Path(__file__).parents[1] / "statusline" / "rate-limit-log.sh"
pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("jq")), reason="needs bash and jq")


def status(session: str = "0e19b488-aaaa", pct5: float = 25, pct7: float = 40, **extra) -> dict:
    """Status-line JSON as Claude Code pipes it, with only the fields the logger reads."""
    limits = dict(
        five_hour=dict(used_percentage=pct5, resets_at=1794486000),
        seven_day=dict(used_percentage=pct7, resets_at=1795068000),
    )
    return dict(session_id=session, model=dict(id="claude-opus-5-5"), rate_limits=limits) | extra


def run(tmp_path: Path, data: dict, *args: str, **env: str) -> str:
    env = os.environ | {"HOME": str(tmp_path), "CC_TUNING_RATE_LOG": str(tmp_path / "log.tsv")} | env
    out = subprocess.run(
        ["bash", str(SCRIPT), *args], input=json.dumps(data), env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_logs_only_changes_in_the_format_collect_reads(tmp_path):
    assert run(tmp_path, status()) == "5h 25% · 7d 40%\n"
    run(tmp_path, status())  # unchanged: no new line
    run(tmp_path, status(pct5=26))
    run(tmp_path, status(session="a4533925-bbbb"))  # another session, same numbers: logged
    lines = (tmp_path / "log.tsv").read_text().splitlines()
    assert lines[0].startswith("time\tsession\tmodel\t")
    assert len(lines) == 4
    S = store.empty()
    merge_rate_log(tmp_path / "log.tsv", S)
    assert len(S["rate_limits"]) == 3
    rows = parse_rate_log(S["rate_limits"])
    assert sorted((r[1], r[3], r[4], r[5]) for r in rows) == [
        ("0e19b488", 25.0, 1794486000, "1795068000"),
        ("0e19b488", 26.0, 1794486000, "1795068000"),
        ("a4533925", 25.0, 1794486000, "1795068000"),
    ]


def test_without_rate_limits_logs_nothing(tmp_path):
    data = status()
    del data["rate_limits"]
    assert run(tmp_path, data) == "\n"
    assert not (tmp_path / "log.tsv").exists()


def test_wraps_another_status_line_with_the_same_input(tmp_path):
    data = status()
    assert json.loads(run(tmp_path, data, "cat")) == data
    assert (tmp_path / "log.tsv").exists()


def test_prunes_to_the_header_and_last_lines(tmp_path):
    for pct in range(1, 8):
        run(tmp_path, status(pct5=pct), CC_TUNING_RATE_LOG_MAX="5", CC_TUNING_RATE_LOG_KEEP="2")
    lines = (tmp_path / "log.tsv").read_text().splitlines()
    assert lines[0].startswith("time\t")
    assert len(lines) <= 6
    assert lines[-1].split("\t")[3] == "7"
