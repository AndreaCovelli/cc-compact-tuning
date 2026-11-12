import datetime as dt
import json
from pathlib import Path

import pytest

from cc_compact_tuning import store

T0 = dt.datetime(2026, 11, 10, 9, 0, tzinfo=dt.UTC).timestamp()


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).isoformat().replace("+00:00", "Z")


def request(t: float, model: str = "claude-opus-5-5", **u) -> dict:
    """A request as collect stores it."""
    return dict(t=t, model=model, session="s", agent=None, side=False, speed=None) | {
        k: u.get(k, 0) for k in ("inp", "cw5", "cw1", "cr", "out")
    }


def assistant(mid: str, t: float, out: int, uuid: str = "", **extra) -> dict:
    """An assistant transcript line."""
    usage = dict(
        input_tokens=3,
        cache_creation_input_tokens=1000,
        cache_creation=dict(ephemeral_1h_input_tokens=800),
        cache_read_input_tokens=50_000,
        output_tokens=out,
    )
    msg = dict(id=mid, model="claude-opus-5-5", usage=usage)
    return dict(type="assistant", uuid=uuid or f"u-{mid}-{out}", timestamp=iso(t), message=msg) | extra


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture
def empty_store() -> store.Store:
    return store.empty()
