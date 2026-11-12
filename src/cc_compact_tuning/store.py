"""The store: everything the analysis needs, kept after Claude Code deletes the transcripts."""

import gzip
import json
import os
from pathlib import Path
from typing import Any

Store = dict[str, dict[str, Any]]
SECTIONS = ("requests", "compactions", "hits", "cost_states", "logins", "rate_limits")


def empty() -> Store:
    return {k: {} for k in SECTIONS}


def load(path: Path) -> Store:
    if not path.exists():
        return empty()
    with gzip.open(path, "rt") as f:
        store = json.load(f)
    for k in SECTIONS:
        store.setdefault(k, {})
    return store


def save(store: Store, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(store, f)
    os.replace(tmp, path)
