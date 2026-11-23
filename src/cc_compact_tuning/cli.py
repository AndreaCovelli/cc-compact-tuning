"""cc-compact-tuning: collect transcripts, fit the usage weights, replay compaction triggers."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import collect, fit, fit_log, replay, store, windows
from .pricing import MODELS
from .settings import Settings


def _collect(s: Settings) -> None:
    st = store.load(s.store)
    before = {k: len(v) for k, v in st.items()}
    n = collect.collect(s.projects_dir, s.rate_log, st)
    store.save(st, s.store)
    print(f"{n} transcript files scanned into {s.store}")
    for k, v in st.items():
        print(f"  {k}: {len(v)} ({len(v) - before.get(k, 0):+d})")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cc-compact-tuning", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect", help="merge the transcripts and the rate-limit log into the store")
    sub.add_parser("windows", help="list the 5-hour windows that ended in a limit hit")
    f = sub.add_parser("fit", help="fit the usage weights to the limit-hit windows")
    f.add_argument("--bootstrap", type=int, default=1000, metavar="N", help="resamples (default 1000)")
    r = sub.add_parser("replay", help="replay a model's sessions at other auto-compact triggers")
    r.add_argument(
        "--model",
        choices=MODELS,
        default="opus",
        help="model to price the replay as, and whose sessions to replay if it has enough compactions"
        " of its own, else the Opus sessions (default: opus)",
    )
    r.add_argument(
        "--weights",
        type=float,
        nargs=5,
        default=None,
        metavar=("INPUT", "CW5", "CW1", "READ", "OUTPUT"),
        help="usage weights per Opus token category (default: fitted weights for the chosen model)",
    )
    lg = sub.add_parser("fit-log", help="fit the usage weights to the status line's rate-limit log")
    lg.add_argument("log", nargs="?", type=Path, help="log file (default: the status line's)")
    lg.add_argument("--synthetic", action="store_true", help="fit a log made from known weights instead")
    a = p.parse_args(argv)

    s = Settings()
    if a.cmd == "collect":
        _collect(s)
        return
    st = store.load(s.store)
    if a.cmd == "windows":
        windows.report(st)
    elif a.cmd == "fit":
        fit.report(st, s.data_dir, a.bootstrap)
    elif a.cmd == "replay":
        replay.report(st, s.data_dir, a.weights, MODELS[a.model])
    elif a.cmd == "fit-log":
        fit_log.report(st, s.data_dir, a.log or s.rate_log, a.synthetic)
