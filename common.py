"""Shared loading, prices and window helpers."""
import datetime as dt
import gzip
import json
import os
from zoneinfo import ZoneInfo

STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "store.json.gz")
ROME = ZoneInfo("Europe/Rome")
CATS = ["inp", "cw5", "cw1", "cr", "out"]

# API list prices per MTok (claude-api skill, models.md / model-migration.md):
# input, 5m write, 1h write, cache read, output.
PRICES = {
    "claude-opus-5-5":   (4.0, 5.0, 8.0, 0.20, 20.0),
    "claude-fable-5-2":  (8.0, 10.0, 16.0, 0.20, 40.0),
    "claude-sonnet-5-5": (2.0, 2.5, 4.0, 0.10, 10.0),
    "claude-haiku-5-5":  (0.10, 0.125, 0.20, 0.01, 0.50),    # prompts <= 100K tokens
}
HAIKU_LONG = (0.50, 0.625, 1.00, 0.05, 2.50)                  # prompts > 100K tokens
OPUS = PRICES["claude-opus-5-5"]
API_W = [p / OPUS[0] for p in OPUS]                            # [1, 1.25, 2, 0.05, 5]


def load():
    with gzip.open(STORE, "rt") as f:
        return json.load(f)


def model_factor(r):
    """Per-category price of this request's model relative to Opus, so that tokens can be
    expressed as Opus-equivalent tokens. Fast mode is billed at 2x."""
    m = r["model"] or ""
    p = PRICES.get(m)
    if m == "claude-haiku-5-5" and r["inp"] + r["cw5"] + r["cw1"] + r["cr"] > 100_000:
        p = HAIKU_LONG
    if p is None:
        base = next((k for k in PRICES if m.startswith(k)), None)
        p = PRICES[base] if base else OPUS
    f = [a / b for a, b in zip(p, OPUS)]
    if r.get("speed") == "fast":
        f = [2 * x for x in f]
    return f


def opus_equiv(r):
    f = model_factor(r)
    return [r[c] * f[i] for i, c in enumerate(CATS)]


def local(t):
    return dt.datetime.fromtimestamp(t, ROME).strftime("%a %m-%d %H:%M")
