"""API list prices, and conversion of any model's tokens into Opus-equivalent tokens."""

import datetime as dt
from typing import Any

CATS = ["inp", "cw5", "cw1", "cr", "out"]

# Per MTok (claude-api skill, models.md / model-migration.md):
# input, 5m write, 1h write, cache read, output.
PRICES = {
    "claude-opus-5-5": (4.0, 5.0, 8.0, 0.20, 20.0),
    "claude-fable-5-2": (8.0, 10.0, 16.0, 0.20, 40.0),
    "claude-sonnet-5-5": (2.0, 2.5, 4.0, 0.10, 10.0),
    "claude-haiku-5-5": (0.10, 0.125, 0.20, 0.01, 0.50),  # prompts <= 100K tokens
}
HAIKU_LONG = (0.50, 0.625, 1.00, 0.05, 2.50)  # prompts > 100K tokens
# Models priced higher above a prompt size: model -> (prompt tokens, prices above them).
LONG_PROMPT = {"claude-haiku-5-5": (100_000, HAIKU_LONG)}
OPUS = PRICES["claude-opus-5-5"]
API_W = [p / OPUS[0] for p in OPUS]  # [1, 1.25, 2, 0.05, 5]
MODELS = {m.split("-")[1]: m for m in PRICES}  # opus, fable, sonnet, haiku -> model id


def price_ratio(model: str, prompt: float = 0) -> list[float]:
    """Per-category price of `model` relative to Opus, for a prompt of `prompt` tokens. Unknown
    models are priced as the listed model their id starts with, else as Opus."""
    m = model if model in PRICES else next((k for k in PRICES if model.startswith(k)), None)
    p = PRICES[m] if m else OPUS
    if m in LONG_PROMPT and prompt > LONG_PROMPT[m][0]:
        p = LONG_PROMPT[m][1]
    return [a / b for a, b in zip(p, OPUS, strict=True)]


def model_factor(r: dict[str, Any]) -> list[float]:
    """Per-category price of this request's model relative to Opus. Fast mode bills 2x."""
    f = price_ratio(r["model"] or "", r["inp"] + r["cw5"] + r["cw1"] + r["cr"])
    if r.get("speed") == "fast":
        f = [2 * x for x in f]
    return f


def is_sonnet(r: dict[str, Any]) -> bool:
    return (r["model"] or "").startswith("claude-sonnet")


def is_fable(r: dict[str, Any]) -> bool:
    return (r["model"] or "").startswith("claude-fable")


def row(r: dict[str, Any]) -> int:
    """Token row a request counts in: 0 Opus and Haiku, 1 Sonnet, 2 Fable. Sonnet and Fable get
    their own rows because the fits measure how they count against the limit."""
    return 1 if is_sonnet(r) else 2 if is_fable(r) else 0


def local(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().strftime("%a %m-%d %H:%M")
