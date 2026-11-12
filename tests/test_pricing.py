import pytest
from conftest import request

from cc_compact_tuning.pricing import model_factor


def test_opus_is_the_unit():
    assert model_factor(request(0)) == [1.0] * 5


def test_other_models_at_their_list_price_ratio():
    assert model_factor(request(0, "claude-sonnet-5-5")) == [0.5] * 5
    assert model_factor(request(0, "claude-fable-5-2")) == [2.0, 2.0, 2.0, 1.0, 2.0]  # same read price


def test_haiku_long_prompts_use_the_long_rate():
    short = model_factor(request(0, "claude-haiku-5-5", cr=90_000))
    long = model_factor(request(0, "claude-haiku-5-5", cr=120_000))
    assert short[0] == pytest.approx(0.025) and long[0] == pytest.approx(0.125)


def test_fast_mode_doubles_and_unknown_models_fall_back():
    assert model_factor(request(0) | dict(speed="fast")) == [2.0] * 5
    assert model_factor(request(0, "claude-sonnet-5-5-20261101")) == [0.5] * 5
    assert model_factor(request(0, "something-else")) == [1.0] * 5
