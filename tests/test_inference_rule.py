"""
S4 — make the min_conf decision rule and `confidence` semantics explicit.

`confidence` is the probability of the *emitted* class. When the top class is
below `min_conf` the signal is forced to HOLD, so `confidence` becomes P(HOLD),
which can be lower than the model's raw top probability. That used to be silent
and confusing; `decide_signal` now also returns `max_prob` (the raw top prob)
and `forced_hold`, so the relationship is observable.
"""
from __future__ import annotations

from model_tech.infer import decide_signal
from model_tech.types import Signal


def test_high_confidence_emits_argmax() -> None:
    sig, conf, max_prob, forced = decide_signal([0.1, 0.2, 0.7], 0.5)
    assert sig == Signal.BUY
    assert conf == 0.7
    assert max_prob == 0.7
    assert forced is False


def test_below_min_conf_forces_hold_with_phold_confidence() -> None:
    # argmax = SELL(0.36) < 0.5 -> forced HOLD; confidence = P(HOLD)=0.33
    sig, conf, max_prob, forced = decide_signal([0.36, 0.33, 0.31], 0.5)
    assert sig == Signal.HOLD
    assert forced is True
    assert max_prob == 0.36                # raw top prob is preserved
    assert abs(conf - 0.33) < 1e-9         # confidence is P(emitted = HOLD)
    assert conf < max_prob                 # the previously-hidden gap is explicit


def test_model_choosing_hold_is_not_flagged_forced() -> None:
    sig, conf, max_prob, forced = decide_signal([0.2, 0.5, 0.3], 0.4)
    assert sig == Signal.HOLD
    assert forced is False                 # HOLD was the model's own argmax
    assert conf == 0.5 == max_prob


def test_max_prob_equal_to_min_conf_keeps_argmax() -> None:
    # Boundary: HOLD is forced only when max_prob < min_conf (strict).
    sig, conf, max_prob, forced = decide_signal([0.5, 0.3, 0.2], 0.5)
    assert sig == Signal.SELL
    assert forced is False
    assert max_prob == 0.5
