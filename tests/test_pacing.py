"""First tests in this repo (see tests/.gitkeep) -- targets pacing.py's pure
helper functions, extracted out of update_delivery_lambda/update_ctr_lambda
as part of the learned-pacing plan (see plans/lucky-coalescing-crystal.md).
These are cheap to test and easy to get subtly wrong in a way that wouldn't
show up until a much later, harder-to-debug stage (a hindsight-search
pipeline silently training on a shifted/miscalibrated signal).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adtech_rtb.pacing import (  # noqa: E402
    MIN_DELIVERED_FOR_CTR_JUDGMENT,
    PACE_CONVEXITY,
    ctr_error,
    pacing_error,
    update_ctr_lambda,
    update_delivery_lambda,
)


def test_pacing_error_on_pace_is_zero():
    assert pacing_error(delivered=50, target_impressions=100, elapsed_fraction=0.5, pace_convexity=1.0) == 0.0


def test_pacing_error_behind_pace_is_positive():
    assert pacing_error(delivered=10, target_impressions=100, elapsed_fraction=0.5, pace_convexity=1.0) > 0.0


def test_pacing_error_ahead_of_pace_is_negative():
    assert pacing_error(delivered=90, target_impressions=100, elapsed_fraction=0.5, pace_convexity=1.0) < 0.0


def test_pacing_error_past_deadline_freezes_expected_at_target():
    # elapsed_fraction > 1.0: expected_cumulative pins at target regardless of convexity.
    error_1_5x = pacing_error(delivered=80, target_impressions=100, elapsed_fraction=1.5, pace_convexity=1.0)
    error_3x = pacing_error(delivered=80, target_impressions=100, elapsed_fraction=3.0, pace_convexity=3.0)
    assert error_1_5x == error_3x == 0.2


def test_ctr_error_zero_below_min_delivered_threshold():
    assert ctr_error(running_ctr=0.0, ctr_floor=0.001, delivered=MIN_DELIVERED_FOR_CTR_JUDGMENT - 1) == 0.0


def test_ctr_error_nonzero_once_threshold_met():
    assert ctr_error(running_ctr=0.0, ctr_floor=0.001, delivered=MIN_DELIVERED_FOR_CTR_JUDGMENT) == 1.0


def test_update_delivery_lambda_clips_to_lambda_max():
    result = update_delivery_lambda(
        lambda_delivery=0.0, delivered=0, target_impressions=100, elapsed_fraction=0.01, eta=1000.0, lambda_max=1.5
    )
    assert result == 1.5


def test_update_delivery_lambda_clips_to_zero():
    result = update_delivery_lambda(
        lambda_delivery=0.0, delivered=1000, target_impressions=100, elapsed_fraction=0.01, eta=1000.0, lambda_max=1.5
    )
    assert result == 0.0


def test_update_ctr_lambda_is_noop_below_min_delivered():
    result = update_ctr_lambda(
        lambda_ctr=0.3, running_ctr=0.0, ctr_floor=0.001, delivered=MIN_DELIVERED_FOR_CTR_JUDGMENT - 1, eta=1000.0
    )
    assert result == 0.3


def test_update_delivery_lambda_default_pace_convexity_matches_module_constant():
    # Regression guard for the pace_convexity param threaded through in this
    # refactor: the default must still resolve to PACE_CONVEXITY so
    # simulate_synthetic_flight's behavior is unchanged by this change.
    with_default = update_delivery_lambda(lambda_delivery=0.0, delivered=10, target_impressions=100, elapsed_fraction=0.5)
    with_explicit = update_delivery_lambda(
        lambda_delivery=0.0, delivered=10, target_impressions=100, elapsed_fraction=0.5, pace_convexity=PACE_CONVEXITY
    )
    assert with_default == with_explicit
