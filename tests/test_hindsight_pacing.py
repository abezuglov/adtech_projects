import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from adtech_rtb.hindsight_pacing import (  # noqa: E402
    STATE_FEATURE_NAMES,
    _sub_result_from_suffix,
    extract_training_pairs,
    hindsight_loss,
)

BASE_RESULT = {
    "delivered_impressions": 100,
    "target_impressions": 100,
    "spend": 20.0,
    "overrun_ratio": 1.0,
    "delivery_cv": 0.0,
    "ctr_floor": 0.01,
    "achieved_ctr": 0.01,
}


def test_hindsight_loss_zero_when_everything_on_target():
    # Delivered exactly to target, no overrun, CPM matches naive exactly, CTR exactly at floor.
    assert hindsight_loss(BASE_RESULT, naive_cpm=200.0) == 0.0


def test_hindsight_loss_increases_with_shortfall():
    short = {**BASE_RESULT, "delivered_impressions": 50}
    assert hindsight_loss(short, naive_cpm=200.0) > hindsight_loss(BASE_RESULT, naive_cpm=200.0)


def test_hindsight_loss_increases_with_overrun():
    over = {**BASE_RESULT, "overrun_ratio": 1.5}
    assert hindsight_loss(over, naive_cpm=200.0) > hindsight_loss(BASE_RESULT, naive_cpm=200.0)


def test_hindsight_loss_increases_with_delivery_cv():
    lumpy = {**BASE_RESULT, "delivery_cv": 0.8}
    assert hindsight_loss(lumpy, naive_cpm=200.0) > hindsight_loss(BASE_RESULT, naive_cpm=200.0)


def test_hindsight_loss_penalizes_ctr_below_floor():
    below_floor = {**BASE_RESULT, "achieved_ctr": 0.005}
    assert hindsight_loss(below_floor, naive_cpm=200.0) > hindsight_loss(BASE_RESULT, naive_cpm=200.0)


def test_hindsight_loss_rewards_beating_naive_cpm():
    # spend=20 on 100 delivered -> CPM=200. A higher naive_cpm means this run
    # beat the naive baseline, which should be a NEGATIVE (loss-reducing) term.
    beat_naive = hindsight_loss(BASE_RESULT, naive_cpm=300.0)
    matched_naive = hindsight_loss(BASE_RESULT, naive_cpm=200.0)
    assert beat_naive < matched_naive


FAKE_SCENARIO = {"id": "fake", "flight_length_days": 1, "target_impressions": 100, "ctr_floor": 0.01}

FAKE_TRAJECTORY = [
    {
        "days_used": 0.5,
        "cumulative_delivered": 50,
        "cumulative_spend": 10.0,
        "cumulative_clicks": 1,
        "running_ctr": 0.02,
        "batch_won": 50,
        "lambda_delivery": 0.1,
        "lambda_ctr": 0.0,
    },
    {
        "days_used": 1.0,
        "cumulative_delivered": 100,
        "cumulative_spend": 20.0,
        "cumulative_clicks": 2,
        "running_ctr": 0.02,
        "batch_won": 50,
        "lambda_delivery": 0.2,
        "lambda_ctr": 0.0,
    },
]

FAKE_FINAL_RESULT = {
    "delivered_impressions": 100,
    "target_impressions": 100,
    "spend": 20.0,
    "clicks": 2,
    "days_used": 1.0,
    "overrun_ratio": 1.0,
    "delivery_cv": 0.0,
    "ctr_floor": 0.01,
    "achieved_ctr": 0.02,
    "trajectory": FAKE_TRAJECTORY,
}

FAKE_RUN = {"params": {"pace_convexity": 1.0}, "loss": 0.0, "result": FAKE_FINAL_RESULT}


def test_sub_result_from_suffix_at_start_matches_full_flight():
    # start_idx=0 (whole trajectory is the "tail") must reproduce the full-flight totals.
    sub = _sub_result_from_suffix(FAKE_TRAJECTORY, 0, FAKE_SCENARIO, FAKE_FINAL_RESULT)
    assert sub["delivered_impressions"] == 100
    assert sub["target_impressions"] == 100
    assert sub["spend"] == 20.0
    assert sub["achieved_ctr"] == 0.02


def test_sub_result_from_suffix_at_last_step_is_the_final_batch_only():
    sub = _sub_result_from_suffix(FAKE_TRAJECTORY, 1, FAKE_SCENARIO, FAKE_FINAL_RESULT)
    assert sub["delivered_impressions"] == 50  # 100 - 50 (step 0's cumulative)
    assert sub["target_impressions"] == 50  # 100 - 50 remaining
    assert sub["spend"] == 10.0  # 20 - 10


def test_extract_training_pairs_state_target_alignment():
    # loss_percentile=100 keeps every row -- this test is purely about
    # (state, target) alignment, not the filtering behavior.
    X, y = extract_training_pairs([FAKE_RUN], FAKE_SCENARIO, naive_cpm=200.0, loss_percentile=100.0)

    assert X.shape == (2, len(STATE_FEATURE_NAMES))
    assert y.shape == (2, 2)

    # Step 0: elapsed_fraction=0.5, delivered=50/target=100 exactly on a
    # linear (pace_convexity=1.0) pace -> pacing_error=0 -> sq_behind=0;
    # running_ctr=0.02 vs floor=0.01 -> ctr_error=(0.01-0.02)/0.01=-1.0;
    # delivered_fraction=0.5. overrun_fraction=max(0, 0.5-1)=0 -> both
    # overrun interaction terms are 0 regardless of pacing_error/ctr_error.
    np.testing.assert_allclose(X[0], [0.5, 0.0, 0.0, -1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(y[0], [0.1, 0.0])

    # Step 1: elapsed_fraction=1.0, delivered=100/target=100 exactly on pace
    # -> pacing_error=0 -> sq_behind=0; same running_ctr/floor -> ctr_error=-1.0;
    # delivered_fraction=1.0. overrun_fraction=max(0, 1.0-1)=0 (not yet past
    # nominal length) -> both overrun interaction terms still 0.
    np.testing.assert_allclose(X[1], [1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(y[1], [0.2, 0.0])


def test_extract_training_pairs_filters_by_loss_percentile():
    # A second, worse-scoring run (delivers nothing) should be dropped
    # entirely at a low percentile, leaving only the good run's rows.
    bad_trajectory = [
        {
            "days_used": 1.0,
            "cumulative_delivered": 0,
            "cumulative_spend": 0.0,
            "cumulative_clicks": 0,
            "running_ctr": 0.0,
            "batch_won": 0,
            "lambda_delivery": 0.0,
            "lambda_ctr": 0.0,
        }
    ]
    bad_final_result = {
        "delivered_impressions": 0,
        "target_impressions": 100,
        "spend": 0.0,
        "clicks": 0,
        "days_used": 1.0,
        "overrun_ratio": 1.0,
        "delivery_cv": 0.0,
        "ctr_floor": 0.01,
        "achieved_ctr": 0.0,
        "trajectory": bad_trajectory,
    }
    bad_run = {"params": {"pace_convexity": 1.0}, "loss": 999.0, "result": bad_final_result}

    X, y = extract_training_pairs([FAKE_RUN, bad_run], FAKE_SCENARIO, naive_cpm=200.0, loss_percentile=50.0)
    # Only the good run's 2 rows should survive a 50th-percentile filter
    # against one badly-underdelivering row.
    assert X.shape[0] == 2
    assert np.all(y[:, 0] >= 0.1)
