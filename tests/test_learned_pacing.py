import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from adtech_rtb.learned_pacing import LearnedPacingController, pacing_state_features  # noqa: E402
from adtech_rtb.pacing import AnalyticPacingController, update_ctr_lambda, update_delivery_lambda  # noqa: E402


def test_pacing_state_features_caps_elapsed_fraction():
    # elapsed_fraction=5.0 (well past overrun) must be capped at 3.0, matching
    # extract_training_pairs' identical cap in hindsight_pacing.py.
    features = pacing_state_features(
        delivered=100, target_impressions=100, elapsed_fraction=5.0, running_ctr=0.01, ctr_floor=0.01
    )
    assert features[0] == 3.0


def test_pacing_state_features_shape_and_order():
    features = pacing_state_features(
        delivered=50, target_impressions=100, elapsed_fraction=0.5, running_ctr=0.02, ctr_floor=0.01
    )
    assert features.shape == (4,)
    # delivered_fraction is the last feature.
    assert features[3] == 0.5


def test_learned_pacing_controller_clips_to_lambda_max():
    # Extreme positive weights should still be clipped at lambda_delivery_max/lambda_ctr_max.
    controller = LearnedPacingController(
        delivery_weights=[100.0, 0.0, 0.0, 0.0, 0.0],
        ctr_weights=[100.0, 0.0, 0.0, 0.0, 0.0],
        lambda_delivery_max=1.5,
        lambda_ctr_max=1.5,
    )
    ld, lc = controller.update(
        lambda_delivery=0.0,
        lambda_ctr=0.0,
        delivered=10,
        target_impressions=100,
        elapsed_fraction=0.1,
        running_ctr=0.01,
        ctr_floor=0.01,
    )
    assert ld == 1.5
    assert lc == 1.5


def test_learned_pacing_controller_clips_to_zero():
    controller = LearnedPacingController(
        delivery_weights=[-100.0, 0.0, 0.0, 0.0, 0.0],
        ctr_weights=[-100.0, 0.0, 0.0, 0.0, 0.0],
        lambda_delivery_max=1.5,
        lambda_ctr_max=1.5,
    )
    ld, lc = controller.update(
        lambda_delivery=0.0,
        lambda_ctr=0.0,
        delivered=10,
        target_impressions=100,
        elapsed_fraction=0.1,
        running_ctr=0.01,
        ctr_floor=0.01,
    )
    assert ld == 0.0
    assert lc == 0.0


def test_analytic_pacing_controller_matches_direct_calls():
    # AnalyticPacingController.update must exactly reproduce calling
    # update_delivery_lambda/update_ctr_lambda directly with the same params.
    controller = AnalyticPacingController(eta_delivery=0.1, eta_ctr=0.15, lambda_delivery_max=1.5, lambda_ctr_max=1.5)
    args = dict(
        lambda_delivery=0.3,
        lambda_ctr=0.2,
        delivered=500,
        target_impressions=1000,
        elapsed_fraction=0.4,
        running_ctr=0.005,
        ctr_floor=0.006,
    )
    ld, lc = controller.update(**args)

    expected_ld = update_delivery_lambda(
        args["lambda_delivery"], args["delivered"], args["target_impressions"], args["elapsed_fraction"], eta=0.1, lambda_max=1.5
    )
    expected_lc = update_ctr_lambda(
        args["lambda_ctr"], args["running_ctr"], args["ctr_floor"], args["delivered"], eta=0.15, lambda_max=1.5
    )
    assert ld == expected_ld
    assert lc == expected_lc
