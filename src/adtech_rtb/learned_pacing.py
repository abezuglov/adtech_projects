"""LearnedPacingController -- Stages 3/4 of the learned-pacing plan
(plans/lucky-coalescing-crystal.md).

Wraps a frozen linear map from pacing state to (lambda_delivery, lambda_ctr),
fit once offline by scripts/fit_learned_pacing.py from
hindsight_pacing.py's training pairs. Frozen at construction: this class
never updates its own weights during an eval flight -- online-updating the
pacing model mid-flight would reopen the exact feedback-loop/non-stationarity
concern "supervised regression, not RL" was chosen to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pacing import ctr_error, pacing_error

# Must stay in sync with hindsight_pacing.STATE_FEATURE_NAMES / its feature
# construction in extract_training_pairs -- both are responsible for the
# same feature order/definition, since a model fit on one and evaluated with
# the other would silently score wrong-column features against each weight.
#
# `overrun_x_pacing_error`/`overrun_x_ctr_error` ADDED (2026-09-02): found via
# grid-90d-highctr-challenging's trajectory that lambda_delivery/lambda_ctr
# both collapse right as elapsed_fraction crosses 1.0, even 20% short of
# target -- a linear model can only give elapsed_fraction and pacing_error
# additive, independent effects, so it can't represent "once overdue, being
# behind should matter MORE, not less" (needed only in that AND-of-two-
# conditions regime, everywhere else elapsed_fraction rising while on pace
# correctly signals no extra reaction is needed). These two interaction
# terms are exactly 0 while elapsed_fraction<=1 (no effect on any
# already-validated on-time behavior) and only activate once BOTH overdue
# and still behind/below floor, giving the fit a dedicated slope for that
# combination instead of forcing it through the same two independent terms
# used everywhere else.
STATE_FEATURE_NAMES = (
    "elapsed_fraction",
    "pacing_error",
    "pacing_error_sq_behind",
    "ctr_error",
    "delivered_fraction",
    "overrun_x_pacing_error",
    "overrun_x_ctr_error",
)


def pacing_state_features(
    delivered: int,
    target_impressions: int,
    elapsed_fraction: float,
    running_ctr: float,
    ctr_floor: float,
    pace_convexity: float = 1.0,
) -> np.ndarray:
    """The same 7 features hindsight_pacing.extract_training_pairs computes
    per trajectory step, computed here from live simulate_synthetic_flight
    state instead of a stored trajectory. Reuses pacing.py's own
    pacing_error/ctr_error rather than reimplementing them, so a future
    change to either formula can't silently drift between training and
    inference. `pacing_error_sq_behind` (see hindsight_pacing.py's
    extract_training_pairs for the full rationale) lets the fit react
    superlinearly once genuinely far behind pace, without changing its
    behavior in the mild-deficit regime a single linear pacing_error term
    already handles reasonably. `overrun_x_pacing_error`/`overrun_x_ctr_error`
    (see STATE_FEATURE_NAMES's own comment) give the fit a dedicated slope
    for "overdue AND still behind/below floor" instead of relying on
    elapsed_fraction and pacing_error/ctr_error's independent, additive
    effects to somehow cover that combination too.
    """
    p_err = pacing_error(delivered, target_impressions, elapsed_fraction, pace_convexity)
    c_err = ctr_error(running_ctr, ctr_floor, delivered)
    p_err_sq_behind = max(0.0, p_err) ** 2
    overrun_fraction = max(0.0, elapsed_fraction - 1.0)
    return np.array(
        [
            min(elapsed_fraction, 3.0),
            p_err,
            p_err_sq_behind,
            c_err,
            delivered / max(target_impressions, 1),
            overrun_fraction * max(0.0, p_err),
            overrun_fraction * max(0.0, c_err),
        ]
    )


class LearnedPacingController:
    """Matches AnalyticPacingController's `.update(...)` interface (see
    pacing.py) so simulate_synthetic_flight's `pacing_controller` seam can
    swap between them without any other code change.
    """

    def __init__(
        self,
        delivery_weights,
        ctr_weights,
        lambda_delivery_max: float,
        lambda_ctr_max: float,
        pace_convexity: float = 1.0,
    ):
        self.delivery_weights = np.asarray(delivery_weights, dtype=float)
        self.ctr_weights = np.asarray(ctr_weights, dtype=float)
        self.lambda_delivery_max = lambda_delivery_max
        self.lambda_ctr_max = lambda_ctr_max
        self.pace_convexity = pace_convexity

    def update(
        self,
        lambda_delivery: float,
        lambda_ctr: float,
        delivered: int,
        target_impressions: int,
        elapsed_fraction: float,
        running_ctr: float,
        ctr_floor: float,
    ) -> tuple[float, float]:
        features = pacing_state_features(
            delivered, target_impressions, elapsed_fraction, running_ctr, ctr_floor, self.pace_convexity
        )
        x = np.concatenate([[1.0], features])
        # Hard safety clip, matching this project's belt-and-suspenders
        # clamping style elsewhere (OnlineBayesianLinearModel.update,
        # update_delivery_lambda/update_ctr_lambda) -- a linear fit has no
        # built-in bound the way a clipped analytic formula does.
        new_lambda_delivery = float(np.clip(x @ self.delivery_weights, 0.0, self.lambda_delivery_max))
        new_lambda_ctr = float(np.clip(x @ self.ctr_weights, 0.0, self.lambda_ctr_max))
        return new_lambda_delivery, new_lambda_ctr

    @classmethod
    def load(
        cls, path: Path, lambda_delivery_max: float, lambda_ctr_max: float, pace_convexity: float = 1.0
    ) -> "LearnedPacingController":
        with open(path) as f:
            model = json.load(f)
        return cls(model["delivery_weights"], model["ctr_weights"], lambda_delivery_max, lambda_ctr_max, pace_convexity)
