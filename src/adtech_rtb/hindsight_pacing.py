"""Hindsight-optimal pacing search + training-pair extraction -- Stage 1 of
the learned-pacing plan (plans/lucky-coalescing-crystal.md).

For one simulated flight, search a small grid of the SAME hand-tuned
constants pacing.py already exposes (pace_convexity, eta_delivery) instead
of hand-picking one, run the actual simulate_synthetic_flight per candidate,
and score each with hindsight_loss. A fully free per-step search was
considered and rejected (see the plan): each step's lambda choice changes
what the bandit's online GLMs learn downstream in the *same* flight, so
there's no fixed transition to run DP over, and it would just memorize one
flight's noise draw rather than learn a generalizable rule.

Every candidate's full trajectory is kept (not just the winner) --
extract_training_pairs relabels each step of each run with a per-step
"return-to-go" score (how well did the tail from this point onward do,
under that run's own fixed params) via _sub_result_from_suffix, which is
pure relabeling of already-simulated data, not additional simulation. This
is a deliberately lightweight, non-iterative version of fitted-value
reasoning -- filtered to well-scoring segments and used for direct
state->lambda imitation regression, NOT to fit a value/Q-function, which
would tip this from "supervised regression" into fitted-Q/offline-RL
territory (the complexity/instability class this plan was scoped to avoid).
"""

from __future__ import annotations

import numpy as np

from .bandit import BanditPolicy
from .pacing import ctr_error, pacing_error
from .synthetic import SyntheticEnvironment, simulate_synthetic_flight

# Deliberately small (2 axes x 3 values = 9 candidates -- see the plan's cost
# estimate, ~45-95 CPU-seconds per real flight run, so 9 candidates per
# training flight is already the dominant cost of the whole pipeline).
# eta_ctr/lambda_delivery_max/lambda_ctr_max stay at simulate_synthetic_flight's
# own defaults for this first version -- a reasonable v1 scope limit, not a
# claim that they don't matter; extending the grid to cover them is the
# natural Stage 5 stretch extension if v1's results look sensitive to it.
PACE_CONVEXITY_GRID = (0.7, 1.0, 1.5)
ETA_DELIVERY_GRID = (0.05, 0.1, 0.2)

# Delivery/CTR-floor are this project's hard constraints (the whole reason
# the bandit exists is to hit them while minimizing spend -- see bandit.py's
# ctr_floor framing); overrun/smoothness/cost are what's optimized subject to
# them. Weighting floor violations several times higher than the "nice to
# have" terms reflects that priority ordering directly. A first documented
# cut, not a swept optimum -- matches this project's own convention (e.g.
# pacing.py's eta=0.1) of shipping a reasoned starting point and revisiting
# only if downstream results look sensitive to it.
LOSS_WEIGHTS = {
    "shortfall": 3.0,
    "ctr_violation": 3.0,
    "overrun": 1.0,
    "delivery_cv": 0.5,
    "cpm_ratio": 1.0,
}


def default_param_grid() -> list[dict]:
    return [{"pace_convexity": pc, "eta_delivery": ed} for pc in PACE_CONVEXITY_GRID for ed in ETA_DELIVERY_GRID]


def hindsight_loss(result: dict, naive_cpm: float, weights: dict = LOSS_WEIGHTS) -> float:
    """Lower is better. Every term is relative/normalized (fractions of
    target/floor/naive-CPM, not raw RMB or impression counts) so they sit on
    a comparable scale despite very different native units -- see module
    docstring. Takes a plain `result`-shaped dict (works on both a real
    simulate_synthetic_flight result and the pseudo-result
    _sub_result_from_suffix builds for a trajectory suffix), plus a
    separately-supplied `naive_cpm` scalar (not part of `result`) so this
    stays a pure, easily hand-testable function.
    """
    delivered = result["delivered_impressions"]
    target = max(result["target_impressions"], 1)
    shortfall = max(0.0, 1.0 - delivered / target)
    overrun = max(0.0, result["overrun_ratio"] - 1.0)
    cv = result["delivery_cv"]
    cpm = result["spend"] / max(delivered, 1) * 1000.0
    cpm_ratio = (cpm - naive_cpm) / naive_cpm if naive_cpm > 0 else 0.0
    ctr_floor = max(result["ctr_floor"], 1e-9)
    ctr_violation = max(0.0, ctr_floor - result["achieved_ctr"]) / ctr_floor

    return (
        weights["shortfall"] * shortfall
        + weights["ctr_violation"] * ctr_violation
        + weights["overrun"] * overrun
        + weights["delivery_cv"] * cv
        + weights["cpm_ratio"] * cpm_ratio
    )


def search_hindsight_params(
    scenario: dict,
    environment: SyntheticEnvironment,
    naive_cpm: float,
    policy_kwargs: dict,
    param_grid: list[dict] | None = None,
    policy_seed: int = 0,
    outcome_seed: int | None = None,
    batch_size: int = 2000,
    max_overrun_multiple: float = 6.0,
) -> list[dict]:
    """Run simulate_synthetic_flight once per candidate in `param_grid`, all
    against the SAME scenario/outcome_seed/policy_seed for a fair
    apples-to-apples comparison. Returns every run (`{"params", "loss",
    "result"}`), sorted best-first -- not just the winner, since
    extract_training_pairs reuses non-winning candidates' trajectories too.
    """
    if param_grid is None:
        param_grid = default_param_grid()
    if outcome_seed is None:
        outcome_seed = scenario["seed"]

    runs = []
    for params in param_grid:
        policy = BanditPolicy(ctr_floor=scenario["ctr_floor"], seed=policy_seed, first_price=True, **policy_kwargs)
        result = simulate_synthetic_flight(
            scenario,
            policy,
            environment,
            batch_size=batch_size,
            outcome_seed=outcome_seed,
            max_overrun_multiple=max_overrun_multiple,
            track_trajectory=True,
            **params,
        )
        loss = hindsight_loss(result, naive_cpm)
        runs.append({"params": params, "loss": loss, "result": result})

    runs.sort(key=lambda r: r["loss"])
    return runs


def _sub_result_from_suffix(trajectory: list[dict], start_idx: int, scenario: dict, final_result: dict) -> dict:
    """Reconstruct a hindsight_loss-compatible pseudo-result representing
    "as if a fresh sub-flight had started right after trajectory[start_idx-1]
    (or at time 0 if start_idx==0) and continued exactly as this run's tail
    actually played out." No extra simulation: the tail already happened, in
    this run, under one fixed set of params -- this just re-scores it as its
    own shorter flight. `nominal_days_remaining` is the remaining share of
    the ORIGINAL nominal flight length (not re-derived from the sub-target),
    so a sub-flight starting late still has a realistically small runway.
    """
    tail = trajectory[start_idx:]
    if start_idx == 0:
        base_delivered, base_spend, base_clicks, base_days = 0, 0.0, 0, 0.0
    else:
        prev = trajectory[start_idx - 1]
        base_delivered = prev["cumulative_delivered"]
        base_spend = prev["cumulative_spend"]
        base_clicks = prev["cumulative_clicks"]
        base_days = prev["days_used"]

    final_days = final_result["days_used"]
    sub_delivered = final_result["delivered_impressions"] - base_delivered
    sub_spend = final_result["spend"] - base_spend
    sub_clicks = final_result["clicks"] - base_clicks
    sub_days = max(final_days - base_days, 1e-9)

    nominal_days = scenario["flight_length_days"]
    elapsed_fraction_at_start = min(base_days / nominal_days, 1.0) if nominal_days else 0.0
    nominal_days_remaining = max(nominal_days * (1.0 - elapsed_fraction_at_start), 1e-6)

    # Daily buckets re-derived from this tail's own per-batch records only
    # (not the whole flight's) -- each trajectory entry already carries its
    # own batch's win count (batch_won) and elapsed days (days_used).
    daily: dict[int, int] = {}
    for step in tail:
        day_idx = int(step["days_used"])
        daily[day_idx] = daily.get(day_idx, 0) + step["batch_won"]
    daily_counts = np.array(list(daily.values()), dtype=float)
    sub_cv = float(daily_counts.std() / daily_counts.mean()) if len(daily_counts) > 1 and daily_counts.mean() > 0 else 0.0

    return {
        "delivered_impressions": sub_delivered,
        "target_impressions": max(1, scenario["target_impressions"] - base_delivered),
        "spend": sub_spend,
        "overrun_ratio": sub_days / nominal_days_remaining,
        "delivery_cv": sub_cv,
        "ctr_floor": scenario["ctr_floor"],
        "achieved_ctr": sub_clicks / sub_delivered if sub_delivered > 0 else 0.0,
    }


STATE_FEATURE_NAMES = ("elapsed_fraction", "pacing_error", "ctr_error", "delivered_fraction")


def extract_training_pairs(
    runs: list[dict], scenario: dict, naive_cpm: float, loss_percentile: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel every candidate run's trajectory (not just the best-scoring
    one) with a per-step return-to-go hindsight_loss (via
    _sub_result_from_suffix), keep only the better-scoring half (below
    `loss_percentile`), and return `(X, y)` -- `X` columns match
    STATE_FEATURE_NAMES, `y` is `[lambda_delivery, lambda_ctr]` -- ready for
    direct state->lambda imitation regression (see module docstring for why
    this stops short of a value/Q-function fit).
    """
    rows: list[tuple[float, np.ndarray, float, float]] = []
    for run in runs:
        trajectory = run["result"]["trajectory"]
        if not trajectory:
            continue
        pace_convexity = run["params"].get("pace_convexity", 1.0)
        for i, step in enumerate(trajectory):
            sub_result = _sub_result_from_suffix(trajectory, i, scenario, run["result"])
            loss = hindsight_loss(sub_result, naive_cpm)

            elapsed_fraction = step["days_used"] / scenario["flight_length_days"] if scenario["flight_length_days"] else 0.0
            p_err = pacing_error(step["cumulative_delivered"], scenario["target_impressions"], elapsed_fraction, pace_convexity)
            c_err = ctr_error(step["running_ctr"], scenario["ctr_floor"], step["cumulative_delivered"])
            features = np.array(
                [
                    min(elapsed_fraction, 3.0),
                    p_err,
                    c_err,
                    step["cumulative_delivered"] / max(scenario["target_impressions"], 1),
                ]
            )
            rows.append((loss, features, step["lambda_delivery"], step["lambda_ctr"]))

    if not rows:
        return np.empty((0, len(STATE_FEATURE_NAMES))), np.empty((0, 2))

    losses = np.array([r[0] for r in rows])
    threshold = np.percentile(losses, loss_percentile)
    keep = losses <= threshold

    X = np.array([r[1] for r in rows])[keep]
    y = np.array([[r[2], r[3]] for r in rows])[keep]
    return X, y
