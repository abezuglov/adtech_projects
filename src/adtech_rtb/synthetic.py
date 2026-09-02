"""Phase 5 synthetic environment: first-price auctions over a frozen
`World` (synthetic_world.py), plus a self-contained sequential flight
simulator and naive-baseline solver mirroring simulator.py/bidding.py's
real-data machinery -- deliberately NOT sharing code with those modules
(see the Phase 5 plan's rationale: real data's pool-exhaustion/bootstrap-
extension logic is the most complex, failure-prone part of simulator.py,
and doesn't exist here at all since synthetic generation is unlimited, so
forcing both through one abstraction would touch already-validated
real-data code for little reuse benefit).

First-price, not GSP: you pay exactly what you bid on a win. Real iPinYou
data's GSP pricing made bid *level* a non-decision (pay is independent of
your own bid, so bidding the max level within budget is always at least as
good) -- first-price makes it a genuine cost/quality trade-off ("bid
shading"), which is the whole point of this phase.

Context schema is deliberately small (`placement`, `campaign_id`, `hour`)
-- not a copy of the real iPinYou schema's dozen columns. `campaign_id`
enters ONLY through CTR (true_win_prob has no campaign argument at all):
auction economics are a property of the placement, not who's bidding,
mirroring market_model.py's own campaign-identity exclusion but pushed all
the way into the generative model itself rather than just the feature list.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import pacing
from .bandit import DEFAULT_BID_BOUNDS
from .synthetic_world import World, campaign_affinity_vector, diurnal_ctr_shift, diurnal_price_shift

MARKET_CATEGORICAL_COLUMNS = ["placement"]
CTR_CATEGORICAL_COLUMNS = ["placement", "campaign_id"]
NUMERIC_COLUMNS = ["hour"]

FLIGHT_LENGTH_DAYS_RANGE = (14, 28)
# Narrower than scenarios.py's real-data (0.3, 0.85): that range was
# calibrated against the real fitted model's near-flat win-rate curve
# (0.274-0.288 across the whole supported bid range), where every point in
# (0.3, 0.85) landed close to the same achievable rate regardless. This
# world's win-rate curve is wide and steep by design (~0.10 to ~0.92,
# confirmed on the first scenario tried) -- reusing (0.3, 0.85) unchanged
# made even the *low* end of the range demand a ~35% population win rate
# and the high end ~80%, which SYNTHETIC_LAMBDA_DELIVERY_MAX=0.5 (tuned to
# preserve price differentiation, see above) couldn't reliably clear within
# a bounded overrun -- confirmed on synth-campA-1: delivered only 96% of
# target even at 3x nominal flight length. (0.1, 0.5) keeps targets
# meaningfully demanding (real delivery pressure, real pacing behavior to
# observe) without requiring near-majority population win rates on a curve
# this wide.
#
# That 0.5-cap ceiling no longer applies (see SYNTHETIC_LAMBDA_DELIVERY_MAX's
# 2026-09-01 note below) -- not revisited since the new 1.5 cap clears this
# whole range comfortably (proxy sweep: win_rate_fraction=0.15 finishes at
# 0.99x nominal length; 0.85, well above this range's ceiling, at 1.24x).
TARGET_WIN_RATE_FRACTION = (0.1, 0.5)
CTR_FLOOR_FRACTION = (0.5, 0.9)

# Synthetic auction-opportunity rate, shared across all campaigns (unlike
# the real pipeline, campaign identity doesn't affect auction eligibility
# here -- every campaign sees the same placement universe/traffic, only
# affinity differs). An arbitrary but fixed assumption (120K/day) in the
# same order of magnitude as several real scenarios' n_eligible_auctions,
# used both for generate_synthetic_scenarios' target-impressions math and
# simulate_synthetic_flight's pacing clock.
AUCTIONS_PER_HOUR = 5_000

MC_POOL_SIZE = 200_000  # Monte-Carlo sample size for closed-form expectations (scenario generation, naive baseline)

# pacing.py's LAMBDA_DELIVERY_MAX/LAMBDA_CTR_MAX (1.5) are calibrated for
# real GSP data, where cost is level-independent and the cap's magnitude
# barely matters (see pacing.update_delivery_lambda's docstring). Under
# first-price, cost genuinely spans bandit.BID_LEVELS' full
# bandit.DEFAULT_BID_BOUNDS range, (320-200)/1000 = 0.12 RMB/impression --
# confirmed empirically that reusing 1.5 here swamps that span almost
# immediately once behind pace, collapsing bid-level choice back to
# "always maximize win probability" regardless of price (the bandit came
# out *worse* than the naive baseline, 295 vs 253 CPM, on the first
# validation scenario tried). THAT finding, and the 0.5 chosen from it, both
# predate pacing.PACE_CONVEXITY's 2026-09-01 change from 3.0 to 1.5 -- see
# below, since the swamping threshold turned out to depend on both together,
# not on this cap alone.
#
# RAISED 0.5 -> 1.5 (2026-09-01): 0.5's own empirical sweep (0.2-1.5, notes
# below) never actually escaped the problem it was chosen to solve -- it just
# capped its severity. Trajectory instrumentation on generate_scenario_grid.py's
# "challenging" cells (win_rate_fraction=0.85, near the achievable ceiling)
# showed lambda_delivery saturating at 0.5 within the first ~18% of the
# flight and then sitting flat, unable to escalate further no matter how far
# behind pace delivery fell -- a structurally unrecoverable deficit, not a
# calibration nuance (confirmed via a 4.38x nominal-length overrun even
# though PACE_CONVEXITY's OLD backloaded curve, see pacing.py, meant urgency
# should have had room to still rise late in the flight). The cap itself,
# not the curve shape, was the actual bottleneck. Once PACE_CONVEXITY dropped
# to 1.5 (escalating lambda_delivery earlier and more gradually, rather than
# holding it near-zero until a late, steep catch-up), the swamping threshold
# this constant was originally calibrated against moved: a paired
# easy/challenging proxy sweep of this constant (0.5/0.8/1.1/1.5/2.0/2.5/3.0/
# 4.0) crossed with PACE_CONVEXITY (1.5/3.0) found overrun improving
# monotonically with the cap at PACE_CONVEXITY=1.5 -- 3.49x/1.21x
# (challenging/easy) at 0.5, down to 1.24x/0.99x at 1.5, with CPM still
# within a few percent of the naive baseline on the hard cell and beating it
# by ~17% on the easy one -- then only marginal further overrun improvement
# (1.24x -> 1.17x from 1.5 -> 4.0) at a steadily worsening CPM cost, so 1.5
# is the point past which the cap stops being what's limiting delivery.
# Landing on 1.5 exactly (matching pacing.py's real-data constants) is
# coincidental, not a simplification to rely on -- keep this a separate,
# environment-calibrated constant rather than importing pacing.LAMBDA_
# DELIVERY_MAX/LAMBDA_CTR_MAX directly, since a future change to either
# environment's calibration shouldn't silently drag the other along.
#
# 0.5 was chosen from an empirical sweep (0.2-1.5) on a small scenario
# against three criteria: (1) can it actually hit the delivery target
# within a bounded overrun at all -- 0.2-0.3 deadlocked or badly
# under-delivered, the cost floor (200/1000=0.2) leaves no room to ever
# justify bidding once lambda_delivery can't exceed it; (2) does it beat
# the naive baseline's CPM -- 0.5 gave the best margin (+6.8%) of every
# value tried, degrading smoothly to a *worse*-than-naive result by 0.7+
# as the cap increasingly swamps cost differentiation again; (3) is
# delivery actually smooth, not just "eventually met" -- checked via
# delivery_cv below (a low CV, ~0.4, with only the expected initial
# cold-start dead zone, not scattered gaps later) even though 0.5
# overruns the nominal flight length by ~2x on a delivery-tight scenario:
# it runs a bit long at a steady pace, not lumpily. That sweep was run
# under the OLD PACE_CONVEXITY=3.0 and never tested a near-ceiling target
# (win_rate_fraction up to TARGET_WIN_RATE_FRACTION's 0.5, not 0.85) --
# see the 2026-09-01 note above for why it undersold the achievable cap.
SYNTHETIC_LAMBDA_DELIVERY_MAX = 1.5
SYNTHETIC_LAMBDA_CTR_MAX = 1.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def true_win_prob(world: World, placement: np.ndarray, hour: np.ndarray, bid) -> np.ndarray:
    """P(win | bid, placement, hour), true generating process. No campaign
    argument (deliberate -- see module docstring). Monotone increasing in
    bid by construction (world.beta > 0, guaranteed by its log-normal prior
    in synthetic_world.py, not a fitted monotone constraint)."""
    clearing = world.clearing_level[placement] + diurnal_price_shift(world, hour)
    beta = world.beta[placement]
    return _sigmoid(beta * (np.asarray(bid, dtype=float) - clearing))


def true_ctr(world: World, campaign_id, placement: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """P(click | won, campaign, placement, hour), true generating process.
    Independent of bid (mirrors ctr_model.py's own convention). The only
    place campaign_id enters the generative model."""
    affinity = campaign_affinity_vector(world, campaign_id)
    logit = world.ctr_base[placement] + affinity[placement] + diurnal_ctr_shift(world, hour)
    return _sigmoid(logit)


def _mc_pool(world: World, campaign_id, rng: np.random.Generator, n: int) -> pd.DataFrame:
    """A large iid Monte-Carlo sample of (placement, hour) pairs, weighted
    by the world's own right-skewed placement_weight -- used for closed-
    form population-level expectations (scenario generation, naive
    baseline), analogous to bidding.py's real 2-day auction pool but drawn
    fresh since there's no fixed real pool here."""
    placement = rng.choice(len(world.clearing_level), size=n, p=world.placement_weight)
    hour = rng.uniform(0.0, 24.0, size=n)
    return pd.DataFrame({"placement": placement, "campaign_id": campaign_id, "hour": hour})


class SyntheticEnvironment:
    """First-price auction environment over a frozen World. `.sample_context`
    draws fresh auction-opportunity rows on demand -- unlimited, no pool to
    exhaust (unlike the real pipeline's bootstrap-from-2-days machinery).
    `.resolve` draws the TRUE outcomes a bandit could only ever observe
    realistically; the bandit itself never sees `world` directly (mirrors
    simulator.py's real-data environment/policy split -- see bandit.py's
    own module docstring for why that separation matters).
    """

    def __init__(self, world: World):
        self.world = world

    def sample_context(self, campaign_id, rng: np.random.Generator, n: int, hour: float) -> pd.DataFrame:
        placement = rng.choice(len(self.world.clearing_level), size=n, p=self.world.placement_weight)
        return pd.DataFrame(
            {
                "placement": placement,
                "campaign_id": [campaign_id] * n,
                "hour": np.full(n, hour % 24.0),
            }
        )

    def resolve(
        self, contexts: pd.DataFrame, chosen_bids: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(contexts)
        won = np.zeros(n, dtype=bool)
        price_paid = np.zeros(n)
        clicked = np.zeros(n, dtype=bool)

        bid_mask = chosen_bids > 0
        if bid_mask.any():
            sub = contexts.loc[bid_mask]
            win_prob = true_win_prob(self.world, sub["placement"].to_numpy(), sub["hour"].to_numpy(), chosen_bids[bid_mask])
            win_draw = rng.random(len(win_prob)) < win_prob
            won[np.where(bid_mask)[0][win_draw]] = True

        if won.any():
            # First-price: pay exactly what you bid on a win -- no price
            # model/prediction involved at all, cost is deterministic given
            # your own chosen level (see bandit.py's first_price mode).
            price_paid[won] = chosen_bids[won]
            won_sub = contexts.loc[won]
            # Scenarios are single-campaign (see generate_synthetic_scenarios),
            # so campaign_id is constant across `contexts` -- safe to read once.
            campaign_id = contexts["campaign_id"].iat[0]
            ctr = true_ctr(self.world, campaign_id, won_sub["placement"].to_numpy(), won_sub["hour"].to_numpy())
            click_draw = rng.random(len(ctr)) < ctr
            clicked[np.where(won)[0][click_draw]] = True

        return won, price_paid, clicked


def simulate_synthetic_flight(
    scenario: dict,
    policy,
    environment: SyntheticEnvironment,
    batch_size: int = 2000,
    outcome_seed: int = 0,
    max_overrun_multiple: float = 3.0,
    track_trajectory: bool = False,
    pace_convexity: float = pacing.PACE_CONVEXITY,
    eta_delivery: float = 0.1,
    eta_ctr: float = 0.15,
    lambda_delivery_max: float = SYNTHETIC_LAMBDA_DELIVERY_MAX,
    lambda_ctr_max: float = SYNTHETIC_LAMBDA_CTR_MAX,
    pacing_controller=None,
) -> dict:
    """Sequential auction replay against a SyntheticEnvironment. Same
    result-dict shape as simulator.simulate_flight for direct comparability,
    but a separately-implemented, simpler loop -- see module docstring for
    why this isn't shared code with the real-data simulator.

    One batch = one simulated hour's worth of auction opportunities
    (AUCTIONS_PER_HOUR-scaled, not `batch_size` itself, so pacing's
    elapsed_fraction reflects the scenario's own generation assumptions
    regardless of what batch_size a given run happens to use).

    `pace_convexity`/`eta_delivery`/`eta_ctr`/`lambda_delivery_max`/
    `lambda_ctr_max` default to exactly what this function hardcoded before
    (pacing.py's own defaults + this module's SYNTHETIC_LAMBDA_*_MAX) --
    exposed as parameters so hindsight_pacing.py's grid search can vary them
    per candidate without monkeypatching module globals. This is a narrower,
    earlier seam than the full pacing_controller swap: varying these five
    scalars covers the analytic formula's own hyperparameters, which is all
    the hindsight search needs.

    `pacing_controller`, if given (a pacing.AnalyticPacingController or
    learned_pacing.LearnedPacingController -- anything with a matching
    `.update(...)` signature), REPLACES the pace_convexity/eta_*/lambda_*_max
    parameters above entirely for computing the next lambda_delivery/
    lambda_ctr. Default `None` preserves this function's exact prior
    behavior (byte-for-byte) via the five scalar parameters -- this is
    Stage 4 of plans/lucky-coalescing-crystal.md's plug-in point, letting
    scripts/run_synthetic_bandit.py's `--pacing {analytic,learned}` flag
    swap controllers without touching this function's default call path.
    """
    target = scenario["target_impressions"]
    nominal_days = scenario["flight_length_days"]
    nominal_hours = nominal_days * 24
    ctr_floor = scenario["ctr_floor"]
    campaign_id = scenario["campaign_id"]

    rng = np.random.default_rng(outcome_seed)

    delivered = 0
    spend = 0.0
    clicks = 0
    hours_used = 0.0
    bid_level_counts: dict[float, int] = {}
    daily_delivered: dict[int, int] = {}  # smoothness diagnostic, see delivery_cv below
    trajectory = []
    capped = False

    while delivered < target:
        if hours_used >= nominal_hours * max_overrun_multiple:
            capped = True
            break

        hour_of_day = hours_used % 24.0
        contexts = environment.sample_context(campaign_id, rng, batch_size, hour_of_day)

        chosen_bids = policy.choose_bids(contexts)
        batch_level_counts: dict[float, int] = {}
        for level, count in zip(*np.unique(chosen_bids[chosen_bids > 0], return_counts=True)):
            level = float(level)
            count = int(count)
            bid_level_counts[level] = bid_level_counts.get(level, 0) + count
            batch_level_counts[level] = count

        won, price_paid, clicked = environment.resolve(contexts, chosen_bids, rng)

        batch_delivered = int(won.sum())
        delivered += batch_delivered
        # RMB/CPM -> RMB per single impression, same convention as the real pipeline.
        spend += float(price_paid[won].sum()) / 1000.0
        clicks += int(clicked.sum())
        daily_delivered[int(hours_used // 24)] = daily_delivered.get(int(hours_used // 24), 0) + batch_delivered

        policy.observe(contexts, chosen_bids, won.astype(float), clicked.astype(float), price_paid)

        hours_used += batch_size / AUCTIONS_PER_HOUR
        elapsed_fraction = hours_used / nominal_hours
        running_ctr = clicks / delivered if delivered > 0 else 0.0
        if pacing_controller is None:
            policy.lambda_delivery = pacing.update_delivery_lambda(
                policy.lambda_delivery,
                delivered,
                target,
                elapsed_fraction,
                eta=eta_delivery,
                lambda_max=lambda_delivery_max,
                pace_convexity=pace_convexity,
            )
            policy.lambda_ctr = pacing.update_ctr_lambda(
                policy.lambda_ctr, running_ctr, ctr_floor, delivered, eta=eta_ctr, lambda_max=lambda_ctr_max
            )
        else:
            policy.lambda_delivery, policy.lambda_ctr = pacing_controller.update(
                policy.lambda_delivery, policy.lambda_ctr, delivered, target, elapsed_fraction, running_ctr, ctr_floor
            )

        if track_trajectory:
            batch_won = int(won.sum())
            batch_spend = float(price_paid[won].sum()) / 1000.0
            trajectory.append(
                {
                    "days_used": hours_used / 24.0,
                    "cumulative_delivered": delivered,
                    "cumulative_spend": spend,
                    "cumulative_clicks": clicks,
                    "running_ctr": running_ctr,
                    "batch_size": len(contexts),
                    "batch_won": batch_won,
                    "batch_cpm": (batch_spend / batch_won * 1000.0) if batch_won > 0 else None,
                    "batch_bid_level_counts": batch_level_counts,
                    "lambda_delivery": policy.lambda_delivery,
                    "lambda_ctr": policy.lambda_ctr,
                }
            )

    achieved_ctr = clicks / delivered if delivered > 0 else 0.0
    days_used = hours_used / 24.0

    # Delivery smoothness: coefficient of variation of per-day delivered
    # volume -- "eventually hits the target" isn't the same as "delivers
    # steadily," and a real campaign owner cares about the latter too (a
    # criterion raised mid-tuning, not in the original plan; tracked here
    # unconditionally, unlike `trajectory`, so it's always reportable
    # without the batch-level detail's size cost). The naive flat-bid
    # baseline is smooth by construction (a closed-form constant rate), so
    # this is really a bandit-specific number worth contrasting against
    # that implicit CV-of-0. The last bucket can be a partial day (delivery
    # stops mid-day once the target is met), which mechanically lowers it
    # and modestly inflates CV -- a known, minor bias in this diagnostic,
    # not corrected for since it only ever makes reported smoothness look
    # slightly worse than the truth, never better.
    daily_counts = np.array(list(daily_delivered.values()), dtype=float)
    delivery_cv = float(daily_counts.std() / daily_counts.mean()) if len(daily_counts) > 1 and daily_counts.mean() > 0 else 0.0

    return {
        "scenario_id": scenario["id"],
        "delivered_impressions": delivered,
        "target_impressions": target,
        "delivery_met": delivered >= target,
        "delivery_capped": capped,
        "spend": spend,
        "ctr_floor": ctr_floor,
        "achieved_ctr": achieved_ctr,
        "ctr_met": achieved_ctr >= ctr_floor,
        "clicks": clicks,
        "days_used": days_used,
        "flight_length_days": nominal_days,
        "overrun_ratio": days_used / nominal_days,
        "bid_level_counts": bid_level_counts,
        "delivery_cv": delivery_cv,
        "trajectory": trajectory if track_trajectory else None,
    }


def evaluate_flat_bid_synthetic(
    world: World, campaign_id, bid: float, n_flight: int, rng: np.random.Generator, mc_pool_size: int = MC_POOL_SIZE
) -> dict:
    """Expected delivery/spend/CTR a constant `bid` produces over a flight
    of `n_flight` auctions -- closed-form population expectation over a
    Monte-Carlo pool, mirroring bidding.evaluate_flat_bid. First-price, so
    expected spend per auction is simply win_prob * bid (no separate price
    prediction needed, unlike the real GSP-mode version)."""
    pool = _mc_pool(world, campaign_id, rng, mc_pool_size)
    placement = pool["placement"].to_numpy()
    hour = pool["hour"].to_numpy()
    win_probs = true_win_prob(world, placement, hour, bid)
    ctr_preds = true_ctr(world, campaign_id, placement, hour)

    expected_impressions = float(win_probs.mean()) * n_flight
    expected_spend = float(win_probs.mean()) * bid / 1000.0 * n_flight
    expected_clicks = float(np.mean(win_probs * ctr_preds)) * n_flight
    achieved_ctr = expected_clicks / expected_impressions if expected_impressions > 0 else 0.0

    return {
        "bid": bid,
        "expected_impressions": expected_impressions,
        "expected_spend": expected_spend,
        "expected_clicks": expected_clicks,
        "achieved_ctr": achieved_ctr,
    }


def solve_delivery_bid_synthetic(
    world: World,
    campaign_id,
    target_impressions: int,
    n_flight: int,
    rng: np.random.Generator,
    bid_bounds: tuple[float, float] = DEFAULT_BID_BOUNDS,
    tol: float = 1e-4,
    max_iter: int = 40,
    mc_pool_size: int = MC_POOL_SIZE,
    bid_levels: np.ndarray | None = None,
) -> dict:
    """Bisection search for the fixed bid expected to deliver
    `target_impressions` over a flight of `n_flight` auctions -- mirrors
    bidding.solve_delivery_bid; valid since true_win_prob is monotone in
    bid by construction (world.beta > 0), not merely a fitted constraint.

    `bid_levels`, if given, restricts the search to the cheapest of those
    discrete levels whose expected rate still clears the target -- for a
    fair comparison against a bandit that's itself confined to a discrete
    grid (bandit.BID_LEVELS). Continuous bisection (the default) hands the
    naive baseline a degree of bid-precision no discretized policy has,
    which was found to explain a real chunk of the apparent naive-vs-bandit
    CPM gap (see reports/pacing_comparison.json's discussion) -- this isn't
    a hypothetical concern, it materially changes the comparison."""
    pool = _mc_pool(world, campaign_id, rng, mc_pool_size)
    placement = pool["placement"].to_numpy()
    hour = pool["hour"].to_numpy()
    target_rate = target_impressions / n_flight

    def rate_at(bid: float) -> float:
        return float(true_win_prob(world, placement, hour, bid).mean())

    if bid_levels is not None:
        levels = np.sort(np.asarray(bid_levels))
        rates = np.array([rate_at(level) for level in levels])
        eligible = np.where(rates >= target_rate)[0]
        idx = int(eligible[0]) if len(eligible) > 0 else int(np.argmax(rates))
        bid = float(levels[idx])
        achieved_rate = float(rates[idx])
        return {
            "bid": bid,
            "n_eligible": n_flight,
            "target_impressions": target_impressions,
            "target_win_rate": target_rate,
            "achieved_win_rate": achieved_rate,
            "expected_impressions": achieved_rate * n_flight,
            "target_reachable_in_bounds": bool(target_rate <= rates.max()),
        }

    lo, hi = bid_bounds
    rate_lo, rate_hi = rate_at(lo), rate_at(hi)

    if target_rate <= rate_lo:
        bid = lo
    elif target_rate >= rate_hi:
        bid = hi
    else:
        bid = (lo + hi) / 2
        for _ in range(max_iter):
            bid = (lo + hi) / 2
            rate = rate_at(bid)
            if abs(rate - target_rate) < tol:
                break
            if rate < target_rate:
                lo = bid
            else:
                hi = bid

    achieved_rate = rate_at(bid)
    return {
        "bid": bid,
        "n_eligible": n_flight,
        "target_impressions": target_impressions,
        "target_win_rate": target_rate,
        "achieved_win_rate": achieved_rate,
        "expected_impressions": achieved_rate * n_flight,
        "target_reachable_in_bounds": target_rate < rate_hi,
    }


def population_rate_bounds(
    world: World, campaign_id, rng: np.random.Generator, mc_pool_size: int = MC_POOL_SIZE
) -> tuple[float, float, float]:
    """(rate_floor, rate_ceiling, mean_ctr): the achievable population-level
    win-rate band at DEFAULT_BID_BOUNDS' two ends, plus mean CTR, for
    `campaign_id` -- shared by generate_synthetic_scenarios and any other
    scenario generator that needs to place a delivery/CTR target relative to
    what this world can actually produce."""
    pool = _mc_pool(world, campaign_id, rng, mc_pool_size)
    placement = pool["placement"].to_numpy()
    hour = pool["hour"].to_numpy()
    rate_floor = float(true_win_prob(world, placement, hour, DEFAULT_BID_BOUNDS[0]).mean())
    rate_ceiling = float(true_win_prob(world, placement, hour, DEFAULT_BID_BOUNDS[1]).mean())
    mean_ctr = float(true_ctr(world, campaign_id, placement, hour).mean())
    return rate_floor, rate_ceiling, mean_ctr


def generate_synthetic_scenarios(
    world: World,
    campaign_ids: list,
    n_per_campaign: int = 3,
    seed: int = 42,
    mc_pool_size: int = MC_POOL_SIZE,
) -> list[dict]:
    """Analogous to scenarios.iter_scenarios, but computed from the closed-
    form generator instead of querying a LightGBM booster over a real pool.
    Each scenario spans the full N_PLACEMENTS-placement universe for its
    campaign (no geo subset -- right-skewed placement_weight already gives
    a natural head/long-tail split without one)."""
    scenarios = []
    for campaign_id in campaign_ids:
        for index in range(1, n_per_campaign + 1):
            scenario_id = f"synth-{campaign_id}-{index}"
            rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(str(campaign_id).encode()), index]))

            flight_length_days = int(rng.integers(FLIGHT_LENGTH_DAYS_RANGE[0], FLIGHT_LENGTH_DAYS_RANGE[1] + 1))
            n_eligible_auctions = AUCTIONS_PER_HOUR * 24 * flight_length_days

            rate_floor, rate_ceiling, mean_ctr = population_rate_bounds(world, campaign_id, rng, mc_pool_size)

            target_rate = rate_floor + rng.uniform(*TARGET_WIN_RATE_FRACTION) * (rate_ceiling - rate_floor)
            target_impressions = max(1, round(target_rate * n_eligible_auctions))

            ctr_floor = rng.uniform(*CTR_FLOOR_FRACTION) * mean_ctr

            outcome_seed = int(rng.integers(0, 2**31 - 1))

            scenarios.append(
                {
                    "id": scenario_id,
                    "campaign_id": campaign_id,
                    "flight_length_days": flight_length_days,
                    "seed": outcome_seed,
                    "n_eligible_auctions": n_eligible_auctions,
                    "target_impressions": target_impressions,
                    "ctr_floor": round(ctr_floor, 6),
                    "population_mean_ctr": round(mean_ctr, 6),
                    "naive_baseline_reachable": bool(target_rate < rate_ceiling),
                }
            )
    return scenarios
