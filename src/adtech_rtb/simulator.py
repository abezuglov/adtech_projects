"""Sequential, policy-agnostic auction replay for one scenario.

Unlike bidding.evaluate_flat_bid (a closed-form population expectation,
valid because a flat bid's outcome doesn't depend on order or state), a
contextual policy's decisions depend on evolving state (pacing debt,
running CTR) -- so this genuinely has to step through auctions in
chronological order, batch by batch, resolving each batch's outcomes
against the *fitted* environment models and feeding them back to the
policy for its own online learning update.

Time granularity is hourly, not daily: the pacing controller's
`elapsed_fraction` is driven by `sim_hour` (bidding.bootstrap_flight_
contexts' finer clock, `sim_day*24 + the row's own real hour-of-day`),
not the coarser `sim_day` an earlier version used. Daily granularity meant
lambda_delivery only got ~15-28 update opportunities per flight (one per
day, since many batches can share a day for a high-volume scenario) --
confirmed too coarse for pacing.PACE_CONVEXITY's patient-early/urgent-late
behavior to actually operate at a useful resolution.

Contractual delivery: if the scenario's pre-materialized flight
(scenarios.materialize_scenario, sorted by sim_hour) runs out before the
target is met, this pulls another bootstrap chunk from the same pool
(bidding.bootstrap_flight_contexts, continuing the sim_hour count) rather
than stopping -- matches the "campaign doesn't get turned off until the
contractual target is delivered" rule the user set. A hard safety cap
(`max_overrun_multiple` x the nominal flight length) catches a policy
that's pathologically failing to ever bid, reported as a failure rather
than an infinite loop.
"""

from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import bidding, ctr_model, market_model, pacing
from .bandit import BanditPolicy
from .scenarios import TEST_DAYS, materialize_scenario


def _save_checkpoint(checkpoint_path: Path, state: dict) -> None:
    """Atomic write (temp file + rename) so a crash mid-write can't leave a
    corrupt checkpoint that would silently poison a resume."""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    tmp_path.replace(checkpoint_path)


def _load_checkpoint(checkpoint_path: Path | None, scenario_id: str) -> dict | None:
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)
    if state["scenario_id"] != scenario_id:
        return None  # stale checkpoint from a different scenario -- ignore, start fresh
    return state


def simulate_flight(
    scenario: dict,
    policy: BanditPolicy,
    df: pd.DataFrame,
    market_booster: lgb.Booster,
    market_category_maps: dict,
    price_booster: lgb.Booster,
    price_smear_factor: float,
    ctr_booster: lgb.Booster,
    ctr_category_maps: dict,
    batch_size: int = 5000,
    outcome_seed: int = 0,
    max_overrun_multiple: float = 3.0,
    track_trajectory: bool = False,
    checkpoint_path: Path | None = None,
    checkpoint_every_batches: int = 20,
) -> dict:
    """`track_trajectory=True` additionally records a per-batch snapshot
    (cumulative delivered/spend, this batch's own marginal CPM, day,
    lambda_delivery/lambda_ctr) -- off by default to keep the normal
    per-scenario result small, but this is exactly the evidence needed to
    check whether the bandit's spend efficiency actually improves over the
    course of a flight as its online models see more data (the premise
    behind both "cold start needs runway to learn" and Phase 4's
    warm-start exploration-cost comparison).

    `checkpoint_path`, if given, saves full resumable state (the policy's
    learned weights, loop counters, trajectory-so-far) every
    `checkpoint_every_batches` batches -- the largest scenarios take
    20+ minutes, so a mid-scenario crash without this would lose all of
    it, not just the current scenario's *result*. `policy`'s attributes
    are overwritten in place from a found checkpoint (it's a mutable
    object the caller passed in, so this doesn't need the caller to do
    anything differently to resume -- just call again with the same
    checkpoint_path). The checkpoint is deleted on successful completion;
    a stale one is only reused if it matches this scenario's id.
    """
    target = scenario["target_impressions"]
    nominal_days = scenario["flight_length_days"]
    ctr_floor = scenario["ctr_floor"]
    pool_dates = [dt.date.fromisoformat(d) for d in scenario["pool_dates"]]

    rng = np.random.default_rng(outcome_seed)

    checkpoint = _load_checkpoint(checkpoint_path, scenario["id"])
    if checkpoint is not None:
        ps = checkpoint["policy_state"]
        policy.win_rate_model.mu = ps["win_rate_mu"]
        policy.win_rate_model.q = ps["win_rate_q"]
        policy.ctr_model.mu = ps["ctr_mu"]
        policy.ctr_model.q = ps["ctr_q"]
        policy.price_model.mu = ps["price_mu"]
        policy.price_model.q = ps["price_q"]
        policy.lambda_delivery = ps["lambda_delivery"]
        policy.lambda_ctr = ps["lambda_ctr"]
        delivered = checkpoint["delivered"]
        spend = checkpoint["spend"]
        clicks = checkpoint["clicks"]
        hours_used = checkpoint["hours_used"]
        stream_pos = checkpoint["stream_pos"]
        extension_round = checkpoint["extension_round"]
        stream_hour_offset = checkpoint["stream_hour_offset"]
        bid_level_counts = checkpoint["bid_level_counts"]
        trajectory = checkpoint["trajectory"]
        print(
            f"  resumed {scenario['id']} from checkpoint: delivered={delivered:,}/{target:,}, "
            f"extension_round={extension_round}, stream_pos={stream_pos}",
            flush=True,
        )
    else:
        delivered = 0
        spend = 0.0
        clicks = 0
        hours_used = 0.0
        stream_pos = 0
        extension_round = 0
        stream_hour_offset = 0.0
        bid_level_counts = {}
        trajectory = []

    if extension_round == 0:
        stream = materialize_scenario(df, scenario)
    else:
        stream = bidding.bootstrap_flight_contexts(
            df,
            scenario["advertiser_id"],
            pool_dates,
            nominal_days,
            seed=scenario["seed"] + 1000 * extension_round,
            region_ids=scenario["region_ids"],
            city_ids=scenario["city_ids"],
        )
        stream["sim_hour"] = stream["sim_hour"] + stream_hour_offset
    capped = False
    batch_count = 0
    nominal_hours = nominal_days * 24

    while delivered < target:
        if stream_pos >= len(stream):
            if hours_used >= nominal_hours * max_overrun_multiple:
                capped = True
                break
            extension_round += 1
            stream_hour_offset = hours_used
            extra = bidding.bootstrap_flight_contexts(
                df,
                scenario["advertiser_id"],
                pool_dates,
                nominal_days,
                seed=scenario["seed"] + 1000 * extension_round,
                region_ids=scenario["region_ids"],
                city_ids=scenario["city_ids"],
            )
            if len(extra) == 0:
                capped = True
                break
            extra["sim_hour"] = extra["sim_hour"] + stream_hour_offset
            stream = extra
            stream_pos = 0

        batch_raw = stream.iloc[stream_pos : stream_pos + batch_size]
        stream_pos += len(batch_raw)
        batch_contexts = market_model.prepare_all_bids(batch_raw)
        n_batch = len(batch_contexts)

        chosen_bids = policy.choose_bids(batch_contexts)
        for level, count in zip(*np.unique(chosen_bids[chosen_bids > 0], return_counts=True)):
            bid_level_counts[float(level)] = bid_level_counts.get(float(level), 0) + int(count)

        won = np.zeros(n_batch, dtype=bool)
        clicked = np.zeros(n_batch, dtype=bool)
        price_paid = np.zeros(n_batch)

        bid_mask = chosen_bids > 0
        if bid_mask.any():
            sub = batch_contexts.loc[bid_mask]
            win_probs = market_model.predict_at_prices(market_booster, sub, market_category_maps, chosen_bids[bid_mask])
            win_draw = rng.random(len(win_probs)) < win_probs
            won[np.where(bid_mask)[0][win_draw]] = True

        if won.any():
            won_sub = batch_contexts.loc[won]
            price_paid[won] = market_model.predict_price(
                price_booster, won_sub, market_category_maps, smear_factor=price_smear_factor
            )
            click_probs = ctr_model.predict(ctr_booster, won_sub, ctr_category_maps)
            click_draw = rng.random(len(click_probs)) < click_probs
            clicked[np.where(won)[0][click_draw]] = True

        delivered += int(won.sum())
        # RMB/CPM -> RMB per single impression, same conversion bidding.evaluate_flat_bid uses.
        spend += float(price_paid[won].sum()) / 1000.0
        clicks += int(clicked.sum())

        policy.observe(batch_contexts, chosen_bids, won.astype(float), clicked.astype(float), price_paid)

        hours_used = float(batch_raw["sim_hour"].max()) + 1
        elapsed_fraction = hours_used / nominal_hours
        running_ctr = clicks / delivered if delivered > 0 else 0.0
        policy.lambda_delivery = pacing.update_delivery_lambda(policy.lambda_delivery, delivered, target, elapsed_fraction)
        policy.lambda_ctr = pacing.update_ctr_lambda(policy.lambda_ctr, running_ctr, ctr_floor, delivered)

        if track_trajectory:
            batch_won = int(won.sum())
            batch_spend = float(price_paid[won].sum()) / 1000.0
            trajectory.append(
                {
                    "days_used": hours_used / 24.0,
                    "cumulative_delivered": delivered,
                    "cumulative_spend": spend,
                    "batch_won": batch_won,
                    "batch_cpm": (batch_spend / batch_won * 1000.0) if batch_won > 0 else None,
                    "lambda_delivery": policy.lambda_delivery,
                    "lambda_ctr": policy.lambda_ctr,
                }
            )

        batch_count += 1
        if checkpoint_path is not None and batch_count % checkpoint_every_batches == 0:
            _save_checkpoint(
                checkpoint_path,
                {
                    "scenario_id": scenario["id"],
                    "policy_state": {
                        "win_rate_mu": policy.win_rate_model.mu,
                        "win_rate_q": policy.win_rate_model.q,
                        "ctr_mu": policy.ctr_model.mu,
                        "ctr_q": policy.ctr_model.q,
                        "price_mu": policy.price_model.mu,
                        "price_q": policy.price_model.q,
                        "lambda_delivery": policy.lambda_delivery,
                        "lambda_ctr": policy.lambda_ctr,
                    },
                    "delivered": delivered,
                    "spend": spend,
                    "clicks": clicks,
                    "hours_used": hours_used,
                    "stream_pos": stream_pos,
                    "extension_round": extension_round,
                    "stream_hour_offset": stream_hour_offset,
                    "bid_level_counts": bid_level_counts,
                    "trajectory": trajectory,
                },
            )

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()  # scenario finished (or hit the overrun cap) -- no longer needed

    achieved_ctr = clicks / delivered if delivered > 0 else 0.0
    days_used = hours_used / 24.0
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
        "trajectory": trajectory if track_trajectory else None,
    }
