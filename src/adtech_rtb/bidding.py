"""Naive baseline bidder: a single fixed bid chosen to hit a delivery target.

Phase 2's "naive" baseline (see the plan doc) is deliberately a flat bid --
no per-auction context, no exploration, no budget-aware pacing across the
window. The only intelligence it has is *choosing* that flat bid correctly:
reusing the already-trained win-rate model (market_model.py) to find the bid
whose marginal (context-averaged) win rate clears a target impression count
over one scenario's eligible-auction population (a campaign's own bid
requests in a target geo, over the scenario's flight -- see
bootstrap_flight_contexts for how flights longer than the 2 real held-out
test days are synthesized).

Deliberately does NOT build this curve from the empirical paying_price
distribution. paying_price is only observed for WON auctions
(market_model.py's censoring note -- we only learn our historical bid
was insufficient on a loss, never by how much), so an empirical quantile
of it is a biased view of the market: it silently excludes every auction
we'd have needed a higher bid to win. Querying the win-rate classifier
instead is unbiased, because it was trained on the `won` label, which is
complete for every bid, won or lost.

Eligible auctions are filtered by advertiser_id too, not just geo/date:
data/processed/*.parquet only retains the 5 campaigns configured in
config.yaml (see make_dataset.py) -- there's no pooled cross-advertiser
"whole market" in this dataset, so "eligible auctions" can only mean a
given campaign's own historical bid-request volume in a geo/day.
"""

from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import ctr_model, market_model

# market_model.py's own bid-price sanity check only queried 200-320
# RMB/CPM, the historically observed range for this dataset's ~227-300
# fixed-price bidding strategies. Default search bounds stay inside that
# support; solving outside it is extrapolation with no ground truth
# behind it (see market_model.py's module docstring).
DEFAULT_BID_BOUNDS = (200.0, 320.0)


def select_eligible_auctions(
    df: pd.DataFrame,
    advertiser_id: int,
    dates: list[dt.date],
    region_ids: list[int] | None = None,
    city_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Filter to one scenario's campaign + flight (one or more days) + geo.

    Geo granularity (region-only, city-only, or both) is left to the
    caller -- which regions/cities to use per campaign is still being
    picked experimentally against how much volume each has, not
    hardcoded here as a fixed policy. `dates` takes a list so a flight
    can span more than one day (test set only has 2013-06-11/12, so a
    flight is 1 or 2 days in practice).
    """
    mask = (df["advertiser_id"] == advertiser_id) & (df["timestamp"].dt.date.isin(dates))
    if region_ids is not None:
        mask &= df["region_id"].isin(region_ids)
    if city_ids is not None:
        mask &= df["city_id"].isin(city_ids)
    return df[mask]


def bootstrap_flight_contexts(
    df: pd.DataFrame,
    advertiser_id: int,
    pool_dates: list[dt.date],
    flight_length_days: int,
    seed: int,
    region_ids: list[int] | None = None,
    city_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Bootstrap-resample a `flight_length_days`-day auction stream from the
    real `pool_dates` population.

    The processed dataset only has 2 held-out test days (make_dataset.py),
    which is too short a flight for a contextual bandit to get meaningful
    exploration -- so campaigns longer than that are synthesized by
    resampling real rows with replacement, at the pool's own average daily
    rate, rather than being capped at 2 literal calendar days. Each
    resampled row is tagged with a synthetic 0-indexed `sim_day` (its
    original hour-of-day feature is left untouched) so callers can replay
    the flight in chronological order.

    CAVEAT, deliberately not fixed: the pool is only 2 real calendar days
    (2013-06-11 Tue, 06-12 Wed), so a bootstrapped multi-week flight
    reproduces their hour-of-day pattern repeatedly but has no real
    across-day trend (no weekend, no ramp-up/spike) beyond what those 2
    days show. Captures within-day structure, not real day-to-day drift.

    Deterministic given `seed` -- the naive baseline and the bandit must
    call this with the same scenario's stored seed to evaluate against
    identical synthetic auction streams.
    """
    pool = select_eligible_auctions(df, advertiser_id, pool_dates, region_ids, city_ids)
    if len(pool) == 0:
        return pool

    daily_rate = len(pool) / len(pool_dates)
    n_total = round(daily_rate * flight_length_days)

    rng = np.random.default_rng(seed)
    row_positions = rng.integers(0, len(pool), size=n_total)
    sim_days = rng.integers(0, flight_length_days, size=n_total)

    sampled = pool.iloc[row_positions].copy()
    sampled["sim_day"] = sim_days
    return sampled.sort_values("sim_day", kind="stable").reset_index(drop=True)


def geo_volume_table(
    df: pd.DataFrame,
    advertiser_id: int,
    dates: list[dt.date],
    by: str = "region_id",
) -> pd.Series:
    """Auction-request volume per geo value over `dates`, for picking scenario geos.

    `by` is "region_id" or "city_id". Meant for the experimentation the
    scenario definitions still need: sort descending and pick geos with
    enough volume that the marginal win-rate curve isn't noisy.
    """
    mask = (df["advertiser_id"] == advertiser_id) & (df["timestamp"].dt.date.isin(dates))
    return df.loc[mask, by].value_counts().sort_values(ascending=False)


def marginal_win_rate(
    booster: lgb.Booster,
    contexts: pd.DataFrame,
    category_maps: dict,
    bid_price: float,
) -> float:
    """Mean P(win | bid_price, context) over `contexts`.

    `contexts` must already be feature-engineered
    (market_model.prepare_all_bids) -- this is the win rate a single
    constant bid of `bid_price` would achieve across this auction
    population.
    """
    return float(np.mean(market_model.predict_at_price(booster, contexts, category_maps, bid_price)))


def solve_delivery_bid(
    booster: lgb.Booster,
    contexts: pd.DataFrame,
    category_maps: dict,
    target_impressions: int,
    bid_bounds: tuple[float, float] = DEFAULT_BID_BOUNDS,
    tol: float = 1e-4,
    max_iter: int = 40,
) -> dict:
    """Bisection search for the fixed bid expected to deliver `target_impressions`
    over `contexts` (one scenario's eligible-auction population).

    Relies on the win-rate model's monotone_constraints on bidding_price
    (see market_model.py) to guarantee marginal_win_rate is non-decreasing
    in bid -- that monotonicity is what makes bisection valid here, and it
    was independently verified in scripts/train_market_model.py's bid-price
    sanity check.
    """
    n = len(contexts)
    if n == 0:
        raise ValueError("contexts is empty -- no eligible auctions for this scenario")
    target_rate = target_impressions / n

    lo, hi = bid_bounds
    rate_lo = marginal_win_rate(booster, contexts, category_maps, lo)
    rate_hi = marginal_win_rate(booster, contexts, category_maps, hi)

    if target_rate <= rate_lo:
        bid = lo
    elif target_rate >= rate_hi:
        bid = hi
    else:
        bid = (lo + hi) / 2
        for _ in range(max_iter):
            bid = (lo + hi) / 2
            rate = marginal_win_rate(booster, contexts, category_maps, bid)
            if abs(rate - target_rate) < tol:
                break
            if rate < target_rate:
                lo = bid
            else:
                hi = bid

    achieved_rate = marginal_win_rate(booster, contexts, category_maps, bid)
    return {
        "bid": bid,
        "n_eligible": n,
        "target_impressions": target_impressions,
        "target_win_rate": target_rate,
        "achieved_win_rate": achieved_rate,
        "expected_impressions": achieved_rate * n,
        "target_reachable_in_bounds": target_rate < rate_hi,
    }


def evaluate_flat_bid(
    market_booster: lgb.Booster,
    market_category_maps: dict,
    price_booster: lgb.Booster,
    price_smear_factor: float,
    ctr_booster: lgb.Booster,
    ctr_category_maps: dict,
    pool_contexts: pd.DataFrame,
    bid: float,
    n_flight: int,
) -> dict:
    """Expected delivery/spend/CTR a constant `bid` produces over a flight
    of `n_flight` auctions, estimated from `pool_contexts` (the real,
    pre-bootstrap population feeding that flight).

    Per-auction expectations (win probability x price, win probability x
    CTR) are population averages over `pool_contexts`, then scaled by
    `n_flight` -- valid because a bootstrapped flight is a resample of
    this same population (bootstrap_flight_contexts), so its
    population-level expectations equal the pool's; only the sample size
    differs. A flat bid's outcome doesn't depend on auction order or any
    evolving state, so there's no need to materialize the (possibly
    multi-million-row) bootstrapped flight itself just to evaluate it --
    unlike the sequential bandit, which will actually need the
    materialized per-auction stream (see scenarios.materialize_scenario).

    `bidding_price`/`paying_price` are both RMB/CPM in this dataset (see
    data/README.md's schema table) -- cost per 1000 impressions, not per
    impression. price_preds is on that same RMB/CPM scale, so the actual
    cost of a single won impression is price_preds/1000; skipping that
    conversion would overstate spend (and CPM) by 1000x.
    """
    win_probs = market_model.predict_at_price(market_booster, pool_contexts, market_category_maps, bid)
    price_preds = market_model.predict_price(
        price_booster, pool_contexts, market_category_maps, smear_factor=price_smear_factor
    )
    ctr_preds = ctr_model.predict(ctr_booster, pool_contexts, ctr_category_maps)

    expected_impressions = float(win_probs.mean()) * n_flight
    expected_spend = float(np.mean(win_probs * price_preds)) / 1000.0 * n_flight
    expected_clicks = float(np.mean(win_probs * ctr_preds)) * n_flight
    achieved_ctr = expected_clicks / expected_impressions if expected_impressions > 0 else 0.0

    return {
        "bid": bid,
        "expected_impressions": expected_impressions,
        "expected_spend": expected_spend,
        "expected_clicks": expected_clicks,
        "achieved_ctr": achieved_ctr,
    }
