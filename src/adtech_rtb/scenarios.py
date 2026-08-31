"""Random scenario generation shared by the naive baseline and the bandit.

A "scenario" is a fixed evaluation problem: one campaign delivering a
target impression count in a target geo, over a flight of
`flight_length_days`, subject to a CTR floor. Both the naive baseline
(bidding.py) and the Phase 3 bandit are run against the *same* generated
scenarios so their results (spend, delivery, CTR) are directly comparable
-- generation happens once, with a fixed seed, and is persisted to
config/scenarios.yaml rather than re-randomized per run.

The processed test set only spans 2013-06-11/12 (make_dataset.py's
TEST_DAYS) -- 2 real calendar days, too short a flight for a contextual
bandit to get meaningful exploration out of. Flights are instead
synthesized by bootstrap-resampling from that 2-day pool
(bidding.bootstrap_flight_contexts) at each geo's own real daily rate,
scaled up to FLIGHT_LENGTH_DAYS_RANGE. See that function's docstring for
the caveat this trades away (no real day-to-day trend beyond the 2
pooled days) in exchange for giving the bandit's pacing/exploration logic
an actual multi-week problem to work with.

Delivery targets are sampled as a fraction of the *achievable* marginal
win rate (bidding.marginal_win_rate at the top of the supported bid
range), not a fraction of raw auction volume. explore_naive_baseline.py
showed that most geo populations top out around 20-35% win rate within
the historically supported 200-320 RMB/CPM bid range -- sampling a raw
volume fraction (e.g. up to 60%) would generate targets no flat bid
could ever reach. Sampling within a fraction of the ceiling instead
keeps every generated target reachable by construction, for *any*
bidder at least as capable as picking the best single flat bid --
which the bandit, with per-auction context, always is.
"""

from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import bidding, ctr_model, market_model

TEST_DAYS = [dt.date(2013, 6, 11), dt.date(2013, 6, 12)]

# Bootstrapped flight length, in days -- biased long (2-4 weeks) rather
# than short so every scenario gives the bandit enough auction volume to
# actually explore/learn within the flight, not just a token few days.
FLIGHT_LENGTH_DAYS_RANGE = (14, 28)

# Sample the delivery target as this fraction of the achievable win-rate
# ceiling (marginal_win_rate at DEFAULT_BID_BOUNDS[1]) -- keeps every
# scenario reachable by a flat bid, per the module docstring.
TARGET_WIN_RATE_FRACTION = (0.3, 0.85)

# CTR floor as a fraction of the population's natural (unconstrained)
# expected CTR -- a real constraint, but not one already violated by the
# population average.
CTR_FLOOR_FRACTION = (0.5, 0.9)

# Below this many real (pre-bootstrap) auctions in the 2-day pool, the
# daily rate estimate feeding the bootstrap is too noisy to trust; such
# geo candidates are skipped and re-drawn.
MIN_POOL_AUCTIONS = 5_000

N_TOP_REGIONS = 5  # candidate pool for the broad (region-level) geo draw


def _sample_geo(rng: np.random.Generator, df: pd.DataFrame, advertiser_id: int) -> dict | None:
    """Pick a geo: a region, or (50% of the time, if it has enough volume
    on its own) a narrower region+city pair -- variety in population size
    is deliberate, so scenarios span both broad and thin targeting.

    Always assessed against the full 2-day TEST_DAYS pool, independent of
    the scenario's (separately sampled) flight length -- geo quality is
    about how noisy the real daily-rate estimate is, not about how long
    the synthesized flight will be.
    """
    region_volumes = bidding.geo_volume_table(df, advertiser_id, TEST_DAYS, by="region_id")
    region_volumes = region_volumes[region_volumes >= MIN_POOL_AUCTIONS]
    if region_volumes.empty:
        return None

    top_regions = region_volumes.head(N_TOP_REGIONS)
    region_id = int(rng.choice(top_regions.index.to_numpy()))

    if rng.random() < 0.5:
        region_df = bidding.select_eligible_auctions(df, advertiser_id, TEST_DAYS, region_ids=[region_id])
        city_volumes = region_df["city_id"].value_counts()
        city_volumes = city_volumes[city_volumes >= MIN_POOL_AUCTIONS]
        if not city_volumes.empty:
            city_id = int(rng.choice(city_volumes.index.to_numpy()))
            return {"region_ids": [region_id], "city_ids": [city_id]}

    return {"region_ids": [region_id], "city_ids": None}


def materialize_scenario(df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    """Regenerate a scenario's exact bootstrapped auction stream (raw
    columns, not yet feature-engineered) from its persisted definition.

    Both the naive baseline and the bandit must call this (not their own
    ad hoc sampling) so they're evaluated against identical synthetic
    flights for a given scenario id.
    """
    return bidding.bootstrap_flight_contexts(
        df,
        scenario["advertiser_id"],
        TEST_DAYS,
        scenario["flight_length_days"],
        seed=scenario["seed"],
        region_ids=scenario["region_ids"],
        city_ids=scenario["city_ids"],
    )


def _attempt_scenario(
    rng: np.random.Generator,
    df: pd.DataFrame,
    market_booster: lgb.Booster,
    market_category_maps: dict,
    ctr_booster: lgb.Booster,
    ctr_category_maps: dict,
    advertiser_id: int,
    scenario_id: str,
) -> dict | None:
    """One self-contained attempt to build `scenario_id`; None means retry
    with a fresh rng (thin geo draw, or a degenerate win-rate ceiling)."""
    geo = _sample_geo(rng, df, advertiser_id)
    if geo is None:
        return None

    # Stats (win-rate ceiling, mean CTR, daily rate) come from the real
    # 2-day pool only -- bootstrapping is i.i.d. resampling from that same
    # population, so it can't shift these statistics, only the sample
    # size. No need to materialize a multi-week bootstrap sample (millions
    # of rows, expensive to run every model on) just to compute numbers
    # already determined by the much smaller real pool; the actual
    # bootstrapped auction stream is generated later, lazily, by
    # materialize_scenario() when the baseline/bandit actually simulates
    # this scenario.
    pool_df = bidding.select_eligible_auctions(
        df, advertiser_id, TEST_DAYS, region_ids=geo["region_ids"], city_ids=geo["city_ids"]
    )
    if len(pool_df) < MIN_POOL_AUCTIONS:
        return None

    pool_contexts = market_model.prepare_all_bids(pool_df)
    rate_ceiling = bidding.marginal_win_rate(
        market_booster, pool_contexts, market_category_maps, bidding.DEFAULT_BID_BOUNDS[1]
    )
    if rate_ceiling <= 0:
        return None

    flight_length_days = int(rng.integers(FLIGHT_LENGTH_DAYS_RANGE[0], FLIGHT_LENGTH_DAYS_RANGE[1] + 1))
    bootstrap_seed = int(rng.integers(0, 2**31 - 1))
    daily_rate = len(pool_contexts) / len(TEST_DAYS)
    n_eligible_auctions = round(daily_rate * flight_length_days)

    # Guaranteed reachable by construction: target_rate is a strict
    # fraction (<1) of rate_ceiling, the same ceiling that bounds what any
    # flat bid can achieve over this population.
    target_rate = rng.uniform(*TARGET_WIN_RATE_FRACTION) * rate_ceiling
    target_impressions = max(1, round(target_rate * n_eligible_auctions))

    mean_ctr = float(np.mean(ctr_model.predict(ctr_booster, pool_contexts, ctr_category_maps)))
    ctr_floor = rng.uniform(*CTR_FLOOR_FRACTION) * mean_ctr

    return {
        "id": scenario_id,
        "advertiser_id": advertiser_id,
        "region_ids": geo["region_ids"],
        "city_ids": geo["city_ids"],
        "flight_length_days": flight_length_days,
        "pool_dates": [d.isoformat() for d in TEST_DAYS],
        "seed": bootstrap_seed,
        "n_eligible_auctions": n_eligible_auctions,
        "target_impressions": target_impressions,
        "ctr_floor": round(ctr_floor, 6),
        "population_mean_ctr": round(mean_ctr, 6),
        "naive_baseline_reachable": bool(target_rate < rate_ceiling),
    }


def iter_scenarios(
    df: pd.DataFrame,
    market_booster: lgb.Booster,
    market_category_maps: dict,
    ctr_booster: lgb.Booster,
    ctr_category_maps: dict,
    advertiser_ids: list[int],
    n_per_advertiser: int = 3,
    seed: int = 42,
    max_attempts_per_scenario: int = 10,
    skip_ids: frozenset[str] = frozenset(),
):
    """Yield `n_per_advertiser` random, reachable scenarios per advertiser,
    one at a time, as soon as each is built.

    Each scenario's random draws come from a seed derived only from
    `(seed, advertiser_id, its own index, attempt number)`
    (np.random.SeedSequence), not a single mutating rng shared across the
    whole run -- so scenario "1458-2" is fully independent of whether
    "1458-1" took 1 attempt or 9. That's what makes `skip_ids` a real
    resume, not just a re-run that happens to overwrite the same ids: a
    scenario already in `skip_ids` costs one dict lookup, nothing else,
    and every scenario not skipped reproduces bit-for-bit regardless of
    what ran before it in this or a prior process.
    """
    for advertiser_id in advertiser_ids:
        generated = 0
        for index in range(1, n_per_advertiser + 1):
            scenario_id = f"{advertiser_id}-{index}"
            if scenario_id in skip_ids:
                generated += 1
                continue

            result = None
            for attempt in range(1, max_attempts_per_scenario + 1):
                rng = np.random.default_rng(np.random.SeedSequence([seed, advertiser_id, index, attempt]))
                result = _attempt_scenario(
                    rng, df, market_booster, market_category_maps, ctr_booster, ctr_category_maps,
                    advertiser_id, scenario_id,
                )
                if result is not None:
                    break

            if result is None:
                print(
                    f"Warning: could not build scenario {scenario_id} in "
                    f"{max_attempts_per_scenario} attempts", flush=True,
                )
                continue

            generated += 1
            yield result

        if generated < n_per_advertiser:
            print(
                f"Warning: only generated {generated}/{n_per_advertiser} scenarios "
                f"for advertiser {advertiser_id}", flush=True,
            )
