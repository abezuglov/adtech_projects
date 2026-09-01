"""Phase 4: warm-start the bandit's online learners from the *other* 4
campaigns' real historical data, applied to 3358 (config.yaml's
`warm_start_holdout` -- the sparsest campaign, held out for exactly this).

The prior-fitting step is not a simulation: it replays real historical
(context, bidding_price, won, clicked, paying_price) tuples through
BanditPolicy.observe() exactly as simulator.py does with *simulated*
outcomes -- the online learners don't know or care whether an observation
came from a real log or a simulated auction, so this reuses that method
directly rather than duplicating its update logic. This is also why it's
principled and not "cheating": the fitted LightGBM environment models are
never touched here, this is purely historical (bid, outcome) replay, the
same kind of evidence a real bidder would have logged from running other
campaigns before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bandit import BanditPolicy
from .features import ALL_CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .market_model import MARKET_CATEGORICAL_COLUMNS, prepare_all_bids

# Cap on rows used to fit the prior -- the other 4 campaigns' full train
# split is ~41M rows; N_HASH_BUCKETS=16384 means even 500K rows gives a
# well-calibrated prior (~30 observations/dimension on average) without
# the multi-minute read+fit cost of the full history. Also a legitimate
# real-world framing: a production warm-start system typically bounds
# itself to a recent window of history, not literally all of it forever.
SAMPLE_SIZE = 500_000
CHUNK_SIZE = 20_000


def fit_prior(train_df: pd.DataFrame, exclude_advertiser_id: int, seed: int = 0) -> BanditPolicy:
    """Returns a BanditPolicy whose three online learners have been
    pretrained on real (bid, outcome) history from every advertiser in
    `train_df` except `exclude_advertiser_id`. `ctr_floor`/`lambda_*` on
    the returned policy are placeholders (fitting never calls choose_bids,
    only observe) -- callers should copy just the learned mu/q into a
    fresh, scenario-specific policy rather than using this one directly.
    """
    other_df = train_df[train_df["advertiser_id"] != exclude_advertiser_id]
    rng = np.random.default_rng(seed)
    if len(other_df) > SAMPLE_SIZE:
        idx = np.sort(rng.choice(len(other_df), size=SAMPLE_SIZE, replace=False))
        other_df = other_df.iloc[idx]

    contexts = prepare_all_bids(other_df)
    policy = BanditPolicy(
        ctr_floor=0.0,
        market_categorical_columns=MARKET_CATEGORICAL_COLUMNS,
        ctr_categorical_columns=ALL_CATEGORICAL_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
        seed=seed,
    )

    n = len(contexts)
    for start in range(0, n, CHUNK_SIZE):
        chunk = contexts.iloc[start : start + CHUNK_SIZE]
        chosen_bids = chunk["bidding_price"].to_numpy(dtype=float)
        won = chunk["won"].to_numpy(dtype=float)
        clicked = chunk["clicked"].to_numpy(dtype=float)
        price_paid = chunk["paying_price"].fillna(0).to_numpy(dtype=float)
        policy.observe(chunk, chosen_bids, won, clicked, price_paid)

    return policy


def apply_prior(prior_policy: BanditPolicy, ctr_floor: float, seed: int) -> BanditPolicy:
    """A fresh, scenario-ready BanditPolicy carrying `prior_policy`'s
    learned weights -- same pattern simulator.py's checkpoint-resume uses
    to restore state into a policy object, just from a fitted prior
    instead of a mid-flight snapshot.
    """
    policy = BanditPolicy(
        ctr_floor=ctr_floor,
        market_categorical_columns=MARKET_CATEGORICAL_COLUMNS,
        ctr_categorical_columns=ALL_CATEGORICAL_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
        seed=seed,
    )
    policy.win_rate_model.mu = prior_policy.win_rate_model.mu.copy()
    policy.win_rate_model.q = prior_policy.win_rate_model.q.copy()
    policy.ctr_model.mu = prior_policy.ctr_model.mu.copy()
    policy.ctr_model.q = prior_policy.ctr_model.q.copy()
    policy.price_model.mu = prior_policy.price_model.mu.copy()
    policy.price_model.q = prior_policy.price_model.q.copy()
    return policy
