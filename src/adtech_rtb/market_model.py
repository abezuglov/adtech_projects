"""LightGBM win-rate model: P(win | bidding_price, context).

Trained as a direct binary classifier (the `won` label) on ALL bids, not
a paying_price regression -- that matters. Paying price itself is only
observed for won bids (right-censored for losses: we only know our bid
was insufficient, not by how much), so regressing it directly would bias
the model toward the truncated, winner's-curse view of the market. "Did
we win at our submitted bidding_price" has no such problem: every bid,
won or lost, carries a complete and correct label for that question.

Caveat that matters for the bandit: submitted bidding_price in this
dataset is narrow (~227-300 RMB/CPM, the "fixed relatively high-price"
strategy -- see data/README.md), so the model has real support only in
that range. Querying it at candidate prices far outside [227, 300] is
extrapolation with no ground truth backing it.

Monotonic constraint on bidding_price: each campaign in this dataset
bids at close to one fixed price level, so bidding_price is nearly a
proxy for *which campaign placed the bid* rather than a variable that's
experimentally varied within a fixed context. An unconstrained model
picked this up and predicted *lower* win rates at higher bidding_price
(confounded by which advertiser tends to bid high vs. low, not a causal
price effect) -- it failed the plan's own bid-price monotonicity check.
Under GSP (win iff bid > paying_price), P(win) is mechanically
non-decreasing in your own bid, holding the rest of the market fixed.
That's a fact about the auction, not something to hope the model infers
from a dataset with almost no within-context price variation to learn
it from -- so it's enforced directly via LightGBM's monotone_constraints
on the bidding_price feature.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .features import ALL_CATEGORICAL_COLUMNS, ALL_FEATURE_COLUMNS, apply_category_maps, build_features

RAW_COLUMNS = [
    "bid_id",
    "timestamp",
    "user_agent",
    "region_id",
    "city_id",
    "ad_exchange",
    "domain",
    "ad_slot_width",
    "ad_slot_height",
    "ad_slot_visibility",
    "ad_slot_format",
    "ad_slot_floor_price",
    "creative_id",
    "advertiser_id",
    "bidding_price",
    "won",
]

MARKET_FEATURE_COLUMNS = ALL_FEATURE_COLUMNS + ["bidding_price"]

# +1 (non-decreasing) on bidding_price only, per the module docstring;
# 0 (unconstrained) on every other feature.
_MONOTONE_CONSTRAINTS = [1 if col == "bidding_price" else 0 for col in MARKET_FEATURE_COLUMNS]

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "verbose": -1,
    "monotone_constraints": _MONOTONE_CONSTRAINTS,
    "monotone_constraints_method": "advanced",
}


def prepare_all_bids(df: pd.DataFrame) -> pd.DataFrame:
    out = build_features(df)
    out["bidding_price"] = out["bidding_price"].to_numpy(dtype="float64")
    return out


def train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 25,
) -> lgb.Booster:
    """Train, checkpointing the booster to `checkpoint_path` periodically.

    This dataset (~38M rows, several high-cardinality categoricals) is
    large enough that a training run has genuinely OOM-killed without any
    Python traceback -- LightGBM's C++ core doesn't raise a catchable
    exception for that, the process just dies. If `checkpoint_path` already
    exists, resumes boosting from it via `init_model` rather than starting
    over from tree 0.
    """
    train_enc = apply_category_maps(train_df, category_maps, extra_numeric_columns=["bidding_price"])
    valid_enc = apply_category_maps(valid_df, category_maps, extra_numeric_columns=["bidding_price"])
    params = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(
        train_enc[MARKET_FEATURE_COLUMNS],
        label=train_df["won"].astype(int),
        categorical_feature=ALL_CATEGORICAL_COLUMNS,
    )
    valid_set = lgb.Dataset(
        valid_enc[MARKET_FEATURE_COLUMNS],
        label=valid_df["won"].astype(int),
        reference=train_set,
        categorical_feature=ALL_CATEGORICAL_COLUMNS,
    )

    init_model = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        init_model = str(checkpoint_path)

    callbacks = [lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)]
    if checkpoint_path is not None:

        def _save_checkpoint(env: lgb.callback.CallbackEnv) -> None:
            if env.iteration % checkpoint_every == 0:
                env.model.save_model(str(checkpoint_path))

        callbacks.append(_save_checkpoint)

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        init_model=init_model,
        callbacks=callbacks,
    )


def predict(booster: lgb.Booster, df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> np.ndarray:
    enc = apply_category_maps(df, category_maps, extra_numeric_columns=["bidding_price"])
    return booster.predict(enc[MARKET_FEATURE_COLUMNS], num_iteration=booster.best_iteration)


def predict_at_price(
    booster: lgb.Booster,
    df: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    bid_price: float,
) -> np.ndarray:
    """P(win) if `df`'s contexts had instead bid `bid_price`.

    Only trustworthy within the historically observed range (~227-300
    RMB/CPM) -- see module docstring.
    """
    overridden = df.copy()
    overridden["bidding_price"] = bid_price
    return predict(booster, overridden, category_maps)


def evaluate(booster: lgb.Booster, df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> dict:
    y_true = df["won"].astype(int).to_numpy()
    y_prob = predict(booster, df, category_maps)
    return {
        "n": int(len(df)),
        "n_won": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
