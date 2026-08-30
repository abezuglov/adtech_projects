"""LightGBM click-through-rate model: P(click | impression shown).

Trained only on won bids -- CTR is only defined for impressions actually
served. A losing bid was never shown, so "clicked" is trivially False for
every one of them; including them would bias the model toward whatever
distinguishes won from lost bids rather than what drives clicks.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .features import ALL_CATEGORICAL_COLUMNS, ALL_FEATURE_COLUMNS, apply_category_maps, build_features

# Columns actually needed from data/processed/*.parquet -- deliberately
# excludes ip/url/anon_url_id/ad_slot_id/ipinyou_id/paying_price/etc.
# Loading those unused high-cardinality string columns for the full
# ~43M-row frame is what triggered an Arrow "offset overflow" (>2GB of
# string data in one contiguous buffer) during the won/lost boolean filter.
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
    "won",
    "clicked",
]

# No is_unbalance/scale_pos_weight: those optimize ranking at the cost of
# calibration (predicted probabilities stop meaning P(click)), which broke
# badly here -- log loss of ~21 vs. ~0.006 for a trivial constant-rate
# model, and boosting stalled after one tree. The bandit's constraint
# gating needs an actual probability estimate, not just a ranking score.
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
}


def prepare_won_subset(df: pd.DataFrame) -> pd.DataFrame:
    won = df[df["won"]].copy()
    return build_features(won)


def train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> lgb.Booster:
    train_enc = apply_category_maps(train_df, category_maps)
    valid_enc = apply_category_maps(valid_df, category_maps)
    params = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(
        train_enc[ALL_FEATURE_COLUMNS],
        label=train_df["clicked"].astype(int),
        categorical_feature=ALL_CATEGORICAL_COLUMNS,
    )
    valid_set = lgb.Dataset(
        valid_enc[ALL_FEATURE_COLUMNS],
        label=valid_df["clicked"].astype(int),
        reference=train_set,
        categorical_feature=ALL_CATEGORICAL_COLUMNS,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)],
    )


def predict(booster: lgb.Booster, df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> np.ndarray:
    enc = apply_category_maps(df, category_maps)
    return booster.predict(enc[ALL_FEATURE_COLUMNS], num_iteration=booster.best_iteration)


def evaluate(booster: lgb.Booster, df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> dict:
    y_true = df["clicked"].astype(int).to_numpy()
    y_prob = predict(booster, df, category_maps)
    return {
        "n": int(len(df)),
        "n_clicks": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
