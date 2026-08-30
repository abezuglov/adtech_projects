"""LightGBM market models: P(win | bidding_price, context) and E[paying_price | won, context].

Two separate models, because they have different valid training
populations and a naive combination would double-count the censoring
problem:

- **Win-rate**: a direct binary classifier (the `won` label) on ALL bids,
  not a paying_price regression. Paying price itself is only observed for
  won bids (right-censored for losses: we only know our bid was
  insufficient, not by how much), so regressing it directly on all bids
  would bias the model toward the truncated, winner's-curse view of the
  market. "Did we win at our submitted bidding_price" has no such
  problem: every bid, won or lost, carries a complete and correct label
  for that question.

- **Paying price**: under GSP, what you pay if you win is set by the
  competing bids, not your own -- so it's a regression trained on the WON
  subset only (mirroring ctr_model.py's own-subset training), and
  deliberately does NOT take bidding_price as a feature (your own bid
  cannot cause the market-clearing price). Needed because a naive
  "assume you pay your own bid" placeholder would silently turn this
  project's spend metric into a first-price approximation, which gives
  the wrong bidding incentives entirely for a strategy meant to be
  evaluated under GSP semantics -- shading bids down as if you paid what
  you bid is not the same optimization problem as GSP truthful-ish
  bidding. See PRICE_FEATURE_COLUMNS / train_price_model below.
  Caveat: within the won subset, observed paying_price for a given
  context is inherently less than whatever that context's historical
  bidding_price was (that's the win condition) -- so, like the win-rate
  model, this has the least support (most truncation) for contexts whose
  historical bidder used the lowest fixed price level (~227).

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

advertiser_id and creative_id are excluded from both market models (unlike
ctr_model.py, which legitimately needs them -- click-through rate really
does depend on the creative and brand shown). Auction dynamics -- how much
competition exists for a given impression, and what it clears at -- are a
property of the opportunity (page/domain, exchange, slot, floor price,
time, region) and the pool of competing bidders, not of which of our own
five campaigns happened to be the one bidding. Leaving advertiser_id in
would be a second channel for the same campaign-identity confound the
monotonic constraint above addresses (the model could shortcut through
"which campaign" instead of learning genuine context-driven dynamics), and
creative_id is chosen by the advertiser rather than being a property of
the auction at all. It also matters directly for Phase 4 (warm-start): a
market model keyed on advertiser_id can't meaningfully generalize its
win-rate/price estimates to a held-out campaign it never trained on,
which is exactly the scenario warm-start needs to work in.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .features import ALL_CATEGORICAL_COLUMNS, ALL_FEATURE_COLUMNS, apply_category_maps, build_features

# Auction-context features only -- see module docstring for why
# advertiser_id/creative_id (campaign identity, not auction context) are
# excluded here even though ctr_model.py legitimately uses them.
_CAMPAIGN_IDENTITY_COLUMNS = {"advertiser_id", "creative_id"}
MARKET_CATEGORICAL_COLUMNS = [c for c in ALL_CATEGORICAL_COLUMNS if c not in _CAMPAIGN_IDENTITY_COLUMNS]
MARKET_CONTEXT_COLUMNS = [c for c in ALL_FEATURE_COLUMNS if c not in _CAMPAIGN_IDENTITY_COLUMNS]

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
    "paying_price",
    "won",
]

MARKET_FEATURE_COLUMNS = MARKET_CONTEXT_COLUMNS + ["bidding_price"]

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
        categorical_feature=MARKET_CATEGORICAL_COLUMNS,
    )
    valid_set = lgb.Dataset(
        valid_enc[MARKET_FEATURE_COLUMNS],
        label=valid_df["won"].astype(int),
        reference=train_set,
        categorical_feature=MARKET_CATEGORICAL_COLUMNS,
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


# --- Paying-price regression (won subset only; see module docstring) -------

# Deliberately excludes bidding_price: under GSP, the price you pay if you
# win is set by competing bids, not your own, so it shouldn't be a feature
# of "what will the market charge me." Also excludes advertiser_id/
# creative_id -- see module docstring.
PRICE_FEATURE_COLUMNS = MARKET_CONTEXT_COLUMNS

PRICE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
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


def train_price_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 25,
) -> lgb.Booster:
    """Train E[log1p(paying_price) | won, context] on the won subset of `train_df`/`valid_df`.

    Callers pass the full (won + lost) frames; this filters to `won` itself
    so the caller doesn't have to duplicate that logic at every call site.
    """
    train_won = train_df[train_df["won"]]
    valid_won = valid_df[valid_df["won"]]
    train_enc = apply_category_maps(train_won, category_maps)
    valid_enc = apply_category_maps(valid_won, category_maps)
    y_train = np.log1p(train_won["paying_price"].to_numpy(dtype="float64"))
    y_valid = np.log1p(valid_won["paying_price"].to_numpy(dtype="float64"))
    params = {**PRICE_PARAMS, **(params or {})}
    train_set = lgb.Dataset(
        train_enc[PRICE_FEATURE_COLUMNS],
        label=y_train,
        categorical_feature=MARKET_CATEGORICAL_COLUMNS,
    )
    valid_set = lgb.Dataset(
        valid_enc[PRICE_FEATURE_COLUMNS],
        label=y_valid,
        reference=train_set,
        categorical_feature=MARKET_CATEGORICAL_COLUMNS,
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


def _raw_log_predict(booster: lgb.Booster, df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> np.ndarray:
    enc = apply_category_maps(df, category_maps)
    return booster.predict(enc[PRICE_FEATURE_COLUMNS], num_iteration=booster.best_iteration)


def compute_smear_factor(booster: lgb.Booster, df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> float:
    """Retransformation-bias correction for training on log1p(price), evaluated on held-out data.

    E[expm1(log1p(Y))] = E[Y] exactly, but the model only gives us
    expm1(E[log1p(Y) | x]) -- and exp is concave, so by Jensen's inequality
    that systematically *underestimates* E[Y | x] whenever there's residual
    variance (the higher the variance, the bigger the gap). This shows up
    empirically as the price model reading low in market_price_calibration.png,
    worst at the cheap end where relative price variance is highest.

    Textbook Duan smearing computes this as a *mean of ratios*,
    mean(Z_true / Z_pred) -- which is provably consistent only when the
    retransformation error is homoscedastic (independent of x). Tried that
    here first and it overshot badly: it overcorrected the highest-volume
    price regime (worst-affected bucket went from -6% to +27% biased) and
    made MAE noticeably worse, because the plain average is dragged around
    by the small-Z_pred/high-relative-variance rows that produce the
    largest ratios -- exactly the heteroscedasticity this dataset has,
    since relative price variance differs by price level (see the
    module's cheap-end caveat above).

    Using a *ratio of sums*, sum(Z_true) / sum(Z_pred), instead: this
    guarantees the corrected predictions match the true aggregate exactly
    on the calibration population, rather than being skewed by the
    noisiest individual ratios. That's also the quantity that actually
    matters downstream -- the bandit's budget simulation cares about total
    predicted spend, not a pointwise-consistent E[Y|x] estimator. Must be
    computed on a set disjoint from training (here, valid_df) or it would
    just refit the training bias back in.
    """
    won = df[df["won"]]
    log_pred = _raw_log_predict(booster, won, category_maps)
    log_true = np.log1p(won["paying_price"].to_numpy(dtype="float64"))
    return float(np.sum(np.exp(log_true)) / np.sum(np.exp(log_pred)))


def predict_price(
    booster: lgb.Booster,
    df: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    smear_factor: float = 1.0,
) -> np.ndarray:
    """E[paying_price | context], in the original RMB/CPM scale (not log).

    `smear_factor` (see compute_smear_factor) corrects the systematic
    underestimation from the log1p<->expm1 round trip; defaults to 1.0
    (uncorrected) so this stays a pure inference function callers can also
    use before a smear factor has been computed.
    """
    log_price = _raw_log_predict(booster, df, category_maps)
    return np.expm1(log_price) * smear_factor + (smear_factor - 1.0)


def evaluate_price(
    booster: lgb.Booster,
    df: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    smear_factor: float = 1.0,
) -> dict:
    won = df[df["won"]]
    y_true = won["paying_price"].to_numpy(dtype="float64")
    y_pred = predict_price(booster, won, category_maps, smear_factor=smear_factor)
    error = y_pred - y_true
    return {
        "n": int(len(won)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_true": float(y_true.mean()),
        "mean_pred": float(y_pred.mean()),
    }
