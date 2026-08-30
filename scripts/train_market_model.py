"""Train the win-rate/market model (Phase 1 market simulator, price side).

Checkpoints the feature-engineered full bid population (won + lost) to
data/interim/ so a crash doesn't require re-reading the multi-GB
train/test parquet files, and saves the trained model + metrics.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.features import fit_category_maps  # noqa: E402
from adtech_rtb.market_model import (  # noqa: E402
    MARKET_FEATURE_COLUMNS,
    RAW_COLUMNS,
    evaluate,
    predict,
    predict_at_price,
    prepare_all_bids,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

CACHE_COLUMNS = MARKET_FEATURE_COLUMNS + ["bid_id", "timestamp", "won"]


def _load_or_build_features(name: str, source_path: Path) -> pd.DataFrame:
    cache_path = INTERIM_DIR / f"market_{name}_features.parquet"
    if cache_path.exists():
        print(f"  {name}: loading cached features from {cache_path.name}")
        return pd.read_parquet(cache_path, dtype_backend="pyarrow")
    print(f"  {name}: reading {source_path.name} and building features...")
    df = pd.read_parquet(source_path, columns=RAW_COLUMNS, dtype_backend="pyarrow")
    # See models/ctr_* lessons: user_agent's raw strings alone exceed the
    # 2GB limit for the compact "string" arrow type once concatenated
    # across row groups (triggered here by the fit/valid boolean split).
    df["user_agent"] = df["user_agent"].astype(pd.ArrowDtype(pa.large_string()))
    feats = prepare_all_bids(df)[CACHE_COLUMNS]
    feats.to_parquet(cache_path, index=False)
    print(f"  {name}: {len(feats):,} bids -> cached")
    return feats


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_all = _load_or_build_features("train", PROCESSED_DIR / "train.parquet")
    test_all = _load_or_build_features("test", PROCESSED_DIR / "test.parquet")

    last_day = train_all["timestamp"].dt.date.max()
    is_valid = train_all["timestamp"].dt.date == last_day
    fit_df = train_all[~is_valid]
    valid_df = train_all[is_valid]

    # The full fit set (~38M rows) OOM-killed a training run with no
    # Python traceback -- Windows just terminated the process. AUC gains
    # were already small between rounds 50 and 100 (0.783 -> 0.789), so
    # subsampling trades a small amount of signal for a training run that
    # actually finishes reliably.
    MAX_FIT_ROWS = 15_000_000
    if len(fit_df) > MAX_FIT_ROWS:
        fit_df = fit_df.sample(n=MAX_FIT_ROWS, random_state=0)
        print(f"Subsampled fit set to {len(fit_df):,} rows for memory safety")

    print(
        f"Fit: {len(fit_df):,} rows ({fit_df['won'].sum():,} won) | "
        f"Valid ({last_day}): {len(valid_df):,} rows ({valid_df['won'].sum():,} won)"
    )

    category_maps = fit_category_maps(fit_df)
    with open(MODELS_DIR / "market_category_maps.json", "w") as f:
        json.dump(category_maps, f)

    checkpoint_path = MODELS_DIR / "market_model_checkpoint.txt"
    booster = train(fit_df, valid_df, category_maps, checkpoint_path=checkpoint_path)
    checkpoint_path.unlink(missing_ok=True)
    booster.save_model(str(MODELS_DIR / "market_model.txt"))
    print(f"Model saved -> {MODELS_DIR / 'market_model.txt'}")

    metrics = {
        "valid": evaluate(booster, valid_df, category_maps),
        "test": evaluate(booster, test_all, category_maps),
    }
    print(json.dumps(metrics, indent=2))
    with open(MODELS_DIR / "market_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Calibration: predicted win rate vs actual win rate, binned by
    # predicted probability -- the direct sanity check the plan calls for
    # ("the win-rate model's calibration against real historical win
    # rates by context").
    from sklearn.calibration import calibration_curve
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = test_all["won"].astype(int).to_numpy()
    y_prob = predict(booster, test_all, category_maps)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="win-rate model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("Mean predicted win rate")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Win-rate model calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "market_calibration.png", dpi=150)
    print(f"Calibration plot -> {FIGURES_DIR / 'market_calibration.png'}")

    # Bid-price response sanity check: under GSP semantics, P(win) should
    # be non-decreasing in bid price. If it isn't, something is wrong with
    # either the model or the feature encoding -- this is the check that
    # decides whether the bandit can trust this model's price-response
    # curve at all.
    sample = test_all.sample(n=min(200_000, len(test_all)), random_state=0)
    prices = np.arange(200, 320, 10)
    win_rates = [float(predict_at_price(booster, sample, category_maps, float(p)).mean()) for p in prices]
    print("Bid price -> mean predicted win rate:")
    for p, w in zip(prices, win_rates):
        print(f"  {p}: {w:.4f}")
    is_monotonic = all(w2 >= w1 - 1e-6 for w1, w2 in zip(win_rates, win_rates[1:]))
    print(f"Monotonic in price: {is_monotonic}")
    with open(MODELS_DIR / "market_price_response.json", "w") as f:
        json.dump({"prices": prices.tolist(), "win_rates": win_rates, "monotonic": is_monotonic}, f, indent=2)


if __name__ == "__main__":
    main()
