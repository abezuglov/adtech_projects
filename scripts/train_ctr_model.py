"""Train the CTR model (Phase 1 market simulator, click side).

Checkpoints the won-only, feature-engineered subsets to data/interim/ so a
crash doesn't require re-reading the multi-GB train/test parquet files,
and saves the trained model + metrics so later phases don't need to retrain.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.ctr_model import RAW_COLUMNS, evaluate, predict, prepare_won_subset, train  # noqa: E402
from adtech_rtb.features import ALL_FEATURE_COLUMNS, fit_category_maps  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Only what the model or downstream reporting needs -- dropping the
# high-cardinality raw ID/hash columns (ip, url, anon_url_id, ad_slot_id,
# ipinyou_id, raw user_agent) keeps this checkpoint small.
CACHE_COLUMNS = ALL_FEATURE_COLUMNS + ["bid_id", "timestamp", "clicked"]


def _load_or_build_won_features(name: str, source_path: Path) -> pd.DataFrame:
    cache_path = INTERIM_DIR / f"ctr_{name}_features.parquet"
    if cache_path.exists():
        print(f"  {name}: loading cached features from {cache_path.name}")
        return pd.read_parquet(cache_path, dtype_backend="pyarrow")
    print(f"  {name}: reading {source_path.name} and building features...")
    df = pd.read_parquet(source_path, columns=RAW_COLUMNS, dtype_backend="pyarrow")
    # user_agent's full raw strings total ~5GB across all rows -- over the
    # 2GB limit for the compact "string" arrow type once concatenated
    # across row groups for the won/lost boolean filter below.
    df["user_agent"] = df["user_agent"].astype(pd.ArrowDtype(pa.large_string()))
    won = prepare_won_subset(df)[CACHE_COLUMNS]
    won.to_parquet(cache_path, index=False)
    print(f"  {name}: {len(won):,} won impressions -> cached")
    return won


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_won = _load_or_build_won_features("train", PROCESSED_DIR / "train.parquet")
    test_won = _load_or_build_won_features("test", PROCESSED_DIR / "test.parquet")

    # Time-based validation split: hold out the last training day so early
    # stopping isn't tuned on data the model will later be fit on.
    last_day = train_won["timestamp"].dt.date.max()
    is_valid = train_won["timestamp"].dt.date == last_day
    fit_df = train_won[~is_valid]
    valid_df = train_won[is_valid]
    print(
        f"Fit: {len(fit_df):,} rows ({fit_df['clicked'].sum()} clicks) | "
        f"Valid ({last_day}): {len(valid_df):,} rows ({valid_df['clicked'].sum()} clicks)"
    )

    # Fit categorical encoding on the fit portion only (not valid/test), so
    # unseen-category handling (code 0) is exercised the same way it would
    # be in real deployment on new data.
    category_maps = fit_category_maps(fit_df)
    with open(MODELS_DIR / "ctr_category_maps.json", "w") as f:
        json.dump(category_maps, f)

    booster = train(fit_df, valid_df, category_maps)
    booster.save_model(str(MODELS_DIR / "ctr_model.txt"))
    print(f"Model saved -> {MODELS_DIR / 'ctr_model.txt'}")

    metrics = {
        "valid": evaluate(booster, valid_df, category_maps),
        "test": evaluate(booster, test_won, category_maps),
    }
    print(json.dumps(metrics, indent=2))
    with open(MODELS_DIR / "ctr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    from sklearn.calibration import calibration_curve
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = test_won["clicked"].astype(int).to_numpy()
    y_prob = predict(booster, test_won, category_maps)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="CTR model")
    ax.plot(
        [0, mean_pred.max()], [0, mean_pred.max()], linestyle="--", color="gray", label="perfect calibration"
    )
    ax.set_xlabel("Mean predicted CTR")
    ax.set_ylabel("Observed CTR")
    ax.set_title("CTR model calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ctr_calibration.png", dpi=150)
    print(f"Calibration plot -> {FIGURES_DIR / 'ctr_calibration.png'}")


if __name__ == "__main__":
    main()
