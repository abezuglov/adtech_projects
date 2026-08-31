"""Phase 4: fit a warm-start prior from the other 4 campaigns' real
historical data, apply it to 3358 (config.yaml's warm_start_holdout), and
compare against the cold-start results already in reports/bandit_results.json
for the same 3358-* scenarios -- same scenario definitions, same
environment models, only the online learners' starting point differs.

Reports the exploration-cost story directly: day-1 delivery (does
warm-start skip the cold-start dead zone found in Phase 3's trajectory
data?), overall CPM/CPC, CTR floor margin, and overrun ratio.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import pyarrow as pa
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.market_model import RAW_COLUMNS as MARKET_RAW_COLUMNS  # noqa: E402
from adtech_rtb.simulator import simulate_flight  # noqa: E402
from adtech_rtb.warm_start import apply_prior, fit_prior  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

HOLDOUT_ADVERTISER_ID = 3358
BATCH_SIZE = 2000
PRIOR_SEED = 0
POLICY_SEED = 0
OUTCOME_SEED = 1

TRAIN_RAW_COLUMNS = MARKET_RAW_COLUMNS + ["clicked"]


def _day1_delivered(trajectory: list[dict] | None) -> int:
    if not trajectory:
        return 0
    return max((row["cumulative_delivered"] for row in trajectory if row["days_used"] <= 1), default=0)


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(REPO_ROOT / config["scenarios_file"]) as f:
        all_scenarios = yaml.safe_load(f)["scenarios"]
    holdout_scenarios = [s for s in all_scenarios if s["advertiser_id"] == HOLDOUT_ADVERTISER_ID]

    with open(REPORTS_DIR / "bandit_results.json") as f:
        cold_results = {r["scenario_id"]: r for r in json.load(f)}

    market_booster = lgb.Booster(model_file=str(MODELS_DIR / "market_model.txt"))
    with open(MODELS_DIR / "market_category_maps.json") as f:
        market_category_maps = json.load(f)
    price_booster = lgb.Booster(model_file=str(MODELS_DIR / "market_price_model.txt"))
    with open(MODELS_DIR / "market_price_smear.json") as f:
        price_smear_factor = json.load(f)["smear_factor"]
    ctr_booster = lgb.Booster(model_file=str(MODELS_DIR / "ctr_model.txt"))
    with open(MODELS_DIR / "ctr_category_maps.json") as f:
        ctr_category_maps = json.load(f)

    print("Loading train set for the other 4 campaigns (prior fitting)...", flush=True)
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=TRAIN_RAW_COLUMNS, dtype_backend="pyarrow")
    # Same Arrow "string" 2GB offset-overflow fix train_market_model.py
    # already needed: user_agent's raw strings alone exceed the compact
    # string type's limit once pyarrow concatenates chunks for a boolean
    # filter (here, excluding the holdout advertiser) across ~43M rows.
    train_df["user_agent"] = train_df["user_agent"].astype(pd.ArrowDtype(pa.large_string()))
    print(f"  {len(train_df):,} rows loaded, fitting prior (excluding advertiser {HOLDOUT_ADVERTISER_ID})...", flush=True)
    prior_policy = fit_prior(train_df, exclude_advertiser_id=HOLDOUT_ADVERTISER_ID, seed=PRIOR_SEED)
    del train_df
    print("  prior fit done.", flush=True)

    print("Loading test set (for the actual scenario simulations)...", flush=True)
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet", columns=MARKET_RAW_COLUMNS, dtype_backend="pyarrow")

    warm_results = []
    for s in holdout_scenarios:
        policy = apply_prior(prior_policy, ctr_floor=s["ctr_floor"], seed=POLICY_SEED)
        result = simulate_flight(
            s,
            policy,
            test_df,
            market_booster,
            market_category_maps,
            price_booster,
            price_smear_factor,
            ctr_booster,
            ctr_category_maps,
            batch_size=BATCH_SIZE,
            outcome_seed=OUTCOME_SEED,
            track_trajectory=True,
        )
        warm_results.append(result)

        cold = cold_results[s["id"]]
        cold_cpm = cold["spend"] / cold["delivered_impressions"] * 1000
        warm_cpm = result["spend"] / result["delivered_impressions"] * 1000
        cold_cpc = cold["spend"] / cold["clicks"] if cold["clicks"] else float("nan")
        warm_cpc = result["spend"] / result["clicks"] if result["clicks"] else float("nan")
        cold_day1 = _day1_delivered(cold.get("trajectory"))
        warm_day1 = _day1_delivered(result.get("trajectory"))

        print(
            f"  {s['id']}: cold CPM={cold_cpm:.2f} warm CPM={warm_cpm:.2f} | "
            f"cold CPC={cold_cpc:.2f} warm CPC={warm_cpc:.2f} | "
            f"cold day1_delivered={cold_day1} warm day1_delivered={warm_day1} | "
            f"cold overrun={cold['overrun_ratio']:.2f}x warm overrun={result['overrun_ratio']:.2f}x | "
            f"warm ctr_met={result['ctr_met']}",
            flush=True,
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "warm_start_results.json"
    with open(out_path, "w") as f:
        json.dump(warm_results, f, indent=2)
    print(f"\nWrote results -> {out_path}")


if __name__ == "__main__":
    main()
