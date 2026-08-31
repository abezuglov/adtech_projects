"""Exploration tool for picking Phase 2 naive-baseline scenarios.

config/config.yaml's `scenarios` list is still empty -- which geo(s) and
delivery targets to use per campaign is being picked experimentally
against how much auction volume each geo actually has (thin volume makes
the marginal win-rate curve noisy). This script:

  1. Prints per-region_id daily auction volume for each campaign, so a
     geo with enough data can be picked by inspection.
  2. Runs the naive-baseline bid solver (src/adtech_rtb/bidding.py) for
     one illustrative scenario per campaign (its single highest-volume
     region on the first test day, targeting the delivery it actually
     achieved historically in that slice) as a smoke test that the
     solver behaves sensibly against real data -- not a claim that these
     are the final scenario choices for config.yaml.

Run after scripts/train_market_model.py (needs models/market_model.txt +
models/market_category_maps.json).
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.bidding import geo_volume_table, select_eligible_auctions, solve_delivery_bid  # noqa: E402
from adtech_rtb.market_model import RAW_COLUMNS, prepare_all_bids  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"


def main() -> None:
    with open(REPO_ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    advertiser_ids = [a["id"] for a in config["advertisers"]]

    booster = lgb.Booster(model_file=str(MODELS_DIR / "market_model.txt"))
    with open(MODELS_DIR / "market_category_maps.json") as f:
        category_maps = json.load(f)

    print("Loading test set (raw columns, pre-feature-engineering)...")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet", columns=RAW_COLUMNS, dtype_backend="pyarrow")
    first_day = test_df["timestamp"].dt.date.min()
    print(f"Using {first_day} (first test day) for this exploration pass.\n")

    for advertiser_id in advertiser_ids:
        print(f"=== advertiser {advertiser_id} ===")
        volumes = geo_volume_table(test_df, advertiser_id, [first_day], by="region_id")
        print("Top 5 regions by daily bid-request volume:")
        print(volumes.head(5).to_string())

        if volumes.empty:
            print("  no auctions this day -- skipping solver demo\n")
            continue

        top_region = int(volumes.index[0])
        scenario_df = select_eligible_auctions(
            test_df, advertiser_id, [first_day], region_ids=[top_region]
        )
        historical_won = int(scenario_df["won"].sum())
        target = max(1, historical_won)

        contexts = prepare_all_bids(scenario_df)
        result = solve_delivery_bid(booster, contexts, category_maps, target_impressions=target)
        print(
            f"  scenario: region_id={top_region}, day={first_day}, "
            f"n_eligible={result['n_eligible']:,}, target={target} (= historical won count)"
        )
        print(
            f"  solved bid={result['bid']:.1f} -> expected_impressions="
            f"{result['expected_impressions']:.1f} (achieved_win_rate={result['achieved_win_rate']:.4f}), "
            f"reachable_in_bounds={result['target_reachable_in_bounds']}"
        )
        print()


if __name__ == "__main__":
    main()
