"""Run the Phase 2 naive (flat-bid) baseline against every committed
scenario in config/scenarios.yaml, and report delivery/spend/CTR results.

For each scenario: solve for the single flat bid that clears the delivery
target (bidding.solve_delivery_bid), then evaluate that bid's expected
spend and achieved CTR against the scenario's CTR floor
(bidding.evaluate_flat_bid). Both steps run on the scenario's real 2-day
auction pool, not the (possibly multi-million-row) bootstrapped flight --
a flat bid's outcome is a population-level expectation, so the pool
alone determines it; see bidding.evaluate_flat_bid's docstring.

Results are saved to reports/naive_baseline_results.json so the future
bandit evaluation (Phase 5) can be compared against these same numbers
without re-running this script.
"""

import datetime as dt
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.bidding import evaluate_flat_bid, select_eligible_auctions, solve_delivery_bid  # noqa: E402
from adtech_rtb.market_model import RAW_COLUMNS, prepare_all_bids  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

DELIVERY_TOLERANCE = 0.01  # within 1% of target counts as "met"


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(REPO_ROOT / config["scenarios_file"]) as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    market_booster = lgb.Booster(model_file=str(MODELS_DIR / "market_model.txt"))
    with open(MODELS_DIR / "market_category_maps.json") as f:
        market_category_maps = json.load(f)

    price_booster = lgb.Booster(model_file=str(MODELS_DIR / "market_price_model.txt"))
    with open(MODELS_DIR / "market_price_smear.json") as f:
        price_smear_factor = json.load(f)["smear_factor"]

    ctr_booster = lgb.Booster(model_file=str(MODELS_DIR / "ctr_model.txt"))
    with open(MODELS_DIR / "ctr_category_maps.json") as f:
        ctr_category_maps = json.load(f)

    print("Loading test set (raw columns, pre-feature-engineering)...", flush=True)
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet", columns=RAW_COLUMNS, dtype_backend="pyarrow")

    results = []
    n_delivery_met = 0
    n_ctr_met = 0
    total_spend = 0.0

    for s in scenarios:
        pool_dates = [dt.date.fromisoformat(d) for d in s["pool_dates"]]
        pool_df = select_eligible_auctions(
            test_df, s["advertiser_id"], pool_dates, region_ids=s["region_ids"], city_ids=s["city_ids"]
        )
        pool_contexts = prepare_all_bids(pool_df)

        # Scenario's target/n_eligible are flight-scale; translate to the
        # same *rate* on the (much smaller) real pool, since that's what
        # the solver and evaluator both actually operate on.
        target_rate = s["target_impressions"] / s["n_eligible_auctions"]
        pool_target = max(1, round(target_rate * len(pool_contexts)))

        solved = solve_delivery_bid(market_booster, pool_contexts, market_category_maps, pool_target)
        outcome = evaluate_flat_bid(
            market_booster,
            market_category_maps,
            price_booster,
            price_smear_factor,
            ctr_booster,
            ctr_category_maps,
            pool_contexts,
            solved["bid"],
            s["n_eligible_auctions"],
        )

        delivery_ratio = outcome["expected_impressions"] / s["target_impressions"]
        delivery_met = abs(delivery_ratio - 1.0) <= DELIVERY_TOLERANCE or delivery_ratio >= 1.0
        ctr_met = outcome["achieved_ctr"] >= s["ctr_floor"]

        result = {
            "scenario_id": s["id"],
            "bid": outcome["bid"],
            "target_impressions": s["target_impressions"],
            "expected_impressions": outcome["expected_impressions"],
            "delivery_met": delivery_met,
            "expected_spend": outcome["expected_spend"],
            "expected_clicks": outcome["expected_clicks"],
            "ctr_floor": s["ctr_floor"],
            "achieved_ctr": outcome["achieved_ctr"],
            "ctr_met": ctr_met,
            "target_reachable_in_bounds": solved["target_reachable_in_bounds"],
        }
        results.append(result)
        n_delivery_met += delivery_met
        n_ctr_met += ctr_met
        total_spend += outcome["expected_spend"]

        print(
            f"  {s['id']}: bid={outcome['bid']:.1f}, "
            f"impressions={outcome['expected_impressions']:,.0f}/{s['target_impressions']:,} "
            f"(met={delivery_met}), spend={outcome['expected_spend']:,.0f} RMB, "
            f"CTR={outcome['achieved_ctr']:.5f}/{s['ctr_floor']:.5f} (met={ctr_met})",
            flush=True,
        )

    print(
        f"\nSummary: {n_delivery_met}/{len(scenarios)} scenarios met delivery, "
        f"{n_ctr_met}/{len(scenarios)} met CTR floor, total expected spend = {total_spend:,.0f} RMB"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "naive_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results -> {out_path}")


if __name__ == "__main__":
    main()
