"""Run the closed-form naive (flat-bid) baseline against the designed 2x2x2
scenario grid (config/synthetic_scenario_grid.yaml) -- the grid analog of
run_synthetic_naive_baseline.py, which runs against the random per-campaign
scenario set instead. Needed for app.py's side-by-side bandit-vs-naive
comparison once the grid is wired into the live demo.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.synthetic import evaluate_flat_bid_synthetic, solve_delivery_bid_synthetic  # noqa: E402
from adtech_rtb.synthetic_world import load_world  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"
SCENARIOS_PATH = REPO_ROOT / "config" / "synthetic_scenario_grid.yaml"
REPORTS_DIR = REPO_ROOT / "reports"

DELIVERY_TOLERANCE = 0.01


def main() -> None:
    world = load_world(WORLD_PATH)
    with open(SCENARIOS_PATH) as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    results = []
    n_delivery_met = 0
    n_ctr_met = 0
    total_spend = 0.0

    for s in scenarios:
        rng = np.random.default_rng(s["seed"])
        solved = solve_delivery_bid_synthetic(world, s["campaign_id"], s["target_impressions"], s["n_eligible_auctions"], rng)
        outcome = evaluate_flat_bid_synthetic(world, s["campaign_id"], solved["bid"], s["n_eligible_auctions"], rng)

        delivery_ratio = outcome["expected_impressions"] / s["target_impressions"]
        delivery_met = abs(delivery_ratio - 1.0) <= DELIVERY_TOLERANCE or delivery_ratio >= 1.0
        ctr_met = outcome["achieved_ctr"] >= s["ctr_floor"]
        cpm = outcome["expected_spend"] / max(outcome["expected_impressions"], 1) * 1000
        cpc = outcome["expected_spend"] / outcome["expected_clicks"] if outcome["expected_clicks"] > 0 else None

        result = {
            "scenario_id": s["id"],
            "bid": outcome["bid"],
            "cpm": cpm,
            "cpc": cpc,
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
            f"  {s['id']}: bid={outcome['bid']:.1f}, CPM={cpm:.2f}, "
            f"impressions={outcome['expected_impressions']:,.0f}/{s['target_impressions']:,} "
            f"(met={delivery_met}), spend={outcome['expected_spend']:,.0f}, "
            f"CTR={outcome['achieved_ctr']:.5f}/{s['ctr_floor']:.5f} (met={ctr_met})",
            flush=True,
        )

    print(
        f"\nSummary: {n_delivery_met}/{len(scenarios)} scenarios met delivery, "
        f"{n_ctr_met}/{len(scenarios)} met CTR floor, total expected spend = {total_spend:,.0f}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "synthetic_naive_baseline_grid_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results -> {out_path}")


if __name__ == "__main__":
    main()
