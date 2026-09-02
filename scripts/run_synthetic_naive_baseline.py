"""Run the Phase 5 naive (flat-bid) baseline against every scenario in
config/synthetic_scenarios.yaml -- the synthetic analog of
run_naive_baseline.py. Closed-form population expectations
(synthetic.solve_delivery_bid_synthetic / evaluate_flat_bid_synthetic), not
a simulation -- fast even at this scale, no checkpointing needed.

Under first-price, CPM = the flat bid itself by construction (you pay
exactly what you bid on every win, so expected spend / expected impressions
= bid) -- included here anyway for the direct side-by-side with
run_synthetic_bandit.py's results.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.bandit import DEFAULT_BID_BOUNDS  # noqa: E402
from adtech_rtb.synthetic import evaluate_flat_bid_synthetic, solve_delivery_bid_synthetic  # noqa: E402
from adtech_rtb.synthetic_world import load_world  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"
SCENARIOS_PATH = REPO_ROOT / "config" / "synthetic_scenarios.yaml"
REPORTS_DIR = REPO_ROOT / "reports"

DELIVERY_TOLERANCE = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    # Default 0 = continuous bisection (unchanged behavior). A positive value
    # restricts the naive solver to that many discrete bid levels -- the
    # cheapest one that still clears the delivery target -- for a fair
    # comparison against a bandit confined to the same discretization (see
    # run_synthetic_bandit.py's own --bid-levels flag and
    # solve_delivery_bid_synthetic's docstring).
    parser.add_argument("--bid-levels", type=int, default=0)
    args = parser.parse_args()
    bid_levels = None if args.bid_levels == 0 else np.linspace(*DEFAULT_BID_BOUNDS, args.bid_levels)

    world = load_world(WORLD_PATH)
    with open(SCENARIOS_PATH) as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    results = []
    n_delivery_met = 0
    n_ctr_met = 0
    total_spend = 0.0

    for s in scenarios:
        rng = np.random.default_rng(s["seed"])
        solved = solve_delivery_bid_synthetic(
            world, s["campaign_id"], s["target_impressions"], s["n_eligible_auctions"], rng, bid_levels=bid_levels
        )
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
    suffix = "" if args.bid_levels == 0 else f".{args.bid_levels}levels"
    out_path = REPORTS_DIR / f"synthetic_naive_baseline_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results -> {out_path}")


if __name__ == "__main__":
    main()
