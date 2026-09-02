"""Run the Phase 5 first-price cold-start bandit against the designed 2x2x2
scenario grid (config/synthetic_scenario_grid.yaml, see
generate_scenario_grid.py) and report delivery/spend/CTR/CPM by cell -- the
grid-review analog of run_synthetic_bandit.py, which runs the random
per-campaign scenario set instead.

Same fixed seeds / no-checkpointing rationale as run_synthetic_bandit.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.bandit import DEFAULT_BID_BOUNDS, BanditPolicy  # noqa: E402
from adtech_rtb.synthetic import (  # noqa: E402
    CTR_CATEGORICAL_COLUMNS,
    MARKET_CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SyntheticEnvironment,
    simulate_synthetic_flight,
)
from adtech_rtb.synthetic_world import load_world  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_synthetic_bandit import build_pacing_controller  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"
SCENARIOS_PATH = REPO_ROOT / "config" / "synthetic_scenario_grid.yaml"
REPORTS_DIR = REPO_ROOT / "reports"

BATCH_SIZE = 2000
POLICY_SEED = 0
OUTCOME_SEED = 1
# Same rationale as run_synthetic_bandit.py's cap, raised further: the
# "challenging" cells target win_rate_fraction=0.85 (needing to win most
# auctions to pace), which leaves far less delivery-rate slack than that
# script's random draws (max 0.5) ever produced -- a low/no-headroom target
# is expected to run close to the nominal flight length rather than
# overrunning, but cheap to give generous runway here regardless.
MAX_OVERRUN_MULTIPLE = 6.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacing", choices=["analytic", "learned"], default="analytic")
    parser.add_argument("--bid-levels", type=int, default=6)
    args = parser.parse_args()

    world = load_world(WORLD_PATH)
    with open(SCENARIOS_PATH) as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    environment = SyntheticEnvironment(world)
    pacing_controller = build_pacing_controller(args.pacing)
    bid_levels = np.linspace(DEFAULT_BID_BOUNDS[0], DEFAULT_BID_BOUNDS[1], args.bid_levels)

    results = []
    for s in scenarios:
        policy = BanditPolicy(
            ctr_floor=s["ctr_floor"],
            seed=POLICY_SEED,
            first_price=True,
            market_categorical_columns=MARKET_CATEGORICAL_COLUMNS,
            ctr_categorical_columns=CTR_CATEGORICAL_COLUMNS,
            numeric_columns=NUMERIC_COLUMNS,
            bid_levels=bid_levels,
        )
        t0 = time.time()
        result = simulate_synthetic_flight(
            s,
            policy,
            environment,
            batch_size=BATCH_SIZE,
            outcome_seed=OUTCOME_SEED,
            max_overrun_multiple=MAX_OVERRUN_MULTIPLE,
            track_trajectory=True,
            pacing_controller=pacing_controller,
        )
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        result["duration"] = s["duration"]
        result["ctr_level"] = s["ctr_level"]
        result["avails"] = s["avails"]
        results.append(result)

        cpm = result["spend"] / max(result["delivered_impressions"], 1) * 1000
        cpc = result["spend"] / result["clicks"] if result["clicks"] > 0 else None
        print(
            f"  {s['id']:38s} duration={s['duration']:>3s} ctr={s['ctr_level']:>8s} avails={s['avails']:>11s} | "
            f"delivered={result['delivered_impressions']:>9,}/{result['target_impressions']:>9,} "
            f"(met={result['delivery_met']}, capped={result['delivery_capped']}), "
            f"CPM={cpm:6.2f}, CPC={cpc if cpc is None else round(cpc, 2)}, "
            f"CTR={result['achieved_ctr']:.5f}/{result['ctr_floor']:.5f} (met={result['ctr_met']}), "
            f"overrun={result['overrun_ratio']:.2f}x, cv={result['delivery_cv']:.2f}, "
            f"t={result['elapsed_seconds']:.0f}s",
            flush=True,
        )

    n_delivery_met = sum(r["delivery_met"] for r in results)
    n_ctr_met = sum(r["ctr_met"] for r in results)
    total_spend = sum(r["spend"] for r in results)
    print(
        f"\nSummary: {n_delivery_met}/{len(results)} scenarios met delivery, "
        f"{n_ctr_met}/{len(results)} met CTR floor, total spend = {total_spend:,.0f}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.pacing == "analytic" else f".{args.pacing}_pacing"
    suffix += "" if args.bid_levels == 6 else f".{args.bid_levels}levels"
    out_path = REPORTS_DIR / f"synthetic_scenario_grid_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results -> {out_path}")


if __name__ == "__main__":
    main()
