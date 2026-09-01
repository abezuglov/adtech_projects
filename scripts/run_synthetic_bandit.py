"""Run the Phase 5 first-price cold-start bandit against every scenario in
config/synthetic_scenarios.yaml, and report delivery/spend/CTR/CPM/CPC
results alongside the naive baseline's
(reports/synthetic_naive_baseline_results.json) -- the synthetic analog of
run_bandit.py.

No checkpointing (unlike run_bandit.py): synthetic flights are pure numpy
against a frozen World, no LightGBM inference and no multi-million-row
parquet load per batch, so even the largest scenarios run in a few minutes,
not 20+ -- the crash-resilience checkpointing.py machinery real runs needed
isn't worth the complexity here.

One BanditPolicy per scenario, fixed seeds (POLICY_SEED=0, OUTCOME_SEED=1),
first_price=True (see bandit.py's choose_bids/observe) -- a single
realization, not averaged over multiple seeds, same documented limitation
run_bandit.py carries for the real pipeline.
"""

import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.bandit import BanditPolicy  # noqa: E402
from adtech_rtb.synthetic import (  # noqa: E402
    CTR_CATEGORICAL_COLUMNS,
    MARKET_CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SyntheticEnvironment,
    simulate_synthetic_flight,
)
from adtech_rtb.synthetic_world import load_world  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"
SCENARIOS_PATH = REPO_ROOT / "config" / "synthetic_scenarios.yaml"
REPORTS_DIR = REPO_ROOT / "reports"

BATCH_SIZE = 2000
POLICY_SEED = 0
OUTCOME_SEED = 1
# Higher than simulate_flight's real-data default (3.0x): 6/12 scenarios in
# the first full run hit that cap at 90-95% delivered despite a steady,
# non-decelerating delivery rate (low delivery_cv) -- confirmed on the
# worst case that it just needed more runway (finished at 3.29x, well under
# 6x, with no change to CPM or smoothness). Cheap to extend here since
# synthetic flights are pure numpy with no per-extra-day real cost, unlike
# the real pipeline where a higher cap means materializing more bootstrap
# rounds against actual model inference.
MAX_OVERRUN_MULTIPLE = 6.0


def main() -> None:
    world = load_world(WORLD_PATH)
    with open(SCENARIOS_PATH) as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    environment = SyntheticEnvironment(world)

    results = []
    n_delivery_met = 0
    n_ctr_met = 0
    total_spend = 0.0

    for s in scenarios:
        policy = BanditPolicy(
            ctr_floor=s["ctr_floor"],
            seed=POLICY_SEED,
            first_price=True,
            market_categorical_columns=MARKET_CATEGORICAL_COLUMNS,
            ctr_categorical_columns=CTR_CATEGORICAL_COLUMNS,
            numeric_columns=NUMERIC_COLUMNS,
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
        )
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        results.append(result)

        cpm = result["spend"] / max(result["delivered_impressions"], 1) * 1000
        cpc = result["spend"] / result["clicks"] if result["clicks"] > 0 else None
        n_delivery_met += result["delivery_met"]
        n_ctr_met += result["ctr_met"]
        total_spend += result["spend"]

        print(
            f"  {s['id']}: delivered={result['delivered_impressions']:,}/{result['target_impressions']:,} "
            f"(met={result['delivery_met']}, capped={result['delivery_capped']}), "
            f"spend={result['spend']:,.0f}, CPM={cpm:.2f}, CPC={cpc if cpc is None else round(cpc, 2)}, "
            f"CTR={result['achieved_ctr']:.5f}/{result['ctr_floor']:.5f} (met={result['ctr_met']}), "
            f"overrun={result['overrun_ratio']:.2f}x, delivery_cv={result['delivery_cv']:.2f}, "
            f"elapsed={result['elapsed_seconds']:.0f}s",
            flush=True,
        )

    print(
        f"\nSummary: {n_delivery_met}/{len(results)} scenarios met delivery, "
        f"{n_ctr_met}/{len(results)} met CTR floor, total spend = {total_spend:,.0f}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "synthetic_bandit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results -> {out_path}")


if __name__ == "__main__":
    main()
