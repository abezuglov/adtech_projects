"""Run the Phase 3 cold-start bandit against every committed scenario in
config/scenarios.yaml, and report delivery/spend/CTR results alongside
the naive baseline's (reports/naive_baseline_results.json).

Unlike the naive baseline (a closed-form population expectation), this is
a genuine sequential Monte Carlo simulation (simulator.simulate_flight) --
runtime scales with each scenario's target_impressions, and the largest
scenarios (up to ~950K target) can take 20+ minutes. Two levels of
checkpointing, both gitignored under data/interim/:
  - Per-scenario results, in bandit_results_checkpoint.jsonl (same pattern
    as scripts/generate_scenarios.py) -- an interrupted run skips
    scenarios already finished rather than redoing them.
  - Per-scenario *mid-flight* state, in bandit_midflight_<id>.pkl
    (simulate_flight's checkpoint_path/checkpoint_every_batches) -- a
    crash partway through one of the 20+-minute scenarios resumes from
    the last checkpointed batch (policy weights, delivered/spend/clicks,
    stream position) instead of losing the whole scenario. Deleted
    automatically once that scenario finishes.
Delete the results checkpoint to force a clean re-run from scratch; a
stale mid-flight file is only reused if it matches the scenario currently
being run, so it's safe to leave old ones around.

One BanditPolicy per scenario, fixed seeds (policy_seed=0, outcome_seed=1)
for reproducibility -- a single realization, not averaged over multiple
seeds (a documented limitation: a truly rigorous comparison would run
several seeds per scenario and report a distribution, not a point
estimate; deferred given this is a portfolio project, not a production
evaluation).

Runs with track_trajectory=True (simulator.simulate_flight): each result
carries a per-batch trajectory (cumulative delivered/spend, batch CPM,
lambda_delivery/lambda_ctr) needed to actually see the within-flight
learning curve, not just the flight's final numbers -- a small-scenario
side experiment showed spend efficiency isn't a slow multi-day ramp here,
it's a short day-1 cold-start dead zone (zero bids until pacing debt
builds up) followed by immediately-stable CPM, which is exactly the kind
of thing that needs the per-batch record to see at all.
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.bandit import BanditPolicy  # noqa: E402
from adtech_rtb.features import ALL_CATEGORICAL_COLUMNS, NUMERIC_COLUMNS  # noqa: E402
from adtech_rtb.market_model import MARKET_CATEGORICAL_COLUMNS, RAW_COLUMNS  # noqa: E402
from adtech_rtb.simulator import simulate_flight  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
CHECKPOINT_PATH = REPO_ROOT / "data" / "interim" / "bandit_results_checkpoint.jsonl"

BATCH_SIZE = 2000
POLICY_SEED = 0
OUTCOME_SEED = 1


def _load_checkpoint() -> dict[str, dict]:
    if not CHECKPOINT_PATH.exists():
        return {}
    checkpointed = {}
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                checkpointed[r["scenario_id"]] = r
    return checkpointed


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(REPO_ROOT / config["scenarios_file"]) as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    checkpointed = _load_checkpoint()
    if checkpointed:
        print(f"Resuming: {len(checkpointed)} scenario(s) already checkpointed at {CHECKPOINT_PATH}")

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

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "a") as ckpt_f:
        for s in scenarios:
            if s["id"] in checkpointed:
                continue

            policy = BanditPolicy(
                ctr_floor=s["ctr_floor"],
                market_categorical_columns=MARKET_CATEGORICAL_COLUMNS,
                ctr_categorical_columns=ALL_CATEGORICAL_COLUMNS,
                numeric_columns=NUMERIC_COLUMNS,
                seed=POLICY_SEED,
            )
            t0 = time.time()
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
                checkpoint_path=CHECKPOINT_PATH.parent / f"bandit_midflight_{s['id']}.pkl",
            )
            result["elapsed_seconds"] = round(time.time() - t0, 1)

            ckpt_f.write(json.dumps(result) + "\n")
            ckpt_f.flush()
            checkpointed[s["id"]] = result

            cpm = result["spend"] / max(result["delivered_impressions"], 1) * 1000
            print(
                f"  {s['id']}: delivered={result['delivered_impressions']:,}/{result['target_impressions']:,} "
                f"(met={result['delivery_met']}, capped={result['delivery_capped']}), "
                f"spend={result['spend']:,.0f} RMB, CPM={cpm:.2f}, "
                f"CTR={result['achieved_ctr']:.5f}/{result['ctr_floor']:.5f} (met={result['ctr_met']}), "
                f"overrun={result['overrun_ratio']:.2f}x, elapsed={result['elapsed_seconds']:.0f}s",
                flush=True,
            )

    results = list(checkpointed.values())
    n_delivery_met = sum(r["delivery_met"] for r in results)
    n_ctr_met = sum(r["ctr_met"] for r in results)
    total_spend = sum(r["spend"] for r in results)
    print(
        f"\nSummary: {n_delivery_met}/{len(results)} scenarios met delivery, "
        f"{n_ctr_met}/{len(results)} met CTR floor, total spend = {total_spend:,.0f} RMB"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "bandit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results -> {out_path}")


if __name__ == "__main__":
    main()
