"""Stage 2 of the learned-pacing plan (plans/lucky-coalescing-crystal.md):
generate the hindsight-pacing training dataset.

Runs hindsight_pacing.search_hindsight_params + extract_training_pairs
across many simulated flights, using a campaign roster and seed DISJOINT
from every eval scenario file (config/synthetic_scenarios.yaml's
synthA-synthD/seed=42, config/synthetic_scenario_grid.yaml's
meridian_apparel) -- training and eval scenarios must never overlap, or
Stage 4's learned-vs-analytic comparison isn't a real generalization test.

Parallelized across scenarios (each scenario's own 9-candidate grid search
runs sequentially within one process -- see hindsight_pacing.py's cost
estimate, ~45-95s per candidate flight) via multiprocessing.Pool, since the
searches are independent and this is the single most expensive stage in the
whole plan (run as an unattended batch job, not interactively).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.hindsight_pacing import extract_training_pairs, search_hindsight_params  # noqa: E402
from adtech_rtb.synthetic import (  # noqa: E402
    CTR_CATEGORICAL_COLUMNS,
    MARKET_CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SyntheticEnvironment,
    evaluate_flat_bid_synthetic,
    generate_synthetic_scenarios,
    solve_delivery_bid_synthetic,
)
from adtech_rtb.synthetic_world import load_world  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"
REPORTS_DIR = REPO_ROOT / "reports"

# Sized to this machine's 8 cores (~500s/scenario observed for a 9-candidate
# search on a mid-size scenario -- see validation run in
# plans/lucky-coalescing-crystal.md's Stage 1 notes): 24 scenarios / 8
# workers = 3 waves, ~25-35 min wall-clock. A pragmatic v1 scope, not the
# plan's original ~80-flight estimate -- scale TRAINING_CAMPAIGN_IDS/
# N_PER_CAMPAIGN up if the fitted model's residuals look data-starved.
TRAINING_CAMPAIGN_IDS = [f"pacing-train-{i}" for i in range(1, 9)]
TRAINING_SEED = 123
N_PER_CAMPAIGN = 3

POLICY_SEED = 0
BATCH_SIZE = 2000
MAX_OVERRUN_MULTIPLE = 6.0
LOSS_PERCENTILE = 50.0

POLICY_KWARGS = dict(
    market_categorical_columns=MARKET_CATEGORICAL_COLUMNS,
    ctr_categorical_columns=CTR_CATEGORICAL_COLUMNS,
    numeric_columns=NUMERIC_COLUMNS,
)


def _naive_cpm(world, scenario: dict, rng: np.random.Generator) -> float:
    n_flight = scenario["n_eligible_auctions"]
    bid_solution = solve_delivery_bid_synthetic(world, scenario["campaign_id"], scenario["target_impressions"], n_flight, rng)
    naive_eval = evaluate_flat_bid_synthetic(world, scenario["campaign_id"], bid_solution["bid"], n_flight, rng)
    return naive_eval["expected_spend"] / max(naive_eval["expected_impressions"], 1) * 1000.0


def _process_scenario(scenario: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    # Reload the world per-process (spawned on Windows, doesn't inherit
    # parent memory) rather than relying on pickling it across the Pool boundary.
    world = load_world(WORLD_PATH)
    environment = SyntheticEnvironment(world)
    rng = np.random.default_rng(scenario["seed"])
    naive_cpm = _naive_cpm(world, scenario, rng)

    t0 = time.time()
    runs = search_hindsight_params(
        scenario,
        environment,
        naive_cpm,
        POLICY_KWARGS,
        policy_seed=POLICY_SEED,
        outcome_seed=scenario["seed"],
        batch_size=BATCH_SIZE,
        max_overrun_multiple=MAX_OVERRUN_MULTIPLE,
    )
    X, y = extract_training_pairs(runs, scenario, naive_cpm, loss_percentile=LOSS_PERCENTILE)
    elapsed = time.time() - t0
    winner = runs[0]
    print(
        f"  {scenario['id']}: winner={winner['params']} loss={winner['loss']:.4f} "
        f"rows={len(X)} elapsed={elapsed:.0f}s",
        flush=True,
    )
    summary = {
        "scenario_id": scenario["id"],
        "winner_params": winner["params"],
        "winner_loss": round(winner["loss"], 6),
        "naive_cpm": round(naive_cpm, 2),
        "n_rows": len(X),
        "elapsed_seconds": round(elapsed, 1),
    }
    return summary, X, y


def main() -> None:
    world = load_world(WORLD_PATH)
    scenarios = generate_synthetic_scenarios(world, TRAINING_CAMPAIGN_IDS, n_per_campaign=N_PER_CAMPAIGN, seed=TRAINING_SEED)
    print(f"Generated {len(scenarios)} training scenarios (disjoint campaign roster, seed={TRAINING_SEED})")

    n_workers = min(mp.cpu_count(), len(scenarios))
    print(f"Running hindsight search across {n_workers} worker processes...")
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_process_scenario, scenarios)
    print(f"Total wall-clock: {time.time() - t0:.0f}s")

    summaries = [r[0] for r in results]
    non_empty = [(X, y) for _, X, y in results if len(X) > 0]
    all_X = np.concatenate([x for x, _ in non_empty], axis=0) if non_empty else np.empty((0, 4))
    all_y = np.concatenate([y for _, y in non_empty], axis=0) if non_empty else np.empty((0, 2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(REPORTS_DIR / "hindsight_pacing_dataset.npz", X=all_X, y=all_y)
    with open(REPORTS_DIR / "hindsight_pacing_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nTotal training pairs: {len(all_X)}")
    print(f"Wrote dataset -> {REPORTS_DIR / 'hindsight_pacing_dataset.npz'}")
    print(f"Wrote summary -> {REPORTS_DIR / 'hindsight_pacing_summary.json'}")


if __name__ == "__main__":
    main()
