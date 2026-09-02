"""Supplemental Stage 2 training data, generated 2026-09-02: the cached
reports/hindsight_pacing_dataset.npz has ZERO rows with elapsed_fraction > 1
(confirmed directly: `np.load(...)['X'][:, 0].max()` is ~0.99999...), because
generate_hindsight_pacing_data.py's training scenarios use
generate_synthetic_scenarios' default TARGET_WIN_RATE_FRACTION=(0.1, 0.5) --
never near the 0.85 "challenging" difficulty the designed eval grid uses (see
generate_scenario_grid.py). Every one of that stage's hindsight-search
candidates finished on or before nominal flight length, so LearnedPacingController
has literally never seen a post-deadline state during training -- its
behavior there (including the collapse found on grid-90d-highctr-challenging,
see the investigation this script was written to fix) is pure linear
extrapolation into unseen territory, not a fitted response to real overrun
data. Two new interaction features (overrun_x_pacing_error/overrun_x_ctr_error,
see learned_pacing.STATE_FEATURE_NAMES) were added for exactly this regime,
but a linear fit against zero examples of it necessarily assigns them weight
0 -- confirmed empirically after the first refit attempt.

This script generates a small, disjoint-campaign batch of deliberately
near-ceiling scenarios (win_rate_fraction=0.85 AND ctr_floor near the
population mean, mirroring grid-*-highctr-challenging exactly -- the one
combination known to force real overrun) and runs the same Stage 1 hindsight
search + extract_training_pairs used for the main dataset, so the merged
result actually contains labeled overrun states for the model to fit against.
Kept deliberately small (4 campaigns x 2 scenarios = 8) and short-duration
(14-21 days) to bound the added compute -- this is a targeted patch for one
missing regime, not a full dataset regeneration.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.hindsight_pacing import extract_training_pairs, search_hindsight_params  # noqa: E402
from adtech_rtb.synthetic import (  # noqa: E402
    AUCTIONS_PER_HOUR,
    CTR_CATEGORICAL_COLUMNS,
    MARKET_CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SyntheticEnvironment,
    evaluate_flat_bid_synthetic,
    population_rate_bounds,
    solve_delivery_bid_synthetic,
)
from adtech_rtb.synthetic_world import load_world  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"
REPORTS_DIR = REPO_ROOT / "reports"

# Disjoint from both eval rosters (synthA-D, meridian_apparel) AND the main
# training roster (pacing-train-1..8) -- same non-overlap discipline
# generate_hindsight_pacing_data.py documents.
TIGHT_CAMPAIGN_IDS = [f"pacing-tight-{i}" for i in range(1, 5)]
TIGHT_SEED = 456
N_PER_CAMPAIGN = 2
FLIGHT_LENGTH_DAYS_CHOICES = (14, 21)
WIN_RATE_FRACTION = 0.85  # matches grid-*-challenging exactly
CTR_LEVEL_FRACTION = 1.0  # matches grid-*-highctr exactly (fraction of population_mean_ctr)

POLICY_SEED = 0
BATCH_SIZE = 2000
MAX_OVERRUN_MULTIPLE = 6.0
LOSS_PERCENTILE = 50.0

POLICY_KWARGS = dict(
    market_categorical_columns=MARKET_CATEGORICAL_COLUMNS,
    ctr_categorical_columns=CTR_CATEGORICAL_COLUMNS,
    numeric_columns=NUMERIC_COLUMNS,
)


def _build_tight_scenarios(world) -> list[dict]:
    scenarios = []
    for campaign_id in TIGHT_CAMPAIGN_IDS:
        for index in range(1, N_PER_CAMPAIGN + 1):
            scenario_id = f"synth-{campaign_id}-{index}"
            rng = np.random.default_rng(np.random.SeedSequence([TIGHT_SEED, zlib.crc32(str(campaign_id).encode()), index]))

            flight_length_days = int(rng.choice(FLIGHT_LENGTH_DAYS_CHOICES))
            n_eligible_auctions = AUCTIONS_PER_HOUR * 24 * flight_length_days

            rate_floor, rate_ceiling, mean_ctr = population_rate_bounds(world, campaign_id, rng)
            target_rate = rate_floor + WIN_RATE_FRACTION * (rate_ceiling - rate_floor)
            target_impressions = max(1, round(target_rate * n_eligible_auctions))
            ctr_floor = CTR_LEVEL_FRACTION * mean_ctr

            outcome_seed = int(rng.integers(0, 2**31 - 1))
            scenarios.append(
                {
                    "id": scenario_id,
                    "campaign_id": campaign_id,
                    "flight_length_days": flight_length_days,
                    "seed": outcome_seed,
                    "n_eligible_auctions": n_eligible_auctions,
                    "target_impressions": target_impressions,
                    "ctr_floor": round(ctr_floor, 6),
                    "population_mean_ctr": round(mean_ctr, 6),
                    "naive_baseline_reachable": bool(target_rate < rate_ceiling),
                }
            )
    return scenarios


def _naive_cpm(world, scenario: dict, rng: np.random.Generator) -> float:
    n_flight = scenario["n_eligible_auctions"]
    bid_solution = solve_delivery_bid_synthetic(world, scenario["campaign_id"], scenario["target_impressions"], n_flight, rng)
    naive_eval = evaluate_flat_bid_synthetic(world, scenario["campaign_id"], bid_solution["bid"], n_flight, rng)
    return naive_eval["expected_spend"] / max(naive_eval["expected_impressions"], 1) * 1000.0


def _process_scenario(scenario: dict) -> tuple[dict, np.ndarray, np.ndarray]:
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
    max_elapsed_fraction = float(X[:, 0].max()) if len(X) else 0.0
    print(
        f"  {scenario['id']}: winner={winner['params']} loss={winner['loss']:.4f} "
        f"rows={len(X)} max_elapsed_fraction={max_elapsed_fraction:.2f} elapsed={elapsed:.0f}s",
        flush=True,
    )
    summary = {
        "scenario_id": scenario["id"],
        "winner_params": winner["params"],
        "winner_loss": round(winner["loss"], 6),
        "naive_cpm": round(naive_cpm, 2),
        "n_rows": len(X),
        "max_elapsed_fraction": round(max_elapsed_fraction, 3),
        "elapsed_seconds": round(elapsed, 1),
    }
    return summary, X, y


def main() -> None:
    world = load_world(WORLD_PATH)
    scenarios = _build_tight_scenarios(world)
    print(f"Generated {len(scenarios)} near-ceiling training scenarios (disjoint campaign roster, seed={TIGHT_SEED})")

    n_workers = min(mp.cpu_count(), len(scenarios))
    print(f"Running hindsight search across {n_workers} worker processes...")
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_process_scenario, scenarios)
    print(f"Total wall-clock: {time.time() - t0:.0f}s")

    summaries = [r[0] for r in results]
    non_empty = [(X, y) for _, X, y in results if len(X) > 0]
    new_X = np.concatenate([x for x, _ in non_empty], axis=0) if non_empty else np.empty((0, 7))
    new_y = np.concatenate([y for _, y in non_empty], axis=0) if non_empty else np.empty((0, 2))

    existing_path = REPORTS_DIR / "hindsight_pacing_dataset.npz"
    if existing_path.exists():
        existing = np.load(existing_path)
        existing_X = existing["X"]
        if existing_X.shape[1] == 4:  # pre-pacing_error_sq_behind, see fit_learned_pacing._upgrade_legacy_dataset
            pacing_error_col = existing_X[:, 1]
            sq_behind = np.maximum(pacing_error_col, 0.0) ** 2
            existing_X = np.concatenate([existing_X[:, :2], sq_behind[:, None], existing_X[:, 2:]], axis=1)
        if existing_X.shape[1] == 5:  # pre-overrun interaction terms
            ef, p_err, c_err = existing_X[:, 0], existing_X[:, 1], existing_X[:, 3]
            overrun_fraction = np.maximum(0.0, ef - 1.0)
            existing_X = np.concatenate(
                [
                    existing_X,
                    (overrun_fraction * np.maximum(0.0, p_err))[:, None],
                    (overrun_fraction * np.maximum(0.0, c_err))[:, None],
                ],
                axis=1,
            )
        all_X = np.concatenate([existing_X, new_X], axis=0)
        all_y = np.concatenate([existing["y"], new_y], axis=0)
        print(f"Merged with existing dataset: {len(existing_X)} + {len(new_X)} = {len(all_X)} rows")
    else:
        all_X, all_y = new_X, new_y

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(REPORTS_DIR / "hindsight_pacing_dataset.npz", X=all_X, y=all_y)
    with open(REPORTS_DIR / "hindsight_pacing_summary_overrun.json", "w") as f:
        json.dump(summaries, f, indent=2)

    n_overrun_rows = int((new_X[:, 0] > 1.0).sum()) if len(new_X) else 0
    print(f"\nNew rows: {len(new_X)} ({n_overrun_rows} with elapsed_fraction > 1)")
    print(f"Wrote merged dataset -> {REPORTS_DIR / 'hindsight_pacing_dataset.npz'}")
    print(f"Wrote summary -> {REPORTS_DIR / 'hindsight_pacing_summary_overrun.json'}")


if __name__ == "__main__":
    main()
