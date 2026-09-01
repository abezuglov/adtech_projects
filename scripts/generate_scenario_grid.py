"""Generate a designed 2x2x2 scenario grid -- duration x CTR level x avails
difficulty -- for reviewing bandit behavior across explicit, deliberately
chosen operating conditions, as opposed to generate_synthetic_scenarios.py's
per-campaign random draws (useful for coverage, not for isolating one axis
at a time).

campaign_id is held fixed across all 8 cells so campaign CTR affinity isn't
a confound: only flight_length_days, ctr_floor (as a fraction of that one
campaign's own population mean CTR), and target_win_rate_fraction (as a
fraction of that campaign's achievable win-rate band) vary, one axis at a
time, matching a proper factorial design.

Axis -> knob mapping (confirmed with the user):
  - duration: flight_length_days directly (30d vs a 3-month 90d flight --
    wider than FLIGHT_LENGTH_DAYS_RANGE's random-draw ceiling of 28, fine
    since this script sets it explicitly rather than sampling it).
  - ctr level: ctr_floor as a fraction of population_mean_ctr. "Moderate"
    sits at CTR_FLOOR_FRACTION's low end (an easy floor to clear). "High" is
    set to 1.0x population_mean_ctr, not CTR_FLOOR_FRACTION's own 0.9 top --
    a stress test on a different scenario (synthA-2, run by a peer session,
    2026-08-31) found achieved CTR plateaus around ~1.15-1.2x population
    mean regardless of how much higher the floor is pushed past that (the
    eligible pool's above-average-CTR inventory is finite once a delivery
    target also has to be met), with 0.82x landing effortless and 1.0x the
    genuinely tight-but-feasible boundary. 0.88x (this range's original
    choice) sat inside the effortless zone on their data, so didn't actually
    force real placement selection. Note that boundary was measured on a
    15-day flight at a different avails level than this grid's own cells --
    expected to shift with duration/avails, so treat 1.0x here as a
    deliberately-chosen stress point to spot-check, not an exact transplant.
  - avails: target_win_rate_fraction, i.e. where the delivery target sits
    within [rate_floor, rate_ceiling]. "Easy" sits at TARGET_WIN_RATE_FRACTION's
    low end (target reachable with plenty of headroom to skip pricey
    auctions). "Challenging" is pushed well past that range's own top
    (0.85, vs the range's 0.5 max) -- per the user's own framing, challenging
    avails means needing to win *most* auctions just to hold pace, i.e. a
    target close to rate_ceiling, not merely mid-band.
"""

import sys
import zlib
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.synthetic import AUCTIONS_PER_HOUR, population_rate_bounds  # noqa: E402
from adtech_rtb.synthetic_world import load_world  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"
SCENARIOS_PATH = REPO_ROOT / "config" / "synthetic_scenario_grid.yaml"

GRID_CAMPAIGN_ID = "meridian_apparel"
GRID_SEED = 7

DURATIONS = {"30d": 30, "90d": 90}
CTR_LEVELS = {"moderate": 0.55, "high": 1.0}
AVAILS = {"easy": 0.15, "challenging": 0.85}


def main() -> None:
    world = load_world(WORLD_PATH)
    rng = np.random.default_rng(GRID_SEED)
    rate_floor, rate_ceiling, mean_ctr = population_rate_bounds(world, GRID_CAMPAIGN_ID, rng)

    scenarios = []
    for duration_label, days in DURATIONS.items():
        for ctr_label, ctr_frac in CTR_LEVELS.items():
            for avails_label, win_frac in AVAILS.items():
                scenario_id = f"grid-{duration_label}-{ctr_label}ctr-{avails_label}"
                n_eligible_auctions = AUCTIONS_PER_HOUR * 24 * days

                target_rate = rate_floor + win_frac * (rate_ceiling - rate_floor)
                target_impressions = max(1, round(target_rate * n_eligible_auctions))
                ctr_floor = ctr_frac * mean_ctr

                outcome_seed = zlib.crc32(scenario_id.encode()) % (2**31 - 1)

                scenarios.append(
                    {
                        "id": scenario_id,
                        "campaign_id": GRID_CAMPAIGN_ID,
                        "duration": duration_label,
                        "ctr_level": ctr_label,
                        "avails": avails_label,
                        "flight_length_days": days,
                        "seed": int(outcome_seed),
                        "n_eligible_auctions": n_eligible_auctions,
                        "target_impressions": target_impressions,
                        "target_win_rate_fraction": win_frac,
                        "ctr_floor": round(ctr_floor, 6),
                        "population_mean_ctr": round(mean_ctr, 6),
                        "naive_baseline_reachable": bool(target_rate < rate_ceiling),
                    }
                )

    for s in scenarios:
        print(
            f"  {s['id']}: days={s['flight_length_days']}, target={s['target_impressions']:,} "
            f"(win_rate_frac={s['target_win_rate_fraction']}), ctr_floor={s['ctr_floor']:.5f} "
            f"(pop. mean {mean_ctr:.5f}), reachable={s['naive_baseline_reachable']}",
            flush=True,
        )

    SCENARIOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCENARIOS_PATH, "w") as f:
        f.write(
            "# Generated by scripts/generate_scenario_grid.py -- do not hand-edit.\n"
            "# Designed 2x2x2 factorial grid (duration x ctr_level x avails), not a\n"
            "# random draw -- see that script's docstring for the axis -> knob mapping.\n"
        )
        yaml.dump({"scenarios": scenarios}, f, default_flow_style=False, sort_keys=False)
    print(f"\nWrote {len(scenarios)} scenarios -> {SCENARIOS_PATH}")


if __name__ == "__main__":
    main()
