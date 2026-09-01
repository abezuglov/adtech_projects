"""Draw the Phase 5 synthetic world (synthetic_world.build_world) and
persist it to data/interim/synthetic_world.json.

Run once; re-run only if synthetic_world.py's generative model or
WORLD_SEED changes -- every other synthetic script (scenario generation,
naive baseline, bandit) loads the persisted world rather than redrawing it,
so they all see the identical placement/campaign economics.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.synthetic_world import N_PLACEMENTS, build_world, save_world  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "data" / "interim" / "synthetic_world.json"


def main() -> None:
    print(f"Drawing hierarchical world (N_PLACEMENTS={N_PLACEMENTS:,})...", flush=True)
    t0 = time.time()
    world = build_world()
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    pw = world.placement_weight
    sorted_pw = np.sort(pw)[::-1]
    print(f"  placement volume skew: top 1% share={sorted_pw[:100].sum():.3f}, top 10% share={sorted_pw[:1000].sum():.3f}")
    print(f"  beta (price-sensitivity) range: [{world.beta.min():.4f}, {world.beta.max():.4f}], all > 0: {(world.beta > 0).all()}")
    print(f"  clearing_level range: [{world.clearing_level.min():.1f}, {world.clearing_level.max():.1f}]")
    print(f"  affinity_sigma: {world.affinity_sigma:.4f}")

    save_world(world, WORLD_PATH)
    print(f"Wrote world -> {WORLD_PATH}")


if __name__ == "__main__":
    main()
