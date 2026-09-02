"""Stage 4 evaluation (plans/lucky-coalescing-crystal.md): compare
--pacing analytic vs --pacing learned across BOTH eval scenario files
(config/synthetic_scenarios.yaml's 12, config/synthetic_scenario_grid.yaml's
8) -- never the Stage 2 training scenarios, which use a disjoint campaign
roster specifically so this comparison is a real generalization test.

Reads already-produced result files rather than re-running simulations
itself (run scripts/run_synthetic_bandit.py and scripts/run_scenario_grid.py
with both --pacing values first) -- keeps "run a flight" and "aggregate
results" as separate concerns, matching this project's existing script
layout (run_* vs nothing-else-does-both).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

# (analytic results, learned results, naive-baseline results, label)
COMPARISON_SETS = [
    (
        REPORTS_DIR / "synthetic_bandit_results.json",
        REPORTS_DIR / "synthetic_bandit_results.learned_pacing.json",
        REPORTS_DIR / "synthetic_naive_baseline_results.json",
        "random per-campaign scenarios",
    ),
    (
        REPORTS_DIR / "synthetic_scenario_grid_results.json",
        REPORTS_DIR / "synthetic_scenario_grid_results.learned_pacing.json",
        REPORTS_DIR / "synthetic_naive_baseline_grid_results.json",
        "designed 2x2x2 grid",
    ),
]


def _cpm(result: dict) -> float:
    return result["spend"] / max(result["delivered_impressions"], 1) * 1000.0


def _naive_cpm(naive_result: dict) -> float:
    return naive_result["cpm"]


def compare_one_set(analytic_path: Path, learned_path: Path, naive_path: Path, label: str) -> list[dict]:
    if not analytic_path.exists() or not learned_path.exists():
        print(f"Skipping '{label}': missing {analytic_path.name if not analytic_path.exists() else learned_path.name}")
        return []

    with open(analytic_path) as f:
        analytic_results = {r["scenario_id"]: r for r in json.load(f)}
    with open(learned_path) as f:
        learned_results = {r["scenario_id"]: r for r in json.load(f)}
    naive_cpms = {}
    if naive_path.exists():
        with open(naive_path) as f:
            naive_results = json.load(f)
        naive_cpms = {r.get("scenario_id", r.get("id")): _naive_cpm(r) for r in naive_results}

    rows = []
    print(f"\n=== {label} ===")
    for scenario_id in analytic_results:
        if scenario_id not in learned_results:
            continue
        a, l = analytic_results[scenario_id], learned_results[scenario_id]  # noqa: E741
        cpm_a, cpm_l = _cpm(a), _cpm(l)
        naive_cpm = naive_cpms.get(scenario_id)
        row = {
            "scenario_id": scenario_id,
            "cpm_analytic": round(cpm_a, 2),
            "cpm_learned": round(cpm_l, 2),
            "cpm_change_pct": round((cpm_l - cpm_a) / cpm_a * 100, 1) if cpm_a > 0 else None,
            "cpm_vs_naive_analytic_pct": round((cpm_a - naive_cpm) / naive_cpm * 100, 1) if naive_cpm else None,
            "cpm_vs_naive_learned_pct": round((cpm_l - naive_cpm) / naive_cpm * 100, 1) if naive_cpm else None,
            "delivery_cv_analytic": round(a["delivery_cv"], 2),
            "delivery_cv_learned": round(l["delivery_cv"], 2),
            "overrun_analytic": round(a["overrun_ratio"], 2),
            "overrun_learned": round(l["overrun_ratio"], 2),
            "delivery_met_analytic": a["delivery_met"],
            "delivery_met_learned": l["delivery_met"],
            "ctr_met_analytic": a["ctr_met"],
            "ctr_met_learned": l["ctr_met"],
        }
        rows.append(row)
        print(
            f"  {scenario_id:24s} CPM {cpm_a:7.2f} -> {cpm_l:7.2f} ({row['cpm_change_pct']:+.1f}%), "
            f"CV {row['delivery_cv_analytic']:.2f} -> {row['delivery_cv_learned']:.2f}, "
            f"overrun {row['overrun_analytic']:.2f}x -> {row['overrun_learned']:.2f}x, "
            f"met(d/ctr) {row['delivery_met_analytic']}/{row['ctr_met_analytic']} -> "
            f"{row['delivery_met_learned']}/{row['ctr_met_learned']}"
        )
    return rows


def main() -> None:
    all_rows = []
    for analytic_path, learned_path, naive_path, label in COMPARISON_SETS:
        all_rows.extend(compare_one_set(analytic_path, learned_path, naive_path, label))

    if not all_rows:
        print("\nNo comparable result files found -- run both --pacing values first (see module docstring).")
        sys.exit(1)

    n = len(all_rows)
    avg_cpm_change = sum(r["cpm_change_pct"] for r in all_rows if r["cpm_change_pct"] is not None) / n
    n_delivery_met_analytic = sum(r["delivery_met_analytic"] for r in all_rows)
    n_delivery_met_learned = sum(r["delivery_met_learned"] for r in all_rows)
    n_ctr_met_analytic = sum(r["ctr_met_analytic"] for r in all_rows)
    n_ctr_met_learned = sum(r["ctr_met_learned"] for r in all_rows)
    avg_cv_analytic = sum(r["delivery_cv_analytic"] for r in all_rows) / n
    avg_cv_learned = sum(r["delivery_cv_learned"] for r in all_rows) / n

    print(f"\n=== Summary across {n} scenarios ===")
    print(f"Avg CPM change (learned vs analytic): {avg_cpm_change:+.1f}%")
    print(f"Avg delivery_cv: {avg_cv_analytic:.2f} -> {avg_cv_learned:.2f}")
    print(f"Delivery target met: {n_delivery_met_analytic}/{n} -> {n_delivery_met_learned}/{n}")
    print(f"CTR floor met: {n_ctr_met_analytic}/{n} -> {n_ctr_met_learned}/{n}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "pacing_comparison.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nWrote comparison -> {REPORTS_DIR / 'pacing_comparison.json'}")


if __name__ == "__main__":
    main()
