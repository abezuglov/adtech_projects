"""Stage 3 of the learned-pacing plan (plans/lucky-coalescing-crystal.md):
fit the learned pacing function from Stage 2's dataset
(reports/hindsight_pacing_dataset.npz).

De-risks with a plain least-squares fit (bias + 4 features -> 2 targets)
rather than reaching straight for bandit.OnlineBayesianLinearModel's IRLS
machinery -- this is a one-shot batch fit on a static dataset (no online
updates, no sequential correlation to worry about), so lstsq's closed-form
solution is both simpler and exact for this problem; upgrading to the GLM
machinery is only worth it if these residuals look bad enough to need
non-Gaussian noise handling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.learned_pacing import STATE_FEATURE_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "reports" / "hindsight_pacing_dataset.npz"
MODEL_PATH = REPO_ROOT / "data" / "interim" / "learned_pacing_model.json"


def _upgrade_legacy_dataset(X: np.ndarray) -> np.ndarray:
    """The cached Stage 2 dataset (reports/hindsight_pacing_dataset.npz) was
    generated before `pacing_error_sq_behind` existed -- it has 4 columns
    (elapsed_fraction, pacing_error, ctr_error, delivered_fraction), one
    short of the current 5-feature STATE_FEATURE_NAMES. Rather than re-run
    the ~1hr hindsight search just to add a feature that's a deterministic
    function of a column already in the dataset, derive it here and insert
    it in the right slot. A full regeneration (generate_hindsight_pacing_data.py)
    would produce the 5-column version natively -- this is a one-time bridge,
    not the long-term path.
    """
    if X.shape[1] == len(STATE_FEATURE_NAMES):
        return X
    if X.shape[1] != len(STATE_FEATURE_NAMES) - 1:
        raise ValueError(f"Unexpected feature count {X.shape[1]}, expected {len(STATE_FEATURE_NAMES)} or {len(STATE_FEATURE_NAMES) - 1}")
    pacing_error_col = X[:, 1]
    sq_behind = np.maximum(pacing_error_col, 0.0) ** 2
    return np.concatenate([X[:, :2], sq_behind[:, None], X[:, 2:]], axis=1)


def main() -> None:
    data = np.load(DATASET_PATH)
    X, y = _upgrade_legacy_dataset(data["X"]), data["y"]
    print(f"Loaded {len(X)} training pairs, {X.shape[1]} features: {STATE_FEATURE_NAMES}")

    X_bias = np.concatenate([np.ones((len(X), 1)), X], axis=1)

    delivery_weights, *_ = np.linalg.lstsq(X_bias, y[:, 0], rcond=None)
    ctr_weights, *_ = np.linalg.lstsq(X_bias, y[:, 1], rcond=None)

    pred_delivery = X_bias @ delivery_weights
    pred_ctr = X_bias @ ctr_weights
    mae_delivery = float(np.mean(np.abs(pred_delivery - y[:, 0])))
    mae_ctr = float(np.mean(np.abs(pred_ctr - y[:, 1])))
    r2_delivery = 1.0 - np.sum((pred_delivery - y[:, 0]) ** 2) / max(np.sum((y[:, 0] - y[:, 0].mean()) ** 2), 1e-12)
    r2_ctr = 1.0 - np.sum((pred_ctr - y[:, 1]) ** 2) / max(np.sum((y[:, 1] - y[:, 1].mean()) ** 2), 1e-12)

    print(
        f"lambda_delivery: MAE={mae_delivery:.4f}, R^2={r2_delivery:.3f}, "
        f"target mean={y[:, 0].mean():.4f} std={y[:, 0].std():.4f}"
    )
    print(f"lambda_ctr:      MAE={mae_ctr:.4f}, R^2={r2_ctr:.3f}, target mean={y[:, 1].mean():.4f} std={y[:, 1].std():.4f}")

    model = {
        "feature_names": ["bias"] + list(STATE_FEATURE_NAMES),
        "delivery_weights": delivery_weights.tolist(),
        "ctr_weights": ctr_weights.tolist(),
        "mae_delivery": mae_delivery,
        "mae_ctr": mae_ctr,
        "r2_delivery": r2_delivery,
        "r2_ctr": r2_ctr,
        "n_training_rows": len(X),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Wrote model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
