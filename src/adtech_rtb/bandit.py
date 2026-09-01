"""Cold-start constrained contextual bandit: online belief models + a
Lagrangian/Thompson-sampling policy over discretized bid levels.

The bandit never touches the fitted market_model.py/ctr_model.py boosters
-- those stay in the environment (simulator.py), generating realized
outcomes. Everything here learns from scratch, batch by batch, purely from
(context, chosen bid, observed outcome) tuples the policy itself produced.
That's what makes "exploration cost" and cold-start-vs-warm-start
meaningful; giving the bandit oracle access to the fitted models would
collapse both concepts (confirmed with the user before building this).

Three online learners share one small framework (OnlineBayesianLinearModel,
a diagonal Laplace/online-Newton approximation -- hand-rolled rather than
an off-the-shelf bandit library, per the project's original plan doc):
  - win-rate: hashed context + bidding_price, logistic link
  - CTR: hashed context only, logistic link, won rows only
  - price: hashed context only (bid-independent, mirrors market_model.py's
    own price regression), identity link on log1p(price)
A per-bucket running average was considered for price instead of a third
learner, but rejected: exact-context buckets are sparse early in any
flight, the same generalization problem hashed features exist to avoid
for the other two learners.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import bidding
from .features import ALL_CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .market_model import MARKET_CATEGORICAL_COLUMNS

N_HASH_BUCKETS = 2**14
BIAS_INDEX = 0
BID_PRICE_INDEX = 1
_RESERVED = 2  # indices [0, _RESERVED) are bias/bid-price, hashed features live in [_RESERVED, N_HASH_BUCKETS)

BID_LEVELS = np.linspace(bidding.DEFAULT_BID_BOUNDS[0], bidding.DEFAULT_BID_BOUNDS[1], 6)
_BID_NORM = 300.0  # rough centering scale for the bidding_price feature, same order of magnitude as the price itself

# Ad hoc normalization scales so numeric features sit on comparable orders
# of magnitude to the one-hot categorical entries (which are always 1.0) --
# not fit from data, just fixed constants appropriate to this dataset's
# known ranges (see features.py's NUMERIC_COLUMNS).
_NUMERIC_SCALES = {
    "hour": 24.0,
    "weekday": 7.0,
    "ad_slot_width": 1000.0,
    "ad_slot_height": 1000.0,
    "slot_area": 100_000.0,
    "log_floor_price": 6.0,
}


def _hash_index(key: str, n_buckets: int) -> int:
    return _RESERVED + zlib.crc32(key.encode()) % (n_buckets - _RESERVED)


def hash_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    n_buckets: int = N_HASH_BUCKETS,
    bid_prices: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Sparse hashing-trick feature matrix, shape (len(df), n_buckets).

    `df` must already be feature-engineered (features.build_features) --
    categorical columns still raw strings, numeric columns real-valued.
    `bid_prices`, if given, is written to the fixed BID_PRICE_INDEX (not
    hashed) so bandit.py can read/clip that one coefficient directly for
    the GSP monotonicity prior -- collision-prone hashed lookup would make
    that unreliable.
    """
    n_rows = len(df)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    rows.extend(range(n_rows))
    cols.extend([BIAS_INDEX] * n_rows)
    data.extend([1.0] * n_rows)

    if bid_prices is not None:
        rows.extend(range(n_rows))
        cols.extend([BID_PRICE_INDEX] * n_rows)
        data.extend((np.asarray(bid_prices, dtype=float) / _BID_NORM).tolist())

    for col in categorical_columns:
        values = df[col].astype(str).to_numpy()
        for i, v in enumerate(values):
            rows.append(i)
            cols.append(_hash_index(f"{col}={v}", n_buckets))
            data.append(1.0)

    for col in numeric_columns:
        idx = _hash_index(col, n_buckets)
        scale = _NUMERIC_SCALES.get(col, 1.0)
        # Centered (roughly [-0.5, 0.5]), not just scaled: an always-present,
        # always-positive numeric feature is confounded with the always-on
        # bias term under OnlineBayesianLinearModel's diagonal Hessian
        # approximation -- verified empirically (a synthetic case with an
        # uncentered dense feature needed >10x the IRLS iterations to reach
        # the same accuracy centering gets for free).
        values = df[col].to_numpy(dtype=float) / scale - 0.5
        rows.extend(range(n_rows))
        cols.extend([idx] * n_rows)
        data.extend(values.tolist())

    return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_buckets))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class OnlineBayesianLinearModel:
    """Diagonal Laplace-approximated online GLM: mean weight `mu`, per-
    feature precision `q` (inverse variance). One Newton step per batch,
    treating the previous batch's posterior as the next batch's prior
    (assumed-density-filtering style recursive update) -- not claiming to
    reproduce any specific paper's exact formulas, just a standard,
    correct online Laplace update for a diagonal Gaussian posterior.

    `link="logistic"` for win-rate/CTR (binary outcomes); `link="identity"`
    for the price regression (Gaussian working-weight of 1.0, an arbitrary
    but consistent noise-scale choice -- fine since price is only ever
    used here for its point estimate, never an LCB).
    """

    def __init__(self, n_features: int, link: str, prior_precision: float = 10.0, damping: float = 0.5):
        """`prior_precision=10.0` (not the textbook-default 1.0) and
        `damping=0.5` (a partial, not full, Newton step) both exist to fix
        a real divergence found while validating this against the actual
        simulator: a row with ~12 simultaneously-active sparse features
        (bias + several categorical hashes + numeric) lets each dimension
        independently claim credit for the *entire* residual under a
        diagonal Hessian approximation, so with `prior_precision=1.0,
        damping=1.0` the price learner's bias term diverged to ~1e26 within
        20 batches (confirmed on a synthetic reproduction matching the
        real feature-density pattern) -- `expm1` overflow followed
        immediately downstream in choose_bids. Both changes independently
        fixed it in that reproduction; using both together converged
        closest to the true value.
        """
        self.link = link
        self.mu = np.zeros(n_features)
        self.q = np.full(n_features, prior_precision)
        self.damping = damping

    def _mean(self, eta: np.ndarray) -> np.ndarray:
        return _sigmoid(eta) if self.link == "logistic" else eta

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self._mean(X @ self.mu)

    def variance(self, X: sp.csr_matrix) -> np.ndarray:
        """Predictive variance of the linear predictor `eta = X . w`."""
        return X.multiply(X) @ (1.0 / self.q)

    def sample_predict(self, X: sp.csr_matrix, rng: np.random.Generator) -> np.ndarray:
        eta_mean = X @ self.mu
        eta_var = self.variance(X)
        eta_sample = eta_mean + rng.standard_normal(len(eta_mean)) * np.sqrt(np.maximum(eta_var, 0.0))
        return self._mean(eta_sample)

    def update(self, X: sp.csr_matrix, y: np.ndarray, n_iter: int = 5) -> None:
        """A single linearized Newton step (one pass, `n_iter=1`) converges
        poorly when features are correlated -- e.g. the bias term and any
        always-on numeric feature (hour, slot dims, ...) both touch every
        row, and a one-shot diagonal update can't disentangle them well
        (verified empirically: >0.5 mean-abs prediction error on a
        synthetic case with that exact correlation pattern). Several IRLS
        iterations *within* the batch, holding the batch's incoming (mu, q)
        fixed as the Gaussian prior throughout, converges to that batch's
        local MAP properly before moving on -- this is what actually fixes
        it (confirmed: mean-abs error dropped by >10x after adding this).
        """
        if X.shape[0] == 0:
            return
        mu0 = self.mu
        q0 = self.q
        mu = mu0.copy()
        hess = q0.copy()
        X2 = X.multiply(X)
        for _ in range(n_iter):
            eta = X @ mu
            p = self._mean(eta)
            residual = y - p
            working_weight = p * (1.0 - p) if self.link == "logistic" else np.ones_like(p)

            grad = X.T @ residual - q0 * (mu - mu0)
            hess = X2.T @ working_weight + q0
            mu = mu + self.damping * grad / hess
            # Hard safety clamp, checked every inner iteration (not just at
            # the end): the divergence above is a *known* failure mode of a
            # diagonal Hessian approximation when many features are
            # perfectly correlated within one batch (e.g. a geo-constrained
            # scenario where nearly every row shares the same region/city/
            # bias features) -- confirmed on real scenario data, where a
            # single 649-row batch of same-geo wins drove one coefficient
            # to ~1831 in one update call. +-20 is generous headroom over
            # any real coefficient this problem needs (log-price maxes
            # around log1p(1000)=6.9; realistic per-feature logit
            # contributions are single digits) while definitively
            # preventing the runaway from compounding across batches.
            mu = np.clip(mu, -20.0, 20.0)

        self.mu = mu
        self.q = hess

    def clip_bid_price_coef(self, min_value: float = 0.0) -> None:
        """GSP monotonicity prior: P(win) is mechanically non-decreasing in
        your own bid (see market_model.py's docstring for the identical
        argument applied to the fitted offline model) -- the online learner
        sees just as little real price variation, so it needs the same
        nudge, applied here instead of a training-time constraint.
        """
        self.mu[BID_PRICE_INDEX] = max(self.mu[BID_PRICE_INDEX], min_value)


class BanditPolicy:
    """Cold-start policy: three OnlineBayesianLinearModel learners plus the
    pacing dual variables from pacing.py. State (delivered/spend/clicks) is
    owned and updated by simulator.py, passed in read-only where needed.

    `market_categorical_columns`/`ctr_categorical_columns`/`numeric_columns`
    default to the real iPinYou schema (features.py's column lists) so
    existing real-data callers (run_bandit.py, run_warm_start.py) need no
    changes -- but are overridable so the same class also works over the
    Phase 5 synthetic environment's much smaller schema (synthetic.py),
    without duplicating this whole class just to swap a handful of column
    names. `hash_features` was already schema-agnostic at the function
    level; only this class's own wrappers used to hardcode the real lists.
    """

    def __init__(
        self,
        ctr_floor: float,
        seed: int = 0,
        first_price: bool = False,
        market_categorical_columns: list[str] | None = None,
        ctr_categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        # First-price mode (Phase 5 synthetic environment): you pay exactly
        # what you bid on a win, so cost genuinely scales with bid level --
        # unlike GSP (this project's real-data mode), where paying_price is
        # mechanically independent of your own bid. See choose_bids/observe
        # below for how this flag changes the decision and update logic.
        self.first_price = first_price
        self._market_categorical_columns = (
            market_categorical_columns if market_categorical_columns is not None else MARKET_CATEGORICAL_COLUMNS
        )
        self._ctr_categorical_columns = (
            ctr_categorical_columns if ctr_categorical_columns is not None else ALL_CATEGORICAL_COLUMNS
        )
        self._numeric_columns = numeric_columns if numeric_columns is not None else NUMERIC_COLUMNS
        self.win_rate_model = OnlineBayesianLinearModel(N_HASH_BUCKETS, link="logistic")
        self.ctr_model = OnlineBayesianLinearModel(N_HASH_BUCKETS, link="logistic")
        # Stronger regularization/damping than the shared defaults: the
        # identity link has no natural saturation the way logistic does via
        # sigmoid (which bounds _mean(eta) to (0,1) regardless of how large
        # eta gets), so it's uniquely exposed to the same-batch-many-
        # correlated-dimensions overshoot pathology. Confirmed on real data
        # (a narrow-geo scenario, 3427-3): even with the shared defaults'
        # damping=0.5/prior_precision=10/mu-clamp=+-20, the price
        # prediction still swung from ~250 to 0 to the output clip's
        # ceiling (~2980) within two consecutive batches -- individually
        # clamped coefficients still overshoot in aggregate when ~15 of
        # them move the same direction at once for a batch of near-
        # identical rows. Win-rate/CTR don't get this treatment because
        # sigmoid's saturation already protects them.
        self.price_model = OnlineBayesianLinearModel(N_HASH_BUCKETS, link="identity", prior_precision=50.0, damping=0.2)
        # A zero-initialized price prior makes every auction look *free*
        # before any price has been observed -- confirmed as the actual
        # trigger for a real divergence: lambda_delivery ticking barely
        # positive was enough to make the policy bid on an entire batch at
        # once (2000/2000 rows) purely because the cost term was falsely
        # zero. A generic "prices are roughly RMB/CPM scale" prior (not
        # anything from the fitted models -- just the currency/unit
        # knowledge any real DSP integration would already have) avoids
        # that without giving away information the cold-start bandit isn't
        # supposed to have.
        self.price_model.mu[BIAS_INDEX] = np.log1p(250.0)
        self.ctr_floor = ctr_floor
        self.lambda_delivery = 0.0
        self.lambda_ctr = 0.0

    def _market_features(self, contexts: pd.DataFrame, bid_prices: np.ndarray | None = None) -> sp.csr_matrix:
        return hash_features(contexts, self._market_categorical_columns, self._numeric_columns, bid_prices=bid_prices)

    def _ctr_features(self, contexts: pd.DataFrame) -> sp.csr_matrix:
        return hash_features(contexts, self._ctr_categorical_columns, self._numeric_columns)

    def choose_bids(self, contexts: pd.DataFrame) -> np.ndarray:
        """One discretized bid level (or 0.0 = skip) per row in `contexts`,
        chosen as the argmax over levels (skip = 0.0 an explicit competing
        option) of the level's expected net value:

            adjusted_value(level) = P(win|level) * (lambda_delivery +
                lambda_ctr*(ctr - floor) - cost(level)/1000)

        `cost(level)` is `price_sample` (this class's own learned price
        model, level-*independent*) in GSP mode (`first_price=False`, this
        project's real-data mode -- mirrors market_model.py's own models,
        neither of which takes bidding_price as a feature, since under GSP
        what you pay if you win doesn't depend on your own bid at all), or
        `level` itself in first-price mode (Phase 5's synthetic
        environment -- you pay exactly what you bid on a win, so cost
        genuinely scales with level; no price model needed for the
        decision since cost is then deterministic given your own action).

        In GSP mode this reduces exactly to the simpler skip/bid-then-
        maximize-win-probability rule an earlier version of this method
        used directly: cost(level) doesn't vary with level there, so
        `argmax_level P(win|level) * net_value_per_win` (net_value_per_win
        a level-independent constant) is the same level as
        `argmax_level P(win|level)` whenever that constant is positive, and
        correctly falls back to skip (0) whenever it's <=0 -- so this is a
        strict generalization, not a behavior change, for real-data mode.
        (All terms denominated in RMB per impression -- see pacing.py's
        LAMBDA_DELIVERY_MAX/LAMBDA_CTR_MAX calibration notes; an earlier
        version mixed scales here and delivery urgency silently swamped
        price/CTR on every auction past the first batch.)
        """
        n = len(contexts)
        # Zero at BID_PRICE_INDEX -- shared by the win-rate model's base (pre
        # per-level adjustment below) and the price/CTR models, which don't
        # use that feature at all (bid-independent, mirrors market_model.py).
        X_market_base = self._market_features(contexts)
        base_eta_mean = X_market_base @ self.win_rate_model.mu
        base_eta_var = self.win_rate_model.variance(X_market_base)

        if not self.first_price:
            # Clipped to [0, 8] before expm1 -- a log-price ceiling of
            # expm1(8) ~= 2980 RMB/CPM, ~10x the realistic max (see
            # market_model.py's ~227-300 observed range), generous headroom
            # but a hard guard against overflow if several clipped-but-
            # still-large coefficients (see OnlineBayesianLinearModel.
            # update's +-20 clamp) sum on one row.
            price_sample = np.expm1(np.clip(self.price_model.predict(X_market_base), 0.0, 8.0))

        X_ctr = self._ctr_features(contexts)
        ctr_sample = self.ctr_model.sample_predict(X_ctr, self.rng)

        # Relative CTR gap (matches pacing.update_ctr_lambda's own scale
        # fix) -- an absolute gap left lambda_ctr's contribution negligible
        # regardless of eta, since CTR values here are ~1e-4 scale.
        ctr_gap = (ctr_sample - self.ctr_floor) / self.ctr_floor
        base_value = self.lambda_delivery + self.lambda_ctr * ctr_gap  # level-independent part

        best_adjusted_value = np.zeros(n)  # skip = 0.0, an explicit competing option
        best_level = np.zeros(n)
        for level in BID_LEVELS:
            bid_norm = level / _BID_NORM
            eta_mean = base_eta_mean + bid_norm * self.win_rate_model.mu[BID_PRICE_INDEX]
            eta_var = base_eta_var + (bid_norm**2) / self.win_rate_model.q[BID_PRICE_INDEX]
            eta_sample = eta_mean + self.rng.standard_normal(n) * np.sqrt(np.maximum(eta_var, 0.0))
            win_prob_sample = _sigmoid(eta_sample)

            cost = level if self.first_price else price_sample
            adjusted_value = win_prob_sample * (base_value - cost / 1000.0)

            better = adjusted_value > best_adjusted_value
            best_adjusted_value = np.where(better, adjusted_value, best_adjusted_value)
            best_level = np.where(better, level, best_level)

        return best_level

    def observe(self, contexts: pd.DataFrame, chosen_bids: np.ndarray, won: np.ndarray, clicked: np.ndarray, price_paid: np.ndarray) -> None:
        """Update the three online learners from realized outcomes for one
        processed batch. Called by simulator.py after the environment
        resolves the true (fitted-model-generated) outcomes -- the bandit
        only ever sees what it would realistically observe: whether it won,
        whether a won impression was clicked, and what it paid if it won.
        """
        bid_mask = chosen_bids > 0
        if bid_mask.any():
            X_market = self._market_features(contexts.loc[bid_mask], bid_prices=chosen_bids[bid_mask])
            self.win_rate_model.update(X_market, won[bid_mask].astype(float))
            self.win_rate_model.clip_bid_price_coef()

        won_mask = bid_mask & won.astype(bool)
        if won_mask.any():
            X_ctr = self._ctr_features(contexts.loc[won_mask])
            self.ctr_model.update(X_ctr, clicked[won_mask].astype(float))

            if not self.first_price:
                # Nothing to learn under first-price: cost is exactly the
                # bid the policy itself chose, not an unknown to estimate.
                X_price = self._market_features(contexts.loc[won_mask])
                self.price_model.update(X_price, np.log1p(price_paid[won_mask]))
