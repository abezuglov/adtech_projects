"""Hierarchical PyMC generator for the synthetic, first-price RTB world.

Phase 3/4 (real iPinYou data, GSP pricing) showed the bandit's bid-*level*
choice has almost no real lever: under GSP, what you pay if you win doesn't
depend on your own bid at all, so bidding the max level within budget is
provably optimal whenever bidding at all is worth it (see bandit.py's
choose_bids docstring and the Phase 5 plan's Context section). This module
builds a different world instead: a first-price auction (simulator.py's
synthetic counterpart pays exactly your own bid on a win) with sigmoid
win-rate curves and campaign-dependent CTR, so bid level becomes a genuine
cost/quality trade-off ("bid shading" -- a real, well-known adtech problem).

A "world" is one FROZEN realization of every placement-level parameter this
environment needs, drawn once from a hierarchical PyMC model's *prior* (no
observed data -- this is a generative world-builder, not an inference task)
and reused, numpy-only, for the actual auction-level simulation loop. PyMC
generates the world's parameters (tens of thousands of values, instant to
draw); it is not used to sample individual auction outcomes, which run at a
scale (millions of Bernoulli draws per scenario) a PPL isn't built for.

Campaigns are deliberately NOT part of the PyMC world draw and are not a
fixed roster -- any campaign_id is valid, sampled/created on demand
(confirmed with the user). The world fixes only the population-level
hyperparameter governing how much campaigns vary from each other
(affinity_sigma); each individual campaign's own CTR-affinity-by-placement
vector is generated lazily, the first time that campaign_id is seen, by a
numpy RNG seeded deterministically from the campaign_id itself -- so repeat
lookups for the same campaign are cheap and, critically, identical (a
re-sampled vector would silently change a campaign's economics between
scenarios or between cold/warm runs).
"""

from __future__ import annotations

import dataclasses
import json
import zlib
from pathlib import Path

import numpy as np
import pymc as pm

N_PLACEMENTS = 10_000
WORLD_SEED = 20260830  # arbitrary, fixed -- the world is reproducible given this seed alone


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


@dataclasses.dataclass
class World:
    """One frozen draw from the hierarchical prior. All placement-level
    arrays have length N_PLACEMENTS; the rest are population-level scalars.
    """

    clearing_level: np.ndarray  # RMB/CPM where win prob crosses ~0.5, per placement
    beta: np.ndarray  # win-prob sensitivity to bid, per placement -- always > 0 (log-normal)
    ctr_base: np.ndarray  # baseline CTR, logit scale, per placement
    log_volume: np.ndarray  # log relative auction-request share, per placement
    affinity_sigma: float  # population-level: how much campaign CTR affinity varies
    diurnal_price_amp: float  # RMB-scale diurnal swing in clearing_level
    diurnal_ctr_amp: float  # logit-scale diurnal swing in CTR
    diurnal_phase: float  # shared timing (radians) -- price and CTR peak together

    @property
    def placement_weight(self) -> np.ndarray:
        """Right-skewed relative auction volume per placement, normalized
        to sum to 1. Softmax-style normalization (subtract max before exp)
        purely for numerical stability, not a modeling choice."""
        w = np.exp(self.log_volume - self.log_volume.max())
        return w / w.sum()

    def to_dict(self) -> dict:
        return {
            "clearing_level": self.clearing_level.tolist(),
            "beta": self.beta.tolist(),
            "ctr_base": self.ctr_base.tolist(),
            "log_volume": self.log_volume.tolist(),
            "affinity_sigma": self.affinity_sigma,
            "diurnal_price_amp": self.diurnal_price_amp,
            "diurnal_ctr_amp": self.diurnal_ctr_amp,
            "diurnal_phase": self.diurnal_phase,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        return cls(
            clearing_level=np.asarray(d["clearing_level"], dtype=float),
            beta=np.asarray(d["beta"], dtype=float),
            ctr_base=np.asarray(d["ctr_base"], dtype=float),
            log_volume=np.asarray(d["log_volume"], dtype=float),
            affinity_sigma=float(d["affinity_sigma"]),
            diurnal_price_amp=float(d["diurnal_price_amp"]),
            diurnal_ctr_amp=float(d["diurnal_ctr_amp"]),
            diurnal_phase=float(d["diurnal_phase"]),
        )


def build_world(seed: int = WORLD_SEED, n_placements: int = N_PLACEMENTS) -> World:
    """Draw one frozen World from the hierarchical prior. Pure forward
    sampling (pm.sample_prior_predictive) -- no MCMC/NUTS, no observed data,
    so this is instant even at n_placements=10_000."""
    with pm.Model():
        # Placement-level: clearing price (bid level where win prob ~ 0.5).
        clearing_mu = pm.Normal("clearing_mu", mu=260, sigma=15)
        clearing_sigma = pm.HalfNormal("clearing_sigma", sigma=15)
        pm.Normal("clearing_level", mu=clearing_mu, sigma=clearing_sigma, shape=n_placements)

        # Placement-level: price-sensitivity. Must be positive for win-prob
        # monotonicity in bid -- log-normal via exp(Normal), not a plain
        # Normal, guarantees beta > 0 by the prior's own support rather than
        # relying on a post-hoc monotone-constraint hack (the real LightGBM
        # model needed one; this generator doesn't).
        beta_log_mu = pm.Normal("beta_log_mu", mu=np.log(0.03), sigma=0.5)
        beta_log_sigma = pm.HalfNormal("beta_log_sigma", sigma=0.4)
        beta_z = pm.Normal("beta_z", 0, 1, shape=n_placements)
        pm.Deterministic("beta", pm.math.exp(beta_z * beta_log_sigma + beta_log_mu))

        # Placement-level: baseline CTR (logit scale).
        ctr_base_mu = pm.Normal("ctr_base_mu", mu=_logit(0.0008), sigma=0.5)
        ctr_base_sigma = pm.HalfNormal("ctr_base_sigma", sigma=0.5)
        pm.Normal("ctr_base", mu=ctr_base_mu, sigma=ctr_base_sigma, shape=n_placements)

        # Campaign-level: affinity hyperparameter only, NOT a fixed
        # (n_campaigns, n_placements) matrix -- campaigns are sampled on the
        # fly, unlimited in number (see module docstring). This is the one
        # population-level quantity the world fixes about campaigns; each
        # campaign's own affinity vector is drawn lazily elsewhere in this
        # module (campaign_affinity_vector), seeded by its campaign_id.
        pm.HalfNormal("affinity_sigma", sigma=0.7)

        # Placement-level: opportunity volume (relative auction-request
        # share). Right-skewed by construction -- LogNormal via a Normal on
        # the log scale, a standard choice for traffic/popularity
        # distributions -- so a handful of placements carry most of the
        # volume and a long tail carries very little, matching real
        # ad-exchange domain traffic and deliberately giving the bandit a
        # genuine long-tail exploration problem. TruncatedNormal (not
        # HalfNormal) around mu=2.2 -- a wide HalfNormal hyperprior was
        # tried first and rejected: its realized draw varied enormously
        # world-to-world (observed anywhere from ~0.8 to ~2.5 across a few
        # draws), sometimes landing in a barely-skewed regime (top 1% of
        # placements carrying only ~4-9% of volume) purely by chance of the
        # hyperprior draw, not by design. A tighter TruncatedNormal keeps
        # this a genuine hierarchical draw (still random, still continuous)
        # while reliably landing in the intended skew range (empirically,
        # mu=2.2 puts roughly 35-45% of total volume on the top 1% of
        # placements and <2% on the bottom half at n_placements=10_000)
        # regardless of which world seed is used.
        volume_log_sigma = pm.TruncatedNormal("volume_log_sigma", mu=2.2, sigma=0.3, lower=1.0)
        pm.Normal("log_volume", mu=0, sigma=volume_log_sigma, shape=n_placements)

        # Diurnal modifier: shared timing (phase) so price and CTR peak
        # together (a busier hour is plausibly both more competitive and
        # higher-CTR), but separate amplitudes since they live on different
        # scales (RMB vs. logit) -- a single shared amplitude would be a
        # unit mismatch (this project has hit that exact class of bug
        # before, in pacing.py's lambda_delivery/lambda_ctr calibration).
        pm.HalfNormal("diurnal_price_amp", sigma=10.0)
        pm.HalfNormal("diurnal_ctr_amp", sigma=0.3)
        pm.Uniform("diurnal_phase", lower=0, upper=2 * np.pi)

        prior = pm.sample_prior_predictive(samples=1, random_seed=seed)

    draw = prior.prior.isel(chain=0, draw=0)
    return World(
        clearing_level=draw["clearing_level"].to_numpy(),
        beta=draw["beta"].to_numpy(),
        ctr_base=draw["ctr_base"].to_numpy(),
        log_volume=draw["log_volume"].to_numpy(),
        affinity_sigma=float(draw["affinity_sigma"].to_numpy()),
        diurnal_price_amp=float(draw["diurnal_price_amp"].to_numpy()),
        diurnal_ctr_amp=float(draw["diurnal_ctr_amp"].to_numpy()),
        diurnal_phase=float(draw["diurnal_phase"].to_numpy()),
    )


def save_world(world: World, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(world.to_dict(), f)


def load_world(path: Path) -> World:
    with open(path) as f:
        return World.from_dict(json.load(f))


def diurnal_price_shift(world: World, hour: np.ndarray) -> np.ndarray:
    """RMB-scale shift to clearing_level, smooth 24h-periodic."""
    return world.diurnal_price_amp * np.sin(2 * np.pi * np.asarray(hour, dtype=float) / 24.0 + world.diurnal_phase)


def diurnal_ctr_shift(world: World, hour: np.ndarray) -> np.ndarray:
    """Logit-scale shift to CTR, smooth 24h-periodic, same phase as price."""
    return world.diurnal_ctr_amp * np.sin(2 * np.pi * np.asarray(hour, dtype=float) / 24.0 + world.diurnal_phase)


_campaign_affinity_cache: dict[str, np.ndarray] = {}


def campaign_affinity_vector(world: World, campaign_id, seed: int = WORLD_SEED) -> np.ndarray:
    """CTR-affinity-by-placement vector for one campaign, generated lazily
    and cached so repeat lookups for the same campaign_id are cheap and
    bit-identical (see module docstring -- this must not silently change
    between calls). Deterministic given (seed, campaign_id) alone, so it's
    reproducible across processes without needing to persist a matrix.
    """
    key = f"{seed}:{campaign_id}"
    cached = _campaign_affinity_cache.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    vec = rng.normal(0.0, world.affinity_sigma, size=len(world.ctr_base))
    _campaign_affinity_cache[key] = vec
    return vec
