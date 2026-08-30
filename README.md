# RTB Campaign Delivery Optimizer

**Status: in progress (Phase 0 — data pipeline).** This README will be rewritten to lead with real results (calibration plots, strategy comparison table, warm-start results, live demo link) once they exist. Right now it documents the plan.

## Problem

A campaign must serve **X impressions** in a target geo within a time window while keeping observed **click-through rate ≥ Y**, at **minimum total spend**. This is framed as a sequential, resource-constrained decision problem — a **constrained contextual bandit** (Bandits-with-Knapsacks style: Badanidiyuru, Kleinberg & Slivkins) — not a plain classification task, using real bid/impression/click logs from an actual demand-side platform (DSP).

## Why this project

Built to demonstrate the same class of real-time-bidding analysis done professionally (in a confidential, unpublishable context) on public data instead: the [iPinYou RTB dataset](http://contest.ipinyou.com/) — real bidding, impression, click, and conversion logs released after iPinYou's 2013 Global RTB Bidding Algorithm Competition. See `DATA_LICENSE.md` for citation/license terms and `data/README.md` for the full schema and acquisition steps.

## Approach

1. **Market simulator**: a click-propensity model (LightGBM) and a win-rate/market-price model fit from historical bids — validated against real logged win/loss outcomes (every bid's outcome is derivable by joining Bid ID between the bidding log and the impression log).
2. **Naive baseline**: constant-bid / flat pacing.
3. **Constrained contextual bandit**: minimizes spend subject to a delivery target and a CTR floor (gated on a lower-confidence-bound estimate, given iPinYou's ~0.07-0.1% base CTR).
4. **Warm start**: a prior built from other campaigns' historical price/CTR distributions, applied to a held-out "new" campaign to reduce exploration cost.
5. **Demo**: a Streamlit app comparing the three strategies interactively.

The auction mechanism is confirmed second-price (Vickrey/GSP) directly from iPinYou's own dataset paper — and iPinYou's own official offline-evaluation methodology for the original competition is, in fact, the same bid-replay logic used here for simulation (see `data/README.md`).

An **offline-RL (CQL/IQL) version is a planned v2**, not attempted in v1 — deferred deliberately to avoid an unfinished flagship feature blocking release of a working v1. See the project plan for details.

## Repo structure

```
config/            campaign & scenario configuration
data/              raw (gitignored) / interim (gitignored) / processed (small, committed)
scripts/           data download & preprocessing
src/adtech_rtb/    core library (simulator, bandit, evaluation)
notebooks/         exploration & results notebooks
app/               Streamlit demo
tests/             pytest suite
reports/figures/   result figures referenced by this README
```

## Reproduce it

```
pip install -r requirements.txt
python scripts/download_data.py   # see data/README.md if this needs manual steps
python scripts/make_dataset.py
```

## Limitations (will expand as the project develops)

- The bidding log reflects this DSP's own observed bid volume, not the full unconstrained exchange auction population.
- The market simulator is a model fit from a 2013 historical market — not live, not current, and doesn't capture live competitor adaptation.
- The CTR-floor constraint's estimation noise (given the dataset's very low base CTR) is handled via a lower-confidence-bound gate, not a point estimate — documented in the methodology section once written.
