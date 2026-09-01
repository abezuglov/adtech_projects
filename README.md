# RTB Campaign Delivery Optimizer

**Status: Phase 5 (synthetic first-price environment) built and validated.**

## Problem

A campaign must serve **X impressions** within a time window while keeping observed **click-through rate ≥ Y**, at **minimum total spend**. Framed as a sequential, resource-constrained decision problem — a **constrained contextual bandit** (Bandits-with-Knapsacks style: Badanidiyuru, Kleinberg & Slivkins), not a plain classification task.

## Why a synthetic auction environment, not real historical data

This project originally ran entirely against the real [iPinYou RTB dataset](http://contest.ipinyou.com/) (2013 Global RTB Bidding Algorithm Competition logs) — a fitted market simulator (LightGBM win-rate/price/CTR models), a naive flat-bid baseline, the bandit, and a cross-campaign warm-start. That work is preserved in full on the **[`archive/real-data-gsp`](../../tree/archive/real-data-gsp) branch**.

It surfaced a real finding, not a bug: iPinYou's auctions are GSP (second-price) — you pay the second-highest *competing* bid, never your own — so `paying_price` is mechanically independent of your own bid. Once bidding is worth it at all, bidding the maximum allowed level is provably optimal (higher bid never costs more, only wins more). There is no genuine bid-*level* decision under GSP, only bid/skip — confirmed empirically (bids concentrated ~99.9% at the top level on the largest real scenario), and it capped the bandit's edge over a naive flat bid at roughly 2%.

This repo now runs against a **synthetic, hierarchical, first-price environment** instead, purpose-built so bid level is a genuine cost/quality trade-off ("bid shading" — a real, well-known adtech problem since most exchanges moved to first-price around 2019). Result: the bandit beats the naive baseline on **12/12** generated scenarios, by **5.1–15.1% CPM** (avg **11.2%**), an **11.9%** total spend reduction at matched delivery.

## Approach

1. **World generator** (`synthetic_world.py`): a hierarchical PyMC model's *prior* (no inference — pure forward sampling) draws one frozen realization of 10,000 placements' economics (clearing price, price-sensitivity, baseline CTR, right-skewed traffic volume) plus the population-level rule governing how campaigns' CTR affinity varies. Campaigns are not a fixed roster — any campaign id is valid, sampled on the fly, with its own affinity vector generated lazily and deterministically.
2. **Environment** (`synthetic.py`): first-price auctions (pay exactly your bid on a win) resolved against the frozen world.
3. **Naive baseline**: closed-form population expectation for the best constant bid.
4. **Constrained contextual bandit** (`bandit.py`/`pacing.py`): hand-rolled online Bayesian linear/logistic models (win-rate, CTR) updated batch-by-batch, Thompson sampling over discretized bid levels, Lagrangian dual-variable pacing against the delivery target and CTR floor. The same `BanditPolicy` class also drives the archived real-data (GSP) pipeline — schema and auction-mechanism (`first_price` flag) are both passed in by the caller, not hardcoded.

Not yet built: a synthetic warm-start comparison (transferring the win-rate prior across campaigns should help; transferring CTR shouldn't, since each campaign's affinity is an independent draw — see `synthetic_world.py`'s docstring), and a demo UI.

## Repo structure

```
config/synthetic_scenarios.yaml    generated delivery scenarios
scripts/                            generate_synthetic_world.py -> generate_synthetic_scenarios.py
                                     -> run_synthetic_naive_baseline.py -> run_synthetic_bandit.py
src/adtech_rtb/
  synthetic_world.py                 PyMC hierarchical world generator
  synthetic.py                       first-price environment + naive-baseline solver + scenario generator
  bandit.py                          online Bayesian bandit policy (shared with the archived real-data pipeline)
  pacing.py                          Lagrangian dual-variable pacing controller (shared)
reports/synthetic_*.json            latest run results
app/                                Streamlit demo (not yet built)
```

## Reproduce it

```
pip install -r requirements.txt
python scripts/generate_synthetic_world.py
python scripts/generate_synthetic_scenarios.py
python scripts/run_synthetic_naive_baseline.py
python scripts/run_synthetic_bandit.py
```

## Limitations

- Single-seed realization per scenario, not averaged over multiple seeds — a documented simplification, not a rigor claim.
- The bandit's dual-variable cap (how much it's willing to pay above intrinsic value to protect delivery) is a deliberate, empirically-tuned trade-off: it's calibrated to preserve real price sensitivity, which means a handful of delivery-tight scenarios need a generous overrun allowance (up to ~3.3x nominal flight length) to fully complete rather than bidding indiscriminately to finish on time.
- Warm-start and a live demo are planned, not yet built (see Approach above).
- The real iPinYou/GSP exploration this project started from — including the root-caused finding about GSP's bid-independent pricing — is preserved on the `archive/real-data-gsp` branch, not deleted, and is itself a legitimate result worth reading.
