# RTB Campaign Delivery Optimizer

**Status: Phase 5 (synthetic first-price environment) built and validated, including a learned pacing controller and a live Streamlit demo.**

## Problem

A campaign must serve **X impressions** within a time window while keeping observed **click-through rate ≥ Y**, at **minimum total spend**. Framed as a sequential, resource-constrained decision problem — a **constrained contextual bandit** (Bandits-with-Knapsacks style: Badanidiyuru, Kleinberg & Slivkins), not a plain classification task.

## Why a synthetic auction environment, not real historical data

This project originally ran entirely against the real [iPinYou RTB dataset](http://contest.ipinyou.com/) (2013 Global RTB Bidding Algorithm Competition logs) — a fitted market simulator (LightGBM win-rate/price/CTR models), a naive flat-bid baseline, the bandit, and a cross-campaign warm-start. That work is preserved in full on the **[`archive/real-data-gsp`](../../tree/archive/real-data-gsp) branch**.

It surfaced a real finding, not a bug: iPinYou's auctions are GSP (second-price) — you pay the second-highest *competing* bid, never your own — so `paying_price` is mechanically independent of your own bid. Once bidding is worth it at all, bidding the maximum allowed level is provably optimal (higher bid never costs more, only wins more). There is no genuine bid-*level* decision under GSP, only bid/skip — confirmed empirically (bids concentrated ~99.9% at the top level on the largest real scenario), and it capped the bandit's edge over a naive flat bid at roughly 2%.

This repo now runs against a **synthetic, hierarchical, first-price environment** instead, purpose-built so bid level is a genuine cost/quality trade-off ("bid shading" — a real, well-known adtech problem since most exchanges moved to first-price around 2019).

**Headline result:** a bandit using a *learned* pacing controller (see "Learned pacing," below) lands at roughly **parity with a fair naive baseline on average CPM** (-0.1%), while actually **beating it on 11 of 20 evaluation scenarios** — especially the delivery-tight ones, where naive's flat bid has no way to adapt. It hits its delivery target on **20/20** scenarios and the CTR floor on **18/20** (two near-misses, 1.0% and 1.3% below floor, both on the same pair of edge-case scenarios discussed under Limitations). This is a smaller, more hedged claim than an earlier draft of this README made (previously "beats naive on 12/12, avg 11.2% CPM") — that number came from comparing the bandit against a naive baseline with an unfair advantage (see below) and predates a pacing retune that traded some cost efficiency for fixing real delivery-reliability failures. The corrected comparison is less flattering but actually validated, not asserted.

## Approach

1. **World generator** (`synthetic_world.py`): a hierarchical PyMC model's *prior* (no inference — pure forward sampling) draws one frozen realization of 10,000 placements' economics (clearing price, price-sensitivity, baseline CTR, right-skewed traffic volume) plus the population-level rule governing how campaigns' CTR affinity varies. Campaigns are not a fixed roster — any campaign id is valid, sampled on the fly, with its own affinity vector generated lazily and deterministically.
2. **Environment** (`synthetic.py`): first-price auctions (pay exactly your bid on a win) resolved against the frozen world.
3. **Naive baseline**: closed-form population expectation for the best *constant* bid — one flat number for the entire flight, not adapted to placement, hour, or anything else. Computed from the world's own true win-rate/CTR functions directly (an oracle, not an estimate), and — as of this comparison — solved over the *same* discrete bid grid the bandit uses (`solve_delivery_bid_synthetic`'s `bid_levels` option), not its own unconstrained continuous bid. That distinction matters: comparing the bandit against a continuous-bid naive was found to inflate the apparent CPM gap substantially (see "Learned pacing," below) — it wasn't a fair fight. The trade-off runs both ways, though: naive gets oracle knowledge and zero learning cost, but can never adapt bid to context; the bandit gets zero prior knowledge (real cold-start/exploration cost) but *can* price by placement and time, which is the entire "bid shading" story this project is built around.
4. **Constrained contextual bandit** (`bandit.py`/`pacing.py`): hand-rolled online Bayesian linear/logistic models (win-rate, CTR) updated batch-by-batch, Thompson sampling over discretized bid levels (6 by default; overridable via `bid_levels`), pacing dual variables (`lambda_delivery`/`lambda_ctr`) injecting delivery/CTR urgency into the bid decision. The same `BanditPolicy` class also drives the archived real-data (GSP) pipeline — schema and auction-mechanism (`first_price` flag) are both passed in by the caller, not hardcoded.
5. **Pacing controller, two variants** (see "Learned pacing," below): the original hand-tuned Lagrangian formula (`AnalyticPacingController`), and a **learned** controller (`LearnedPacingController`) fit via supervised regression against hindsight-optimal pacing behavior on simulated training flights. The bandit's win-rate/CTR/price models are untouched either way — only how pacing urgency is computed changes.
6. **Live demo** (`app/app.py`, Streamlit): replays an already-validated flight's trajectory batch-by-batch (doesn't run the simulator live — a real flight takes minutes). Featured config: learned pacing at 6 bid levels vs. the fair (grid-matched) naive baseline.

Not yet built: a synthetic warm-start comparison (transferring the win-rate prior across campaigns should help; transferring CTR shouldn't, since each campaign's affinity is an independent draw — see `synthetic_world.py`'s docstring).

## Learned pacing

The bandit's pacing formula was originally hand-tuned (a fixed convexity curve, step size, and cap on `lambda_delivery`/`lambda_ctr`) and needed re-deriving by hand every time the environment's cost scale changed — including a retune (`pacing.py`'s `PACE_CONVEXITY`/`eta` history) partway through this project, made to fix real unrecoverable-delivery-deficit failures on near-ceiling targets. That retune traded away CPM efficiency it hadn't been checked against: re-running the bandit-vs-naive comparison afterward showed the analytic (hand-tuned) controller no longer beats naive on CPM at all (see table below) — reliable, but no longer cost-competitive.

To recover that without hand-retuning again, `hindsight_pacing.py` runs a small grid search of the *same* hand-tuned constants against actual simulated flights, scores each with a composite loss (delivery shortfall, overrun, smoothness, CPM-vs-naive, CTR-floor violation), and relabels every candidate's trajectory — not just the winner's — with a per-step "return-to-go" score via `_sub_result_from_suffix` (free extra training data, no added simulation). `learned_pacing.py`'s `LearnedPacingController` is a linear map, fit once offline (`fit_learned_pacing.py`, plain least-squares) from pacing-state features to `(lambda_delivery, lambda_ctr)`, then frozen for eval — deliberately supervised regression, not RL, to avoid the training-stability problems and complexity of fitted-Q/offline-RL for this problem.

Three honest comparisons ended up mattering, not one:

| Comparison | Analytic (hand-tuned) | Learned |
|---|---|---|
| vs. naive's own continuous bid (unfair — naive gets bid precision no discretized policy has) | -12.5% avg CPM | -4.0% avg CPM |
| **vs. naive matched to the bandit's 6-level grid (fair)** | **-8.3% avg, 0/20 scenarios won** | **-0.1% avg, 11/20 scenarios won** |
| vs. naive matched to a 25-level grid (tested, not adopted) | -10.5% avg, 0/20 won | -1.1% avg, 4/20 won |

(Negative = bandit's CPM is higher than naive's, i.e. naive "wins" that comparison.) Two findings worth stating plainly: fixing the naive baseline's unfair bid precision closed most of the apparent gap on its own, before any pacing change; and giving the bandit *more* bid levels made both controllers slightly worse, not better — most likely the added cold-start exploration cost of 26 actions instead of 7 outweighing whatever precision gain finer bidding offered. 6 levels, matched fairly on both sides, is the featured result.

## Repo structure

```
config/synthetic_scenarios.yaml     generated delivery scenarios (random per-campaign)
config/synthetic_scenario_grid.yaml designed 2x2x2 scenario grid (duration x ctr x avails)
scripts/                            generate_synthetic_world.py -> generate_synthetic_scenarios.py
                                     -> run_synthetic_naive_baseline.py -> run_synthetic_bandit.py
                                     generate_hindsight_pacing_data.py -> fit_learned_pacing.py
                                     compare_pacing.py -- aggregates analytic vs. learned results
src/adtech_rtb/
  synthetic_world.py                 PyMC hierarchical world generator
  synthetic.py                       first-price environment + naive-baseline solver + scenario generator
  bandit.py                          online Bayesian bandit policy (shared with the archived real-data pipeline)
  pacing.py                          Lagrangian pacing primitives + AnalyticPacingController (hand-tuned)
  hindsight_pacing.py                hindsight-optimal pacing search + training-pair extraction
  learned_pacing.py                  LearnedPacingController (fit offline, frozen at eval)
reports/synthetic_*.json            latest run results (large trajectory dumps are gitignored, regenerable)
app/app.py                          Streamlit demo -- replays a validated flight's trajectory
```

## Reproduce it

```
pip install -r requirements.txt
python scripts/generate_synthetic_world.py
python scripts/generate_synthetic_scenarios.py

# Naive baseline, matched to the bandit's 6-level bid grid (the fair comparison)
python scripts/run_synthetic_naive_baseline.py --bid-levels 6
python scripts/run_naive_baseline_grid.py --bid-levels 6

# Analytic (hand-tuned) bandit
python scripts/run_synthetic_bandit.py --pacing analytic
python scripts/run_scenario_grid.py --pacing analytic

# Learned pacing: generate training data, fit, then evaluate (skip straight to
# eval if data/interim/learned_pacing_model.json already exists)
python scripts/generate_hindsight_pacing_data.py
python scripts/fit_learned_pacing.py
python scripts/run_synthetic_bandit.py --pacing learned
python scripts/run_scenario_grid.py --pacing learned

python scripts/compare_pacing.py   # aggregate analytic-vs-learned-vs-naive table

# Live demo (reads the already-produced result files above, doesn't re-simulate)
streamlit run app/app.py
```

## Limitations

- Single-seed realization per scenario, not averaged over multiple seeds — a documented simplification, not a rigor claim.
- Learned pacing is a frozen linear fit (`np.linalg.lstsq`, R² ≈ 0.13 for `lambda_delivery`, ≈ 0.04 for `lambda_ctr`) — it captures real signal (see the comparison table above) but leaves most of the variance unexplained; a richer function class was considered out of scope for this pass.
- Two scenarios (`grid-30d-highctr-easy`, `grid-90d-highctr-challenging`) sit right at the edge of what's deliverable at all: naive's own CTR-floor margin there is only 4.0% and 1.6% respectively (vs. 20–100%+ on every other scenario), and learned pacing's only two CTR-floor near-misses (1.0% and 1.3% below floor) land on exactly this pair. Flagged as a structural difficulty in the scenario itself, not a controller bug.
- Warm-start is planned, not yet built (see Approach above).
- The real iPinYou/GSP exploration this project started from — including the root-caused finding about GSP's bid-independent pricing — is preserved on the `archive/real-data-gsp` branch, not deleted, and is itself a legitimate result worth reading.
