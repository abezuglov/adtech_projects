# RTB Campaign Delivery Optimizer

**Status: Phase 5 (synthetic first-price environment) built and validated, including a learned pacing controller and a live Streamlit demo.**

## Problem

A campaign must serve **X impressions** within a time window while keeping observed **click-through rate ≥ Y**, at **minimum total spend**. Framed as a sequential, resource-constrained decision problem — a **constrained contextual bandit** (Bandits-with-Knapsacks style: Badanidiyuru, Kleinberg & Slivkins), not a plain classification task.

## Why a synthetic auction environment, not real historical data

This project originally ran entirely against the real [iPinYou RTB dataset](http://contest.ipinyou.com/) (2013 Global RTB Bidding Algorithm Competition logs) — a fitted market simulator (LightGBM win-rate/price/CTR models), a naive flat-bid baseline, the bandit, and a cross-campaign warm-start. That work is preserved in full on the **[`archive/real-data-gsp`](../../tree/archive/real-data-gsp) branch**.

It surfaced a real finding, not a bug: iPinYou's auctions are GSP (second-price) — you pay the second-highest *competing* bid, never your own — so `paying_price` is mechanically independent of your own bid. Once bidding is worth it at all, bidding the maximum allowed level is provably optimal (higher bid never costs more, only wins more). There is no genuine bid-*level* decision under GSP, only bid/skip — confirmed empirically (bids concentrated ~99.9% at the top level on the largest real scenario), and it capped the bandit's edge over a naive flat bid at roughly 2%.

This repo now runs against a **synthetic, hierarchical, first-price environment** instead, purpose-built so bid level is a genuine cost/quality trade-off ("bid shading" — a real, well-known adtech problem since most exchanges moved to first-price around 2019).

## The power of context

Naive isn't a strawman. It's the textbook-optimal answer to "what's the best single flat bid for this whole campaign?" — computed with perfect knowledge of the world's true win-rate and CTR functions (an oracle, not a fitted estimate), zero learning cost, zero estimation noise. No strategy that also had to name one constant bid for the entire flight could beat it; by construction, nothing can.

The bandit is worse off in every other respect. It starts knowing nothing and learns win-rate/CTR online from real, noisy auction outcomes — real cold-start cost, real exploration cost. In the featured config it's also confined to 6 discrete bid levels, where naive gets to search a continuous range. Its one advantage is that it doesn't have to commit to a single number: it can price *by context* — placement, time of flight — instead of one flat bid for everyone. That's the entire "bid shading" idea this project is built around.

That one advantage turns out to be enough. With a learned pacing controller (see "Learned pacing," below), the bandit reaches **parity with naive on average CPM** (-0.1%) and **beats this population-optimal, oracle-informed baseline outright on 11 of 20 evaluation scenarios** — not by knowing more, but by acting differently depending on where it is. It hits its delivery target on **20/20** scenarios and the CTR floor on **18/20** (two near-misses, 1.0% and 1.3% below floor, on the same pair of edge-case scenarios discussed under Limitations).

Two honest caveats keep this from being a bigger claim than it is. First, this is a smaller, more hedged result than an earlier draft of this README made ("beats naive on 12/12, avg 11.2% CPM") — that number came from comparing the bandit against naive's own *unconstrained* continuous bid (an advantage on top of the oracle knowledge, not instead of it) and predates a pacing retune that traded away some cost efficiency to fix real delivery-reliability failures; the corrected comparison is less flattering but actually validated. Second, against that harder, fully-unconstrained oracle baseline, neither controller wins on average yet (analytic -12.5%, learned -4.0%, see the table below) — a legitimate stretch goal, not something to claim today.

## Approach

1. **World generator** (`synthetic_world.py`): a hierarchical PyMC model's *prior* (no inference — pure forward sampling) draws one frozen realization of 10,000 placements' economics (clearing price, price-sensitivity, baseline CTR, right-skewed traffic volume) plus the population-level rule governing how campaigns' CTR affinity varies. Campaigns are not a fixed roster — any campaign id is valid, sampled on the fly, with its own affinity vector generated lazily and deterministically.
2. **Environment** (`synthetic.py`): first-price auctions (pay exactly your bid on a win) resolved against the frozen world.
3. **Naive baseline**: closed-form, oracle population expectation for the best *constant* bid — see "The power of context," above, for why this is a genuinely strong baseline, not a strawman. Solved over the *same* discrete bid grid the bandit uses by default (`solve_delivery_bid_synthetic`'s `bid_levels` option), not its own unconstrained continuous bid — the fair comparison point.
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
