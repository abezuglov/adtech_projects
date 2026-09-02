"""Written report page: reachable from the live demo's own Streamlit sidebar
nav (this file lives under app/pages/, auto-discovered as a second page) so
both live at the same host under one link once deployed -- see app.py's
module docstring for why the demo itself only replays precomputed results
rather than simulating live. This page reads the same already-validated
result files and is static narrative + charts, not an interactive replay --
the live demo already owns that job.

Featured flight: synth-synthC-2 (28-day nominal length, campaign "synthC",
from config/synthetic_scenarios.yaml's random per-campaign set -- not the
designed grid). Chosen because it's a clean illustration of the dual
variables actually doing real, visible work: lambda_delivery and lambda_ctr
climb together during cold-start, then diverge as the two constraints
resolve on different schedules. The random scenario set (unlike the
designed grid) has no human-authored description field -- this page's
narrative is effectively that description, for this one flight.

Section 4 ("Finding: a deadline-crossing pacing collapse", added 2026-09-02)
documents an investigation into a separate scenario
(grid-90d-highctr-challenging) where the learned pacing controller was found
to collapse both dual variables right at the deadline -- backwards, since
that's exactly when pressure should stay high -- because it had never seen a
single post-deadline training example. Its before-retrain trajectory numbers
are hardcoded (see that section's own comment) since the fix's re-evaluation
overwrote the broken trajectory on disk; see the repo README's "Learned
pacing" section for the full narrative and the retrain's effect across all
20 eval scenarios.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "config"

FEATURED_SCENARIO_ID = "synth-synthC-2"

st.set_page_config(page_title="RTB Bandit Report", layout="wide")


@st.cache_data
def load_comparison():
    with open(REPORTS_DIR / "pacing_comparison.json") as f:
        return json.load(f)


@st.cache_data
def load_featured_flight():
    with open(REPORTS_DIR / "synthetic_bandit_results.learned_pacing.json") as f:
        bandit_results = {r["scenario_id"]: r for r in json.load(f)}
    with open(REPORTS_DIR / "synthetic_naive_baseline_results.6levels.json") as f:
        naive_results = {r["scenario_id"]: r for r in json.load(f)}
    with open(CONFIG_DIR / "synthetic_scenarios.yaml") as f:
        scenarios = {s["id"]: s for s in yaml.safe_load(f)["scenarios"]}
    return bandit_results[FEATURED_SCENARIO_ID], naive_results[FEATURED_SCENARIO_ID], scenarios[FEATURED_SCENARIO_ID]


st.title("The power of context")
st.caption(
    "How a constrained contextual bandit, starting with zero prior knowledge and confined to 6 discrete "
    "bid levels, comes within a few percent of an oracle flat-bid baseline that gets perfect knowledge "
    "of the world's true win-rate and CTR functions plus an unlimited continuous bid."
)
st.page_link("app.py", label="Back to the live, scrubbable demo", icon="\U0001F3AC")

st.divider()

# ---------------------------------------------------------------------------
# Section 1: the headline result
# ---------------------------------------------------------------------------
st.header("Headline result")
st.markdown(
    """
The oracle baseline isn't a strawman: it's the textbook-optimal answer to *"what's the best single flat
bid for this whole campaign?"*, computed with perfect knowledge of the world's true win-rate and CTR
functions, zero learning cost, zero estimation noise. Nothing that also had to name one constant bid for
the entire flight can beat it, by construction.

The bandit is worse off in every other respect. It starts knowing nothing and learns win-rate and CTR
online from real, noisy auction outcomes — real cold-start cost, real exploration cost. It's also
confined to 6 discrete bid levels, where the oracle baseline gets to search the same 6-level grid (the
fair comparison — see Methodology below). Its one advantage: it doesn't have to commit to a single
number. It can price *by context* — placement, time of flight — instead of one flat bid for everyone.

That one advantage turns out to be enough to close nearly all of the average CPM gap, and to win outright
on 9 of the 20 evaluation scenarios — including every one of the hardest ("challenging") ones.
"""
)

comparison = load_comparison()
n = len(comparison)
vs_naive_a = [r["cpm_vs_naive_analytic_pct"] for r in comparison if r["cpm_vs_naive_analytic_pct"] is not None]
vs_naive_l = [r["cpm_vs_naive_learned_pct"] for r in comparison if r["cpm_vs_naive_learned_pct"] is not None]
avg_a, avg_l = sum(vs_naive_a) / len(vs_naive_a), sum(vs_naive_l) / len(vs_naive_l)
wins_a = sum(1 for v in vs_naive_a if v > 0)
wins_l = sum(1 for v in vs_naive_l if v > 0)
n_delivery_l = sum(r["delivery_met_learned"] for r in comparison)
n_ctr_l = sum(r["ctr_met_learned"] for r in comparison)

m = st.columns(4)
m[0].metric("Learned pacing vs. oracle baseline (avg CPM)", f"{avg_l:+.1f}%", "positive = bandit cheaper")
m[1].metric("Scenarios beating the oracle baseline", f"{wins_l}/{n}", f"analytic: {wins_a}/{n}")
m[2].metric("Delivery target met", f"{n_delivery_l}/{n}")
m[3].metric("CTR floor met", f"{n_ctr_l}/{n}", "incl. 2 structurally-tight edge cases, see Methodology")

st.caption(
    f"Analytic (hand-tuned) pacing controller, for comparison: {avg_a:+.1f}% avg CPM vs. oracle baseline, "
    f"{wins_a}/{n} scenarios won — reliable on delivery/CTR but never cost-competitive. "
    "Learned pacing (a linear regression fit against hindsight-optimal pacing behavior, see the "
    "repo README) recovers the cost efficiency the analytic controller traded away when it was "
    "retuned to fix real delivery failures on near-ceiling targets."
)

# ---------------------------------------------------------------------------
# Section 2: per-scenario comparison chart
# ---------------------------------------------------------------------------
st.header("Every evaluation scenario")
st.markdown(
    "CPM vs. the fair (6-level, grid-matched) oracle baseline, one row per scenario. Positive = bandit "
    "cheaper than the oracle baseline. Sorted by the learned controller's result."
)

rows_sorted = sorted(comparison, key=lambda r: r["cpm_vs_naive_learned_pct"] or 0)
scenario_ids = [r["scenario_id"] for r in rows_sorted]
learned_vals = [r["cpm_vs_naive_learned_pct"] for r in rows_sorted]
analytic_vals = [r["cpm_vs_naive_analytic_pct"] for r in rows_sorted]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=scenario_ids,
        x=analytic_vals,
        name="Analytic pacing",
        orientation="h",
        marker_color="#B0B4BA",
    )
)
fig.add_trace(
    go.Bar(
        y=scenario_ids,
        x=learned_vals,
        name="Learned pacing",
        orientation="h",
        marker_color="#2E6F8E",
    )
)
fig.add_vline(x=0, line_color="#444", line_width=1.5, annotation_text="oracle-baseline parity")
if FEATURED_SCENARIO_ID in scenario_ids:
    idx = scenario_ids.index(FEATURED_SCENARIO_ID)
    fig.add_annotation(
        x=max(learned_vals[idx], analytic_vals[idx]) + 1.5,
        y=idx,
        text="← featured below",
        showarrow=False,
        font={"size": 11, "color": "#2E6F8E"},
        xanchor="left",
    )
fig.update_layout(
    barmode="group",
    height=560,
    xaxis_title="CPM vs. oracle baseline (%)",
    yaxis_title=None,
    margin={"t": 10, "b": 10, "l": 10, "r": 120},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
)
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Section 3: featured flight deep-dive
# ---------------------------------------------------------------------------
st.header(f"Deep dive: {FEATURED_SCENARIO_ID}")

result, naive, scenario = load_featured_flight()
traj = result["trajectory"]
final_cpm = result["spend"] / result["delivered_impressions"] * 1000.0
cpm_vs_naive = (naive["cpm"] - final_cpm) / naive["cpm"] * 100

st.markdown(
    f"""
A {scenario['flight_length_days']}-day nominal flight for campaign `{scenario['campaign_id']}`, target
**{scenario['target_impressions']:,} impressions**, CTR floor **{scenario['ctr_floor']:.5f}**. Picked as
the walkthrough because its pacing trajectory — below — shows the dual variables `lambda_delivery` and
`lambda_ctr` doing real, visible work, not settling immediately: both climb together while the bandit is
still cold, then diverge as the two constraints resolve on different schedules — `lambda_ctr` peaks once
the bandit has enough signal to know CTR is tracking safely above floor and decays back down, while
`lambda_delivery` keeps climbing at a steady pace through the rest of the flight, tracking a delivery
target this flight hits almost exactly on nominal length (overrun ~1.0x).
"""
)

fm = st.columns(4)
fm[0].metric("Final CPM", f"{final_cpm:.1f}", f"{cpm_vs_naive:+.1f}% vs. oracle baseline ({naive['cpm']:.1f})")
fm[1].metric("Delivered", f"{result['delivered_impressions']:,}", f"of {result['target_impressions']:,} target")
fm[2].metric("Achieved CTR", f"{result['achieved_ctr']:.5f}", f"floor {result['ctr_floor']:.5f}")
fm[3].metric("Delivery smoothness (CV)", f"{result['delivery_cv']:.2f}", f"{result['overrun_ratio']:.2f}x nominal length")

days = [row["days_used"] for row in traj]
lam_d = [row["lambda_delivery"] for row in traj]
lam_c = [row["lambda_ctr"] for row in traj]
run_ctr = [row["running_ctr"] for row in traj]
cum_cpm = [
    (row["cumulative_spend"] / row["cumulative_delivered"] * 1000.0) if row["cumulative_delivered"] > 0 else None
    for row in traj
]

d1, d2 = st.columns(2)
with d1:
    fig_lam = go.Figure()
    fig_lam.add_trace(go.Scatter(x=days, y=lam_d, mode="lines", name="lambda_delivery", line={"color": "#2E6F8E"}))
    fig_lam.add_trace(go.Scatter(x=days, y=lam_c, mode="lines", name="lambda_ctr", line={"color": "#C97B3D"}))
    fig_lam.add_vline(
        x=scenario["flight_length_days"],
        line_dash="dot",
        line_color="#888",
        annotation_text="nominal length",
        annotation_position="top",
    )
    fig_lam.update_layout(
        title="Lagrangian pacing pressure (dual variables)",
        xaxis_title="Day",
        yaxis_title="RMB / impression",
        height=340,
        margin={"t": 40, "b": 10, "l": 10, "r": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig_lam, width="stretch")
    st.caption(
        "Both variables climb through the first ~10 days as the bandit cold-starts and finds itself "
        "behind pace. lambda_ctr peaks around day 8-9 once running CTR is clearing the floor with a "
        "comfortable margin, then decays back down through the rest of the flight; lambda_delivery keeps "
        "climbing at a steady rate the whole way, consistent with a flight that finishes almost exactly "
        "on its nominal length rather than coasting to an early finish."
    )

with d2:
    floor = scenario["ctr_floor"]
    ctr_above = [c if c >= floor else None for c in run_ctr]
    ctr_below = [c if c < floor else None for c in run_ctr]
    fig_ctr = go.Figure()
    fig_ctr.add_trace(go.Scatter(x=days, y=ctr_above, mode="lines", name="Running CTR (at/above floor)", line={"color": "#2E8B57"}))
    fig_ctr.add_trace(go.Scatter(x=days, y=ctr_below, mode="lines", name="Running CTR (below floor)", line={"color": "#C0392B"}))
    fig_ctr.add_hline(y=floor, line_dash="dash", line_color="gray", annotation_text="CTR floor")
    fig_ctr.add_vline(
        x=scenario["flight_length_days"],
        line_dash="dot",
        line_color="#888",
        annotation_text="nominal length",
        annotation_position="top",
    )
    fig_ctr.update_layout(
        title="Running CTR vs. floor",
        xaxis_title="Day",
        yaxis_title="CTR",
        height=340,
        margin={"t": 40, "b": 10, "l": 10, "r": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig_ctr, width="stretch")
    st.caption(
        "Noisy in the first few days on a small click count, then settles well above the floor — "
        "consistent with lambda_ctr's decay toward 0 in the same window on the left."
    )

fig_cpm = go.Figure()
fig_cpm.add_trace(go.Scatter(x=days, y=cum_cpm, mode="lines", name="Bandit (cumulative CPM)", line={"color": "#2E6F8E"}))
fig_cpm.add_hline(y=naive["cpm"], line_dash="dash", line_color="gray", annotation_text="Oracle flat-bid CPM")
fig_cpm.add_vline(
    x=scenario["flight_length_days"],
    line_dash="dot",
    line_color="#888",
    annotation_text="nominal length",
    annotation_position="top",
)
fig_cpm.update_layout(
    title="Cumulative CPM: bandit vs. oracle baseline",
    xaxis_title="Day",
    yaxis_title="RMB / CPM",
    height=320,
    margin={"t": 40, "b": 10, "l": 10, "r": 10},
)
st.plotly_chart(fig_cpm, width="stretch")
st.caption(
    "The bandit starts above the oracle baseline's flat bid (paying for cold-start exploration with no "
    "prior knowledge) and closes the gap as its win-rate/CTR models pick up signal — ending "
    f"{cpm_vs_naive:+.1f}% below the oracle baseline's flat bid. Want to scrub through this flight "
    "batch-by-batch instead of viewing it as a static chart? Pick "
    f'"{FEATURED_SCENARIO_ID}" on the live demo page.'
)

# ---------------------------------------------------------------------------
# Section 4: a finding -- deadline-crossing pacing collapse, found and fixed
# 2026-09-02. Before-retrain trajectory numbers below are hardcoded, not
# read from a result file: the investigation that found this bug overwrote
# reports/synthetic_scenario_grid_results.learned_pacing.json with the
# retrained (fixed) trajectory, so the broken one is no longer on disk to
# read live. These are the exact values captured during that investigation
# (see the repo README's "Learned pacing" section for the full narrative)
# -- frozen historical reference, matching this project's existing
# convention of citing specific dated numbers in prose (e.g. the README's
# "beats naive on 12/12" quote) rather than only ever computing live.
# ---------------------------------------------------------------------------
st.divider()
st.header("Finding: a deadline-crossing pacing collapse")
st.markdown(
    """
Investigating `grid-90d-highctr-challenging` (90-day nominal flight, `meridian_apparel`, near-ceiling
delivery target + a CTR floor at the population mean) after a reported observation that pacing pressure
was *dropping* right as the flight crossed its nominal length — exactly backwards, since at that point
20% of the target was still undelivered and CTR was still below floor.

**Root cause:** `LearnedPacingController` had never once seen a post-deadline training example. Every
hindsight-search training flight (the simulated flights its weights are fit against) finished at or
before nominal length, so the frozen linear model was extrapolating blindly whenever a real eval flight
ran long — and extrapolated the wrong direction. Confirmed directly:
`np.load('reports/hindsight_pacing_dataset.npz')['X'][:, 0].max()` was ~0.99999, never above 1.0.
"""
)

before_after = st.columns(2)
with before_after[0]:
    fig_before_after_d = go.Figure()
    fig_before_after_d.add_trace(
        go.Scatter(
            x=[0.990, 1.024, 1.058, 1.093, 1.127, 1.161, 1.195, 1.229, 1.263, 1.297, 1.331, 1.366],
            y=[1.4472, 1.2451, 1.0715, 0.9633, 0.8903, 0.8387, 0.8030, 0.7805, 0.7695, 0.7702, 0.7871, 0.8319],
            mode="lines",
            name="Before retrain",
            line={"color": "#C0392B"},
        )
    )
    fig_before_after_d.add_trace(
        go.Scatter(
            x=[0.994, 1.024, 1.054, 1.085, 1.115, 1.145, 1.175, 1.205],
            y=[1.4360, 1.2887, 1.1601, 1.0740, 1.0152, 0.9755, 0.9517, 0.9432],
            mode="lines",
            name="After retrain",
            line={"color": "#2E8B57"},
        )
    )
    fig_before_after_d.add_vline(x=1.0, line_dash="dot", line_color="#888", annotation_text="deadline")
    fig_before_after_d.update_layout(
        title="lambda_delivery near the deadline crossing",
        xaxis_title="Elapsed fraction",
        yaxis_title="RMB / impression",
        height=320,
        margin={"t": 40, "b": 10, "l": 10, "r": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig_before_after_d, width="stretch")

with before_after[1]:
    fig_before_after_c = go.Figure()
    fig_before_after_c.add_trace(
        go.Scatter(
            x=[0.990, 1.024, 1.058, 1.093, 1.127, 1.161, 1.195, 1.229, 1.263, 1.297, 1.331, 1.366],
            y=[0.0953, 0.0710, 0.0480, 0.0311, 0.0173, 0.0055, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            mode="lines",
            name="Before retrain",
            line={"color": "#C0392B"},
        )
    )
    fig_before_after_c.add_trace(
        go.Scatter(
            x=[0.994, 1.024, 1.054, 1.085, 1.115, 1.145, 1.175, 1.205],
            y=[0.5474, 0.4753, 0.4005, 0.3392, 0.2845, 0.2330, 0.1826, 0.1324],
            mode="lines",
            name="After retrain",
            line={"color": "#2E8B57"},
        )
    )
    fig_before_after_c.add_vline(x=1.0, line_dash="dot", line_color="#888", annotation_text="deadline")
    fig_before_after_c.update_layout(
        title="lambda_ctr near the deadline crossing",
        xaxis_title="Elapsed fraction",
        yaxis_title="RMB / impression",
        height=320,
        margin={"t": 40, "b": 10, "l": 10, "r": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig_before_after_c, width="stretch")

st.markdown(
    """
Before the retrain, `lambda_ctr` collapsed to exactly 0 within ~15% of nominal length past the deadline
and stayed there — even though running CTR never once cleared the floor. `lambda_delivery` crashed by
more than the shrinking delivery deficit alone would justify. Both are signs of a linear model
extrapolating into a region it was never trained on, not a reasoned reaction to the actual state.

**The fix, and its limits:** two new interaction features (`overrun_x_pacing_error`/`overrun_x_ctr_error`,
active only once a flight is both overdue *and* still behind) were added specifically for this regime, plus
a new script generating deliberately near-ceiling training scenarios so the hindsight search would
actually produce overrun trajectories to learn from. The interaction features still fit to weight exactly
0 — even the new near-ceiling scenarios finished on time under the hindsight search's best-tuned candidate,
so genuinely-overrun training rows remain at zero. What worked instead: the broader, harder examples raised
R² substantially (0.13→0.39 for `lambda_delivery`, 0.04→0.27 for `lambda_ctr`) and reduced the severity of
the extrapolation error enough, in practice, to fix this scenario — `ctr_met` flipped **False→True**,
overrun dropped **1.37x→1.21x**. The root kink itself (`pacing_error`'s expected-pace curve freezing once
`elapsed_fraction > 1`) is still there, worked around rather than resolved.
"""
)

comparison_before = {
    "cpm_vs_naive_avg": -0.1,
    "wins": 11,
    "ctr_floor_met": 18,
    "delivery_cv_avg": None,
}
st.markdown(
    f"""
The same fix applied everywhere, not just this one scenario — a real trade-off across all 20 evaluation
scenarios, not a strict improvement:

| | Before | After |
|---|---|---|
| Avg CPM vs. oracle baseline | {comparison_before['cpm_vs_naive_avg']:+.1f}% | {avg_l:+.1f}% |
| Scenarios beating the oracle baseline | {comparison_before['wins']}/{n} | {wins_l}/{n} |
| CTR floor met | {comparison_before['ctr_floor_met']}/{n} | {n_ctr_l}/{n} |
| Overrun ratio | worse on all 20 | better on all 20 |

Adopted as the featured result on that basis: a clean sweep on both hard constraints (delivery + CTR
floor) and steadier delivery broadly, at a real but bounded cost in average CPM competitiveness.
"""
)

# ---------------------------------------------------------------------------
# Section 5: methodology & limitations
# ---------------------------------------------------------------------------
st.divider()
st.header("Methodology & limitations")
st.markdown(
    """
- **Synthetic, not real, auctions** — a hierarchical PyMC-generated world, first-price (pay your own
  bid on a win), purpose-built so bid level is a genuine cost/quality trade-off. This project originally
  ran against real iPinYou (GSP) data; that work is preserved on the `archive/real-data-gsp` branch. See
  the README's "Why a synthetic auction environment" section for the full reasoning.
- **The oracle baseline is genuinely oracle-informed** — solved in closed form from the world's *true*
  win-rate/CTR functions, not fitted estimates, and confined to the bandit's own 6-level discrete bid
  grid rather than its own unconstrained continuous bid (a fairness fix — see the README's "Learned
  pacing" section for the numbers before and after).
- **Learned pacing is a frozen linear fit** (`np.linalg.lstsq`, R² ≈ 0.39 for `lambda_delivery`,
  ≈ 0.27 for `lambda_ctr` after the 2026-09-02 retrain, up from 0.13/0.04) trained once via supervised
  regression against hindsight-optimal pacing behavior, then frozen at eval time — it captures real
  signal (this report is the evidence) but leaves most of the variance unexplained.
- **Two scenarios sit at the edge of feasibility** (`grid-30d-highctr-easy`, `grid-90d-highctr-challenging`)
  — even the oracle baseline's own CTR-floor margin there is only 1.6-4.0%, vs. 20%+ everywhere else, a
  structural difficulty in the scenario, not a controller bug. Learned pacing's first pass missed the CTR
  floor on exactly this pair; the 2026-09-02 retrain (see the repo README's "Learned pacing" section)
  fixed both, after finding the controller had never seen a single post-deadline training example.
- **Single seed per scenario**, not averaged over repeats — a documented simplification.

Full methodology, the fair-baseline finding's history, and the real-data GSP exploration this project
started from are in the
[repo README](https://github.com/abezuglov/adtech_projects).
"""
)
