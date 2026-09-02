"""Live Streamlit demo: replay one synthetic delivery scenario batch-by-
batch, showing the bandit's behavior against a FAIR naive flat-bid baseline
(same discretized bid grid, not naive's own unconstrained continuous bid --
see below) as the flight progresses.

Design: this app does NOT run the simulator itself. A synthetic flight is
pure numpy but still takes minutes per scenario at the pipeline's real
BATCH_SIZE (~2-4 min; a few scenarios run longer -- see
scripts/run_synthetic_bandit.py's own elapsed_seconds output), which is a
multi-minute spinner if triggered live on scenario selection -- unacceptable
for a demo meant to be clicked through quickly. Instead, the app reads
already-validated result files (produced by `python scripts/run_synthetic_bandit.py`
/`run_scenario_grid.py`, which must be re-run any time
simulate_synthetic_flight's trajectory schema changes) and replays the
precomputed `trajectory` list -- downsampled to MAX_FRAMES steps so
scrubbing/autoplay stays smooth regardless of how many raw batches the real
flight took. A useful side effect: this app has no dependency on pymc/scipy/
the modeling stack at all, only on the JSON results and light plotting libs.

Featured bandit config: LEARNED pacing (the outer-loop regression-fit
controller, not the hand-tuned analytic Lagrangian formula -- see
learned_pacing.py) at the default 6 discretized bid levels. This is a
deliberate choice, not the only config that was ever run: the analytic
pacing controller, re-tuned for near-ceiling delivery targets, no longer
beats even a fair naive baseline on CPM (see reports/pacing_comparison.json
and the README's "Learned pacing" section for the full comparison) -- it's
reliable (hits delivery + CTR floor everywhere) but not cost-competitive.
Learned pacing recovers that, landing at ~parity with naive on average and
ahead of it on the hardest ("challenging") scenarios.

Naive's own baseline is deliberately solved over the SAME 6-level discrete
grid the bandit is confined to (`--bid-levels 6`), not naive's own
unconstrained continuous bid -- solve_delivery_bid_synthetic's default
bisection search picks any real-valued bid, which is an advantage no
discretized policy has. Comparing the bandit against that continuous
optimum was found to inflate the apparent gap substantially (see the
README) -- both sides now search the identical discrete action space, which
is the fair, apples-to-apples comparison point. (A 25-level variant was
also tried -- see the README -- and made things *worse* for both
controllers, most likely the added cold-start exploration cost of 26
actions instead of 7 outweighing the finer bid precision; 6 levels stayed
the featured config.)
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = REPO_ROOT / "config" / "synthetic_scenarios.yaml"
# .6levels = naive solved over the same 6-level grid the bandit uses, not its
# own unconstrained continuous bid (see module docstring).
NAIVE_RESULTS_PATH = REPO_ROOT / "reports" / "synthetic_naive_baseline_results.6levels.json"
# .learned_pacing = the outer-loop regression-fit pacing controller, not the
# hand-tuned analytic Lagrangian formula (see module docstring).
BANDIT_RESULTS_PATH = REPO_ROOT / "reports" / "synthetic_bandit_results.learned_pacing.json"

# Designed 2x2x2 factorial grid (duration x ctr_level x avails; see
# scripts/generate_scenario_grid.py) -- merged into the same selector as the
# random per-campaign set above, not a separate view, so the demo can show
# both "broad coverage" and "one axis at a time" scenarios side by side.
GRID_SCENARIOS_PATH = REPO_ROOT / "config" / "synthetic_scenario_grid.yaml"
GRID_NAIVE_RESULTS_PATH = REPO_ROOT / "reports" / "synthetic_naive_baseline_grid_results.6levels.json"
GRID_BANDIT_RESULTS_PATH = REPO_ROOT / "reports" / "synthetic_scenario_grid_results.learned_pacing.json"

# Mirrors bandit.DEFAULT_BID_BOUNDS/BID_LEVELS -- duplicated here (not
# imported) so this app has no import-time dependency on the modeling stack,
# see module docstring. Six discretized first-price bid levels, 200-320
# RMB/CPM, plus an explicit "skip" option the policy also has available.
BID_LEVELS = np.linspace(200.0, 320.0, 6)
BID_LEVEL_LABELS = [f"{int(level)}" for level in BID_LEVELS] + ["skip"]

MAX_FRAMES = 200  # animation steps; a real flight has far more raw batches

st.set_page_config(page_title="RTB Bandit Live Demo", layout="wide")


@st.cache_resource
def get_scenarios():
    with open(SCENARIOS_PATH) as f:
        random_scenarios = yaml.safe_load(f)["scenarios"]
    with open(GRID_SCENARIOS_PATH) as f:
        grid_scenarios = yaml.safe_load(f)["scenarios"]
    return random_scenarios + grid_scenarios


@st.cache_resource
def get_naive_results():
    with open(NAIVE_RESULTS_PATH) as f:
        rows = json.load(f)
    with open(GRID_NAIVE_RESULTS_PATH) as f:
        rows += json.load(f)
    return {r["scenario_id"]: r for r in rows}


@st.cache_resource
def get_bandit_results():
    with open(BANDIT_RESULTS_PATH) as f:
        rows = json.load(f)
    with open(GRID_BANDIT_RESULTS_PATH) as f:
        rows += json.load(f)
    return {r["scenario_id"]: r for r in rows}


def run_flight(scenario_id: str):
    scenario = {s["id"]: s for s in get_scenarios()}[scenario_id]
    result = get_bandit_results()[scenario_id]
    return scenario, result


@st.cache_data
def build_frames(scenario_id: str):
    """Precompute everything the animation needs per frame: cumulative CPM,
    cumulative bid-level share (including skip), and the raw trajectory rows
    to sample, downsampled to at most MAX_FRAMES steps."""
    scenario, result = run_flight(scenario_id)
    trajectory = result["trajectory"]
    n = len(trajectory)
    frame_rows = np.linspace(0, n - 1, min(MAX_FRAMES, n)).astype(int)

    level_order = list(BID_LEVELS)
    cum_level_counts = np.zeros(len(level_order) + 1)  # + skip
    cum_counts_at_frame = []
    days = []
    cum_cpm = []
    cum_delivered = []
    cum_clicks = []
    cum_opportunities = []
    running_ctr = []
    lambda_delivery = []
    lambda_ctr = []

    frame_set = set(frame_rows.tolist())
    opportunities_seen = 0
    for i, row in enumerate(trajectory):
        # JSON object keys are always strings, even though synthetic.py
        # writes float bid levels -- coerce back before matching BID_LEVELS.
        for level, count in row["batch_bid_level_counts"].items():
            idx = level_order.index(float(level))
            cum_level_counts[idx] += count
        bid_total = sum(row["batch_bid_level_counts"].values())
        cum_level_counts[-1] += row["batch_size"] - bid_total
        # "Opportunities" = auctions the policy actually saw (batch_size per
        # hour-batch), distinct from cum_delivered (auctions it won) -- the
        # gap between the two is the bandit being selective, not idle.
        opportunities_seen += row["batch_size"]

        if i in frame_set:
            days.append(row["days_used"])
            spend = row["cumulative_spend"]
            delivered = row["cumulative_delivered"]
            cum_cpm.append((spend / delivered * 1000.0) if delivered > 0 else 0.0)
            cum_delivered.append(delivered)
            cum_clicks.append(row["cumulative_clicks"])
            cum_opportunities.append(opportunities_seen)
            running_ctr.append(row["running_ctr"])
            lambda_delivery.append(row["lambda_delivery"])
            lambda_ctr.append(row["lambda_ctr"])
            cum_counts_at_frame.append(cum_level_counts.copy())

    # Rate = derivative of the cumulative curves w.r.t. days, one value per
    # displayed frame (i.e. averaged over however many raw batches fall
    # between consecutive frames -- coarser than per-batch resolution when
    # downsampled to MAX_FRAMES, but that's the right amount of smoothing
    # for a chart meant to show pacing steadiness rather than per-batch
    # noise). A perfectly steady flight is a horizontal line here; naive's
    # is exactly horizontal by construction (see fig4 below), which is what
    # the bandit's own rate is being judged against.
    rate_delivered = []
    rate_opportunities = []
    for idx in range(len(days)):
        dt = days[idx] - (days[idx - 1] if idx > 0 else 0.0)
        d_delivered = cum_delivered[idx] - (cum_delivered[idx - 1] if idx > 0 else 0)
        d_opp = cum_opportunities[idx] - (cum_opportunities[idx - 1] if idx > 0 else 0)
        rate_delivered.append(d_delivered / dt if dt > 0 else 0.0)
        rate_opportunities.append(d_opp / dt if dt > 0 else 0.0)

    return {
        "scenario": scenario,
        "result": result,
        "n_frames": len(days),
        "days": days,
        "cum_cpm": cum_cpm,
        "cum_delivered": cum_delivered,
        "cum_clicks": cum_clicks,
        "cum_opportunities": cum_opportunities,
        "rate_delivered": rate_delivered,
        "rate_opportunities": rate_opportunities,
        "running_ctr": running_ctr,
        "lambda_delivery": lambda_delivery,
        "lambda_ctr": lambda_ctr,
        "level_shares": [counts / max(counts.sum(), 1) for counts in cum_counts_at_frame],
    }


st.title("RTB Campaign Delivery Optimizer — Live Bandit Demo")
st.caption(
    "The power of context: naive knows the true auction economics perfectly but must bid one flat number "
    "for the whole campaign; the bandit starts knowing nothing but can price by placement and time. "
    "Replaying a precomputed flight batch-by-batch — see the [repo README](https://github.com/abezuglov/adtech_projects) for the full result."
)

scenarios = get_scenarios()
naive_results = get_naive_results()
def _scenario_label(s: dict) -> str:
    if "avails" in s:  # designed grid scenario -- lead with its three axes, not a raw id
        return (
            f"[Grid] {s['duration']} · {s['ctr_level']} CTR · {s['avails']} avails  ·  "
            f"{s['campaign_id']}  ·  target {s['target_impressions']:,} impr"
        )
    return f"{s['id']}  ·  {s['campaign_id']}  ·  {s['flight_length_days']}d flight  ·  target {s['target_impressions']:,} impr"


scenario_labels = {s["id"]: _scenario_label(s) for s in scenarios}
scenario_id = st.selectbox(
    "Scenario", options=[s["id"] for s in scenarios], format_func=lambda sid: scenario_labels[sid]
)

data = build_frames(scenario_id)
scenario, result = data["scenario"], data["result"]
naive = naive_results[scenario_id]
n_frames = data["n_frames"]

# Nominal flight_length_days is sized to the naive flat bid's expected pace
# (see generate_synthetic_scenarios), not to the bandit's. Under the CTR
# floor the bandit bids more conservatively and needs more calendar time to
# hit the same delivery target -- simulate_synthetic_flight explicitly
# allows running past nominal length (up to max_overrun_multiple), so "Day"
# exceeding the nominal count below is expected, not a bug.
if result["overrun_ratio"] > 1.05:
    st.caption(
        f"⏱ This flight runs {result['overrun_ratio']:.2f}x past its nominal {scenario['flight_length_days']}d "
        "window: the bandit cold-starts with no prior knowledge of win rates or CTR, so early exploration "
        "and pacing correction can push a flight past its nominal length even when it ultimately hits target."
    )

if "frame" not in st.session_state or st.session_state.get("scenario_id") != scenario_id:
    st.session_state.frame = 0
    st.session_state.scenario_id = scenario_id

control_cols = st.columns([1, 1, 3, 2])
if control_cols[0].button("▶ Play", width='stretch'):
    st.session_state.play = True
if control_cols[1].button("⏮ Reset", width='stretch'):
    st.session_state.frame = 0
    st.session_state.play = False
speed = control_cols[2].select_slider(
    "Speed", options=["Slow", "Medium", "Fast"], value="Medium", label_visibility="collapsed"
)

# Deliberately unkeyed (no key="frame"): Streamlit forbids writing to
# session_state[key] once a widget with that key has been instantiated in
# the same run, which the autoplay loop below needs to do every frame.
# Using `value=` instead and syncing session_state right after keeps the
# slider reflecting whichever source (drag, Play, Reset) last moved it.
frame = st.slider(
    "Flight progress", 0, n_frames - 1, value=st.session_state.frame, label_visibility="collapsed"
)
st.session_state.frame = frame
control_cols[3].caption(f"Day {data['days'][frame]:.1f} (nominal length {scenario['flight_length_days']}d)")

placeholder = st.empty()

# Chart keys are built from this counter, not from the frame index `i`:
# within one script execution render() can be called twice for the same
# frame (once up front, again as the Play loop's first iteration), which
# would otherwise produce two plotly_chart elements with an identical key.
_render_seq = itertools.count()


def render(i: int):
    seq = next(_render_seq)
    with placeholder.container():
        delivered = data["cum_delivered"][i]
        target = scenario["target_impressions"]
        ctr = data["running_ctr"][i]
        floor = scenario["ctr_floor"]
        bandit_cpm = data["cum_cpm"][i]
        naive_cpm = naive["cpm"]

        m = st.columns(4)
        m[0].metric("Delivered", f"{delivered:,}", f"of {target:,} target")
        m[1].metric(
            "Bandit CPM (so far)",
            f"{bandit_cpm:.1f}",
            f"{(bandit_cpm - naive_cpm):+.1f} vs naive {naive_cpm:.1f}",
            delta_color="inverse",
        )
        m[2].metric("Running CTR", f"{ctr:.5f}", f"floor {floor:.5f}")
        m[3].metric("Day", f"{data['days'][i]:.1f}", f"nominal length {scenario['flight_length_days']}d")

        p1, p2 = st.columns(2)
        with p1:
            st.progress(min(delivered / target, 1.0), text=f"Delivery: {delivered / target:.0%} of target")
        with p2:
            ctr_frac = min(ctr / floor, 1.5) / 1.5 if floor > 0 else 0.0
            met = ctr >= floor
            st.progress(ctr_frac, text=f"CTR vs floor: {'above' if met else 'below'} ({ctr:.5f} / {floor:.5f})")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data["days"][: i + 1], y=data["cum_cpm"][: i + 1], mode="lines", name="Bandit (cumulative CPM)"
                )
            )
            fig.add_hline(y=naive_cpm, line_dash="dash", line_color="gray", annotation_text="Naive flat-bid CPM")
            fig.update_layout(
                title="CPM: bandit vs. naive baseline",
                xaxis_title="Day",
                yaxis_title="RMB / CPM",
                height=320,
                margin={"t": 40, "b": 10, "l": 10, "r": 10},
            )
            st.plotly_chart(fig, width='stretch', key=f"cpm_chart_{seq}")

        with c2:
            shares = data["level_shares"][i]
            fig2 = go.Figure(go.Bar(x=BID_LEVEL_LABELS, y=shares))
            fig2.update_layout(
                title="Bid-level distribution (cumulative share)",
                xaxis_title="Bid level (RMB/CPM)",
                yaxis_title="Share of auctions",
                yaxis_range=[0, 1],
                height=320,
                margin={"t": 40, "b": 10, "l": 10, "r": 10},
            )
            st.plotly_chart(fig2, width='stretch', key=f"bidlevel_chart_{seq}")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data["days"][: i + 1], y=data["lambda_delivery"][: i + 1], name="lambda_delivery"))
        fig3.add_trace(go.Scatter(x=data["days"][: i + 1], y=data["lambda_ctr"][: i + 1], name="lambda_ctr"))
        fig3.update_layout(
            title="Lagrangian pacing pressure (dual variables)",
            xaxis_title="Day",
            yaxis_title="RMB / impression",
            height=260,
            margin={"t": 40, "b": 10, "l": 10, "r": 10},
        )
        st.plotly_chart(fig3, width='stretch', key=f"lambda_chart_{seq}")

        # Rate (derivative), not cumulative: a steady flight reads as a flat
        # horizontal line here, and naive's constant-rate baseline literally
        # is one (added via add_hline, not a traced series) -- deviations of
        # the bandit's actual delivery rate from flat are exactly what
        # delivery_cv is scoring, made visible instead of hidden inside a
        # cumulative curve's slope.
        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(
                x=data["days"][: i + 1],
                y=data["rate_opportunities"][: i + 1],
                mode="lines",
                name="Opportunities rate (bandit)",
                line={"color": "lightgray"},
            )
        )
        fig4.add_trace(
            go.Scatter(
                x=data["days"][: i + 1],
                y=data["rate_delivered"][: i + 1],
                mode="lines",
                name="Delivery rate (bandit)",
            )
        )
        naive_rate = target / scenario["flight_length_days"]
        fig4.add_hline(y=naive_rate, line_dash="dash", line_color="gray", annotation_text="Naive ideal rate")
        fig4.update_layout(
            title="Delivery pacing rate: derivative of delivered vs. opportunities, vs. naive's constant rate",
            xaxis_title="Day",
            yaxis_title="Impressions / day",
            height=320,
            margin={"t": 40, "b": 10, "l": 10, "r": 10},
        )
        st.plotly_chart(fig4, width='stretch', key=f"pacing_chart_{seq}")


render(frame)

if st.session_state.get("play"):
    delay = {"Slow": 0.12, "Medium": 0.05, "Fast": 0.015}[speed]
    start = st.session_state.frame
    for i in range(start, n_frames):
        st.session_state.frame = i
        render(i)
        time.sleep(delay)
    st.session_state.play = False
    st.rerun()

st.divider()
st.subheader("Full-flight result (validated, matches reports/synthetic_bandit_results.learned_pacing.json)")
final_cpm = result["spend"] / max(result["delivered_impressions"], 1) * 1000
cpm_improvement = (naive["cpm"] - final_cpm) / naive["cpm"]
r = st.columns(5)
r[0].metric("Final CPM", f"{final_cpm:.1f}", f"{cpm_improvement:+.1%} vs naive")
r[1].metric("Total spend", f"{result['spend']:,.0f}", f"naive {naive['expected_spend']:,.0f}")
r[2].metric("Delivery met", "Yes" if result["delivery_met"] else "No", f"{result['overrun_ratio']:.2f}x nominal length")
r[3].metric("CTR floor met", "Yes" if result["ctr_met"] else "No")
r[4].metric("Delivery smoothness (CV)", f"{result['delivery_cv']:.2f}", "lower = steadier")

st.caption(
    "Naive baseline is a closed-form constant flat bid, solved over the SAME 6-level discrete bid grid the "
    "bandit uses (see solve_delivery_bid_synthetic's bid_levels option) -- not naive's own unconstrained "
    "continuous bid, which was found to inflate the apparent gap. Under first-price auctions its CPM is "
    "exactly the solved bid, shown as the dashed reference line above. "
    "See the [repo README](https://github.com/abezuglov/adtech_projects) for the full methodology, including why "
    "this project moved from real GSP data to a synthetic first-price environment, and the fair-baseline finding."
)
