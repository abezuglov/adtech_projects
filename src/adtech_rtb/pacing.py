"""Lagrangian dual-variable pacing controller for the constrained bandit.

Two scalar debts, updated after each processed batch of auctions:

- `lambda_delivery`: rises when cumulative delivered impressions are behind
  the flight's expected linear pace, falls when ahead. Interpreted as a
  shadow price *in RMB per impression* -- how much above an auction's
  intrinsic value the policy is willing to pay right now to secure
  delivery. bandit.py adds it directly to a per-impression net-value
  calculation alongside price/1000 (also RMB per impression) and the CTR
  term, so all three terms have to live on the same scale or the largest
  one silently dominates the decision regardless of the other two.
- `lambda_ctr`: rises when the running (cumulative) CTR is below the
  scenario's floor, falls when above it. Same RMB-per-impression scale
  once multiplied by a CTR difference -- since CTR values here sit at
  ~1e-4 to 1e-3 (see market_model.py's caveats), its own cap has to be
  much larger in raw terms than the delivery cap to reach a comparable
  RMB contribution.

CALIBRATION BUG FOUND AND FIXED (2026): both duals originally shared one
LAMBDA_MAX=50 cap with no connection to the price term's actual RMB scale
(~0.2-0.3 per impression at this dataset's observed price levels). Since
lambda_delivery reliably climbed past 1 within a single batch of any real
flight, it swamped both the price and CTR terms almost immediately --
confirmed empirically: bid_level_counts were overwhelmingly concentrated
at the single most expensive level regardless of context, and the bandit's
CPM ended up statistically indistinguishable from the naive flat-bid
baseline (73.23 vs 73.10 RMB blended across 15 scenarios) despite having
genuine per-context CTR/price beliefs it was computing but never actually
using once behind pace, which happens almost immediately. Separate,
RMB-scaled caps fix this: a low-CTR, high-price placement can now
legitimately fail the "is this worth it" test and get skipped, rather
than delivery urgency overriding context on every auction past the first
batch.

Pure scalar arithmetic, no model or data dependencies, so these updates
are independently unit-testable without touching the simulator or any
fitted model.
"""

from __future__ import annotations

# RMB per impression -- comparable to price_sample/1000 (~0.2-0.3 at this
# dataset's observed price range), not an arbitrary large ceiling. Caps how
# much above intrinsic value the policy will pay even at maximum urgency;
# staying in this range (rather than the old 50) is what lets price/CTR
# actually compete in the value calculation instead of being swamped.
LAMBDA_DELIVERY_MAX = 1.5

# update_ctr_lambda uses a *relative* CTR gap (dimensionless, O(1) --
# see its docstring for why: two earlier attempts using an absolute gap,
# at eta=500 and eta=5000, both left lambda_ctr negligible next to
# lambda_delivery regardless of the 10x eta change, since CTR values here
# are ~1e-4 scale and no fixed multiplier reliably compensates for that
# across scenarios). With a relative gap, lambda_ctr is already on the
# same RMB-per-impression scale as lambda_delivery by construction, so it
# gets the same cap -- not the 1500 (itself raised from a deadlock-prone
# 5000) an earlier absolute-gap version needed.
LAMBDA_CTR_MAX = LAMBDA_DELIVERY_MAX

# Below this many delivered impressions, running_ctr is judged from too
# small (or zero) a sample to be a meaningful CTR signal -- update_ctr_lambda
# is a no-op below this floor rather than reading an empty/tiny sample as
# "CTR is failing" (see the deadlock note above).
MIN_DELIVERED_FOR_CTR_JUDGMENT = 30


# Convexity of the expected-pace curve (expected_cumulative = target *
# elapsed_fraction**PACE_CONVEXITY). 1.0 = linear pace, matching an
# earlier version of this function. Confirmed via a direct check
# (predict_price's coefficient of variation across real auction contexts:
# 0.4-0.54, p10-p90 spans 3-5x) that there's substantial price
# heterogeneity within a scenario for a selective policy to exploit -- but
# a *linear* pace target pushes lambda_delivery past the price
# distribution's own range within the first few batches of any flight
# (confirmed on 3427-3's trajectory: lambda_delivery=0.363 by batch 8
# already exceeded that scenario's p99 price of ~0.22 RMB/impression),
# which neutralizes price-based selectivity for the rest of the flight
# and is the direct, confirmed reason CPM tracked the naive baseline so
# closely despite the bandit's selection mechanism actually working. A
# convex target (>1) keeps expected pace -- and therefore urgency --
# deliberately low while a flight has ample remaining runway, so the
# policy stays selective (skip pricier auctions, wait for cheaper ones)
# for most of the flight, only escalating sharply as elapsed_fraction
# approaches 1 and the deadline is genuinely close.
PACE_CONVEXITY = 3.0


def update_delivery_lambda(
    lambda_delivery: float,
    delivered: int,
    target_impressions: int,
    elapsed_fraction: float,
    eta: float = 0.5,
    lambda_max: float = LAMBDA_DELIVERY_MAX,
) -> float:
    """`elapsed_fraction` is time-elapsed / nominal flight length (can
    exceed 1.0 once a flight overruns its nominal length -- see
    simulator.py's contractual-delivery extension). Debt keeps rising while
    genuinely behind an *extended* pace, not just the original one.

    `eta` raised from an earlier 0.15 to 0.5 to compensate for
    PACE_CONVEXITY=3.0 making pacing_error itself much smaller for most of
    a flight (e.g. 0.2 elapsed_fraction -> 0.008 expected-pace fraction,
    not 0.2) -- without a compensating gain, the *catch-up* phase near the
    deadline would ramp too slowly to actually recover a real shortfall
    within the remaining batches.

    `lambda_max` defaults to LAMBDA_DELIVERY_MAX (this module's real-data
    calibration) but is overridable: the right cap isn't a universal
    constant, it's calibrated against how much the *cost* term can vary
    across bid levels in whatever environment is calling this. Under real
    GSP data that variation is moot (cost is level-independent -- see
    bandit.py's choose_bids), so 1.5 vs. a ~0.12 price span never actually
    mattered there. Under Phase 5's synthetic first-price environment, cost
    genuinely scales with level across that same ~0.12 span (200-320
    RMB/CPM), and 1.5 was confirmed (empirically, on a real synthetic run)
    to swamp it just as badly as the original LAMBDA_MAX=50 swamped real
    data -- once lambda_delivery nears 1.5, a 0.12 max cost differential
    barely dents the argmax, collapsing bid-level choice back to "always
    maximize win probability" regardless of price. synthetic.py passes a
    smaller, environment-calibrated cap for this reason.
    """
    if elapsed_fraction <= 1.0:
        expected_cumulative = target_impressions * (elapsed_fraction**PACE_CONVEXITY)
    else:
        expected_cumulative = target_impressions
    pacing_error = (expected_cumulative - delivered) / max(target_impressions, 1)
    return float(min(max(lambda_delivery + eta * pacing_error, 0.0), lambda_max))


def update_ctr_lambda(
    lambda_ctr: float,
    running_ctr: float,
    ctr_floor: float,
    delivered: int,
    eta: float = 0.15,
    lambda_max: float = LAMBDA_CTR_MAX,
) -> float:
    """The error term is RELATIVE, `(ctr_floor - running_ctr) / ctr_floor`
    (dimensionless, O(1): +1.0 if running CTR is zero, -0.5 if it's 1.5x
    the floor), not the raw absolute gap. Two tuning attempts with the
    absolute gap (eta=500, then eta=5000 -- a 10x change) produced
    *bit-for-bit identical* simulation outcomes on a real scenario
    (3358-3): lambda_ctr only reached ~0.2 and ~2.0 respectively, both
    still negligible next to lambda_delivery's ~1.3-1.5, because CTR
    values here are ~1e-4 scale (see market_model.py's caveats) -- no
    single eta multiplier reliably lands an absolute-gap lambda_ctr in an
    RMB-comparable range without being fragile to a specific scenario's
    CTR magnitude. A relative gap sidesteps that: it's already O(1)
    regardless of the floor's absolute size, so eta can match
    update_delivery_lambda's own scale instead of needing an ad hoc
    1000x+ multiplier, and the resulting lambda_ctr lives in the same
    RMB-per-impression range as lambda_delivery by construction.

    No-op below MIN_DELIVERED_FOR_CTR_JUDGMENT: `running_ctr` reads as
    exactly 0 while `delivered=0` (no wins yet to have observed a click on),
    which is always "below floor" -- judging CTR performance from that,
    before any real data exists, was confirmed to create a real deadlock
    (lambda_ctr climbs while bidding is still zero, further delaying the
    first win that would ever produce real CTR data).
    """
    if delivered < MIN_DELIVERED_FOR_CTR_JUDGMENT:
        return lambda_ctr
    ctr_error = (ctr_floor - running_ctr) / ctr_floor
    return float(min(max(lambda_ctr + eta * ctr_error, 0.0), lambda_max))
