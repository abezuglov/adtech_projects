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


def update_delivery_lambda(
    lambda_delivery: float,
    delivered: int,
    target_impressions: int,
    elapsed_fraction: float,
    eta: float = 0.15,
) -> float:
    """`elapsed_fraction` is time-elapsed / nominal flight length (can
    exceed 1.0 once a flight overruns its nominal length -- see
    simulator.py's contractual-delivery extension). Debt keeps rising while
    genuinely behind an *extended* pace, not just the original one.

    `eta` scaled down from an earlier 5.0 to match LAMBDA_DELIVERY_MAX's
    much smaller ceiling (1.5, not 50) -- keeps the same relative ramp-up
    speed (reaches the cap after a similar number of batches under sustained
    maximum pacing error) while staying in an RMB-comparable range.
    """
    expected_cumulative = target_impressions * min(elapsed_fraction, 1.0) if elapsed_fraction <= 1.0 else target_impressions
    pacing_error = (expected_cumulative - delivered) / max(target_impressions, 1)
    return float(min(max(lambda_delivery + eta * pacing_error, 0.0), LAMBDA_DELIVERY_MAX))


def update_ctr_lambda(
    lambda_ctr: float,
    running_ctr: float,
    ctr_floor: float,
    delivered: int,
    eta: float = 0.15,
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
    return float(min(max(lambda_ctr + eta * ctr_error, 0.0), LAMBDA_CTR_MAX))
