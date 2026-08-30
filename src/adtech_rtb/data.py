"""Parsing and joining for the raw iPinYou Season 2 RTB logs.

Schema confirmed against both iPinYou's own dataset paper (Liao et al.,
KDD'14) and direct inspection of the downloaded raw files — see
data/README.md for the full field-by-field documentation and the caveats
that matter for modeling (bidding log = DSP's own bid population; training
data used a deliberately aggressive "fixed relatively high-price" strategy
for price-discovery, not iPinYou's real production bidding algorithm).
"""

from pathlib import Path

import pandas as pd

BID_COLUMNS = [
    "bid_id",
    "timestamp",
    "ipinyou_id",
    "user_agent",
    "ip",
    "region_id",
    "city_id",
    "ad_exchange",
    "domain",
    "url",
    "anon_url_id",
    "ad_slot_id",
    "ad_slot_width",
    "ad_slot_height",
    "ad_slot_visibility",
    "ad_slot_format",
    "ad_slot_floor_price",
    "creative_id",
    "bidding_price",
    "advertiser_id",
    "user_profile_ids",
]

# Impression, click, and conversion logs share this 24-column format;
# `log_type` (1/2/3) distinguishes them, though each is also stored in its
# own file (imp.*/clk.*/conv.*).
EVENT_COLUMNS = [
    "bid_id",
    "timestamp",
    "log_type",
    "ipinyou_id",
    "user_agent",
    "ip",
    "region_id",
    "city_id",
    "ad_exchange",
    "domain",
    "url",
    "anon_url_id",
    "ad_slot_id",
    "ad_slot_width",
    "ad_slot_height",
    "ad_slot_visibility",
    "ad_slot_format",
    "ad_slot_floor_price",
    "creative_id",
    "bidding_price",
    "paying_price",
    "landing_page_url",
    "advertiser_id",
    "user_profile_ids",
]

# Columns needed only from the bid log (context features + the DSP's own
# bid) vs. only from the event log (outcome fields) when joining on bid_id.
_EVENT_ONLY_COLUMNS = ["paying_price"]

_NUMERIC_COLUMNS = [
    "region_id",
    "city_id",
    "ad_exchange",
    "ad_slot_width",
    "ad_slot_height",
    "ad_slot_visibility",
    "ad_slot_format",
    "ad_slot_floor_price",
    "bidding_price",
    "advertiser_id",
]


def _read_log(path: Path, columns: list[str]) -> pd.DataFrame:
    # Forcing dtype=str for every column (the previous approach) parses each
    # value as a Python object, which is 5-10x more memory than letting
    # pandas infer native/Arrow-backed types — enough to OOM on the largest
    # daily bid log (~625MB compressed, several GB decompressed).
    dtype = {col: "Int32" for col in _NUMERIC_COLUMNS if col not in ("bidding_price", "ad_slot_floor_price")}
    dtype["bidding_price"] = "float32"
    dtype["ad_slot_floor_price"] = "float32"
    if "paying_price" in columns:
        dtype["paying_price"] = "float32"
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        dtype=dtype,
        dtype_backend="pyarrow",
        na_values=["null", ""],
        keep_default_na=False,
        compression="infer",
        low_memory=False,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S%f")
    return df


def read_bid_log(path: Path, advertiser_ids: list[int] | None = None) -> pd.DataFrame:
    df = _read_log(path, BID_COLUMNS)
    if advertiser_ids is not None:
        df = df[df["advertiser_id"].isin(advertiser_ids)]
    return df


def read_event_log(path: Path, advertiser_ids: list[int] | None = None) -> pd.DataFrame:
    df = _read_log(path, EVENT_COLUMNS)
    if advertiser_ids is not None:
        df = df[df["advertiser_id"].isin(advertiser_ids)]
    return df


def build_daily_dataset(
    bid_path: Path,
    imp_path: Path,
    clk_path: Path,
    advertiser_ids: list[int],
) -> pd.DataFrame:
    """Join one day's bid/impression/click logs into one row-per-bid table.

    Every row is a bid the DSP submitted (win or lose). `won` and
    `paying_price` come from the impression log (a bid_id present there
    won; absent means it lost). `clicked` comes from the click log.
    Conversion is intentionally left out of this join for now — Phase 0
    scope is CTR + win/loss; conversions are sparse enough to need their
    own dedicated check (see data/README.md's conditional-CVR note)
    before deciding whether to model them.
    """
    bids = read_bid_log(bid_path, advertiser_ids)
    imps = read_event_log(imp_path, advertiser_ids)[["bid_id", "paying_price"]]
    clks = read_event_log(clk_path, advertiser_ids)[["bid_id"]]

    df = bids.merge(imps, on="bid_id", how="left")
    df["won"] = df["paying_price"].notna()
    clicked_ids = set(clks["bid_id"])
    df["clicked"] = df["bid_id"].isin(clicked_ids) & df["won"]

    return df
