"""Feature engineering shared by the CTR and win-rate/market models.

`user_profile_ids` (DAAT usertags) is dropped entirely: the bidding log
never carries it (see data/README.md) so it's an all-null column on every
row we have -- zero information, not worth threading through a join with
the impression log for a v1 model.
"""

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = [
    "region_id",
    "city_id",
    "ad_exchange",
    "ad_slot_visibility",
    "ad_slot_format",
    "advertiser_id",
    "domain",
    "creative_id",
]

NUMERIC_COLUMNS = [
    "hour",
    "weekday",
    "ad_slot_width",
    "ad_slot_height",
    "slot_area",
    "log_floor_price",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def _browser(user_agent: str) -> str:
    ua = user_agent.lower()
    if "chrome" in ua:
        return "chrome"
    if "firefox" in ua:
        return "firefox"
    if "safari" in ua and "chrome" not in ua:
        return "safari"
    if "msie" in ua or "trident" in ua:
        return "ie"
    if "opera" in ua:
        return "opera"
    return "other"


def _os(user_agent: str) -> str:
    ua = user_agent.lower()
    if "windows" in ua:
        return "windows"
    if "mac os" in ua or "macintosh" in ua:
        return "mac"
    if "android" in ua:
        return "android"
    if "iphone" in ua or "ipad" in ua or "ios" in ua:
        return "ios"
    if "linux" in ua:
        return "linux"
    return "other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with model-ready feature columns.

    Ad Slot Visibility of 255 (out of the documented 0/1/2 range) shows up
    consistently across days -- treated here as its own category (an
    "unknown" sentinel), not coerced into the numeric 0/1/2 scale.
    """
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour.astype("int32")
    out["weekday"] = out["timestamp"].dt.dayofweek.astype("int32")
    out["slot_area"] = (out["ad_slot_width"].astype("int64") * out["ad_slot_height"].astype("int64")).astype(
        "int32"
    )
    out["log_floor_price"] = np.log1p(out["ad_slot_floor_price"].astype("float64")).astype("float32")
    ua = out["user_agent"].fillna("").astype(str)
    out["browser"] = ua.map(_browser)
    out["os"] = ua.map(_os)

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("missing").astype(str)

    # LightGBM's pandas bridge only accepts native numpy int/float/bool
    # dtypes, not Arrow-backed extension types -- some of these columns
    # (e.g. ad_slot_width/height) pass straight through from the
    # pyarrow-backed parquet read untouched. `.astype("float64")` alone
    # is not enough here: on this pandas version it keeps Arrow backing
    # (dtype stays "double[pyarrow]"), so go through a raw numpy array.
    for col in NUMERIC_COLUMNS:
        out[col] = out[col].to_numpy(dtype="float64")

    return out


ALL_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS + ["browser", "os"]
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["browser", "os"]


def fit_category_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Fit category->int code maps from training data only.

    LightGBM's native save_model() format does not preserve pandas
    Categorical dtype mappings, so a booster reloaded fresh (as every
    later phase -- bandit, evaluation, the demo -- will do) can't
    consistently score new data against category-dtype columns. Explicit,
    persisted integer codes sidestep that entirely: code 0 is reserved for
    values unseen at fit time (a real possibility for high-cardinality
    columns like domain/creative_id on new data).
    """
    return {
        col: {val: i + 1 for i, val in enumerate(sorted(df[col].unique()))} for col in ALL_CATEGORICAL_COLUMNS
    }


def apply_category_maps(df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in category_maps.items():
        out[col] = out[col].map(mapping).fillna(0).astype("int32")
    return out
