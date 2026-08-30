"""Build the processed, per-bid dataset from raw iPinYou Season 2 logs.

Reads config/config.yaml for the advertiser list, joins each training day's
bid/impression/click logs (see src/adtech_rtb/data.py), concatenates across
days, and writes a temporal train/test split to data/processed/ as parquet.

Season 2 training spans 2013-06-06 through 2013-06-12 (7 days). We hold out
the last 2 days as the test set and train on the first 5 — an in-sample
temporal split, distinct from iPinYou's own separately-released leaderboard
test file (which lacks the losing-bid population our simulator needs).
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adtech_rtb.data import build_daily_dataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "ipinyou.contest.dataset-season2" / "training2nd"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "daily"

TRAIN_DAYS = ["20130606", "20130607", "20130608", "20130609", "20130610"]
TEST_DAYS = ["20130611", "20130612"]


def load_config() -> dict:
    with open(REPO_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def build_split(days: list[str], advertiser_ids: list[int], output_path: Path) -> dict:
    """Stream each day straight into output_path via ParquetWriter.

    Never holds more than one day's DataFrame in memory at a time — the
    previous approach accumulated every day in a Python list and only
    concatenated+wrote at the end, which OOM'd once 3-4 days' worth of
    full-schema bid logs (each several GB) were resident simultaneously.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0
    total_won = 0
    total_clicked = 0
    advertiser_counts: pd.Series = pd.Series(dtype="int64")
    try:
        for day in days:
            day_path = INTERIM_DIR / f"{day}.parquet"
            if day_path.exists():
                print(f"  {day} already processed, loading checkpoint...")
                day_df = pd.read_parquet(day_path, dtype_backend="pyarrow")
            else:
                bid_path = RAW_DIR / f"bid.{day}.txt.bz2"
                imp_path = RAW_DIR / f"imp.{day}.txt.bz2"
                clk_path = RAW_DIR / f"clk.{day}.txt.bz2"
                print(f"  processing {day}...")
                day_df = build_daily_dataset(bid_path, imp_path, clk_path, advertiser_ids)
                day_df.to_parquet(day_path, index=False)
                print(f"    {day}: {len(day_df):,} bids -> checkpointed")

            total_rows += len(day_df)
            total_won += int(day_df["won"].sum())
            total_clicked += int(day_df["clicked"].sum())
            advertiser_counts = advertiser_counts.add(
                day_df["advertiser_id"].value_counts(), fill_value=0
            )

            table = pa.Table.from_pandas(day_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            del day_df, table
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows": total_rows,
        "won": total_won,
        "clicked": total_clicked,
        "advertiser_counts": advertiser_counts,
    }


def main() -> None:
    config = load_config()
    advertiser_ids = [a["id"] for a in config["advertisers"]]
    print(f"Advertisers: {advertiser_ids}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building train split ({TRAIN_DAYS})...")
    train_stats = build_split(TRAIN_DAYS, advertiser_ids, PROCESSED_DIR / "train.parquet")
    print(f"  train: {train_stats['rows']:,} bids, {train_stats['won']:,} won, "
          f"{train_stats['clicked']:,} clicked")

    print(f"Building test split ({TEST_DAYS})...")
    test_stats = build_split(TEST_DAYS, advertiser_ids, PROCESSED_DIR / "test.parquet")
    print(f"  test: {test_stats['rows']:,} bids, {test_stats['won']:,} won, "
          f"{test_stats['clicked']:,} clicked")

    print("Per-advertiser bid counts (train):")
    print(train_stats["advertiser_counts"].astype(int))


if __name__ == "__main__":
    main()
