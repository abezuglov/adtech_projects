# Data Dictionary & Acquisition — iPinYou RTB Dataset

See `DATA_LICENSE.md` at the repo root for license/citation terms. Raw data is **not** committed to this repo (`.gitignore` excludes `data/raw/` and `data/interim/`) — only small, derived `data/processed/` artifacts needed to reproduce results.

## Acquisition

**Note:** the original UCL hosts named in iPinYou's own paper (`bunwell.cs.ucl.ac.uk`, `data.computational-advertising.org`) are dead as of this writing (connection timeouts, likely decommissioned academic servers). A verified-working alternative:

1. Download the raw Season 2 dataset from **Figshare**: `ipinyou.contest.dataset-season2.zip` (3.61 GB, CC0-licensed, DOI `10.6084/m9.figshare.5732328.v1`, uploaded by Zilong Jiang, 2017) — direct link `https://ndownloader.figshare.com/files/10082688`, article page https://figshare.com/articles/dataset/ipinyou_contest_dataset_season2/5732328. This is a third-party mirror of iPinYou's original raw contest files, not a reprocessed/reduced version, so it retains `Paying Price` and the losing-bid population needed for the market simulator. (Other options that exist but were not used: Kaggle competition pages `rtb-dsp-bidding-algorithm-contest` / `ucl-rtb-algorithm-challenge-2015` — unverified reachability in this environment, would need a Kaggle account; Hugging Face `reczoo/iPinYou_x1` — confirmed live but reprocessed into a hashed CTR-benchmark format with no `Paying Price`, unsuitable as the primary source.)
2. Unzip, then process into per-advertiser train/test logs using `wnzhang/make-ipinyou-data` (https://github.com/wnzhang/make-ipinyou-data) — retains the raw text log format below, including `Paying Price`.
3. Place processed per-advertiser folders under `data/raw/<advertiser_id>/`.
4. Run `scripts/make_dataset.py` to filter down to the campaigns selected in `config/config.yaml` and produce `data/processed/`.

We use **Season 2** data: it's the first season with an explicit `Advertiser ID` column (Season 1 advertisers are only distinguishable by hashed Landing Page URL) and user-profile (DAAT) tags.

## Auction mechanism

Confirmed directly from iPinYou's own dataset paper (Liao et al., KDD'14): ad exchanges run the **Vickrey / second-price auction** — the highest bidder wins and pays the *second*-highest bid. In the logs, **Paying Price** is that second-highest/market-clearing price, not the winner's own bid.

iPinYou's own official offline-evaluation methodology (quoted from the paper) is exactly the bid-replay logic this project uses for simulation:

> "If the bidding price of the participant's bidding algorithm is above the paying price in the impression log of testing dataset, the participant's DSP wins this auction and pays the paying price in the impression log for the ad impression."

So a candidate bid wins iff `bid > Paying Price`, and pays `Paying Price` if it wins — this project's replay/simulation logic matches the competition organizer's own scoring method, not an invented approximation.

## Advertiser campaigns (Season 2)

| Advertiser ID | Industry |
|---|---|
| 1458 | Chinese vertical e-commerce |
| 3358 | Software |
| 3386 | International e-commerce |
| 3427 | Oil |
| 3476 | Tire |

Season 3 (for reference, not used by default — no user-profile-tag parity check done yet): 2259 (milk powder), 2261 (telecom), 2821 (footwear), 2997 (mobile e-commerce app install).

`config/config.yaml` selects which subset (3-5) to use.

## Bidding log schema

Columns marked `*` are hashed/modified by iPinYou before release for privacy.

| # | Column | Notes |
|---|---|---|
| 1* | Bid ID | Joins to impression/click/conversion log |
| 2 | Timestamp | When the bid request arrived at the DSP |
| 3* | iPinYou ID | Hashed user cookie |
| 4 | User-Agent | Browser UA string |
| 5* | IP | First 3 bytes only |
| 6 | Region ID | Mainland China region/province |
| 7 | City ID | Mainland China city |
| 8 | Ad Exchange | Confirm exact encoding in downloaded data — paper's Table 2 example uses integer codes 1-6 (Tanx/Adx/Tencent/Baidu/Youku/Amx); other iPinYou documentation uses string labels (`adx`/`tanx`/`tencent`) — verify against the actual files in Phase 0 |
| 9* | Domain | Hashed |
| 10* | URL | Hashed (only one of URL / Anonymous URL ID is meaningful) |
| 11 | Anonymous URL ID | Set when URL isn't passed to DSPs |
| 12 | Ad Slot ID | |
| 13 | Ad Slot Width | |
| 14 | Ad Slot Height | |
| 15 | Ad Slot Visibility | Above/below fold, or unknown |
| 16 | Ad Slot Format | Fixed or popup |
| 17 | Ad Slot Floor Price | RMB/CPM, publisher's reserve price |
| 18 | Creative ID | |
| 19* | Bidding Price | RMB/CPM, what the DSP bid |
| 20 | Advertiser ID | Season 2/3 only |
| 21* | User Profile IDs | DAAT category IDs (Season 2/3); **always null in the bidding log** for privacy — only populated in the impression/click/conversion log |

## Impression / click / conversion log schema

Same columns as above, plus:

| Column | Notes |
|---|---|
| Log Type | 1 = impression, 2 = click, 3 = conversion |
| Paying Price* | RMB/CPM — the market-clearing price (see auction mechanism above) |
| Landing Page URL* | Where the user landed on click/conversion |

**Win/loss derivation**: every row in the bidding log represents a bid the DSP submitted (win or lose — there's no explicit "declined to bid" record). A bid **won** iff its Bid ID appears in the impression log (Log Type 1); if it doesn't appear there, it lost. This is a real outcome label, not an inferred one.

**Caveat carried into modeling**: the bidding log reflects *this DSP's own* bid volume, not the full unconstrained set of auctions the ad exchange saw — so anything built from bid density (e.g. avails/inventory-style estimates) is a proxy for iPinYou's own observed opportunity volume, not raw market avails.

## Region/city and user-profile mapping files

iPinYou also provides `region.txt` (region ID → name), `city.txt` (city ID → name), and a DAAT category ID → name mapping (English & Chinese). Place these under `data/raw/` alongside the per-advertiser folders.
