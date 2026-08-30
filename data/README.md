# Data Dictionary & Acquisition — iPinYou RTB Dataset

See `DATA_LICENSE.md` at the repo root for license/citation terms. No data is committed to this repo (`.gitignore` excludes `data/raw/`, `data/interim/`, and `data/processed/*.parquet`) — `train.parquet`/`test.parquet` alone total ~4.6GB, well past GitHub's 100MB per-file limit. Run `scripts/download_data.py` then `scripts/make_dataset.py` to regenerate `data/processed/` locally; a small sampled subset for the Streamlit demo will be committed separately in Phase 5.

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

### Relevance to today's (first-price) programmatic market

This dataset is from 2013, when GSP/second-price was the dominant RTB mechanism. Most major ad exchanges moved to **first-price** auctions around 2017-2019 (header bidding made GSP's assumptions break down across multiple simultaneous auctions for the same impression; Google Ad Manager fully switched in 2019). So it's fair to ask whether this project's simulator says anything about today's market. The honest answer is nuanced, not a flat "no":

- **The win condition itself is mechanism-agnostic.** "Bid beats the competing threshold" is the win condition under both GSP and first-price — `Paying Price` in this data *is* that threshold. Holding the competing bids fixed, the win-rate model (`market_model.py`'s win-rate classifier) would predict the same win/loss outcome under either payment rule. Only the payment differs: GSP charges `Paying Price`, first-price charges your own bid. In that (counterfactual, static-competitor) sense, a first-price replay would actually be *simpler* than this project's — spend when won is just your own bid, no price-regression needed at all.
- **What doesn't transfer is why the competing bids look the way they do.** GSP (like the Vickrey auction it generalizes) makes truthful bidding a dominant strategy — a bidder has no incentive to shade below their true value. First-price does the opposite: if you pay what you bid, you're incentivized to shade below value, and *how much* you shade is an equilibrium response to your beliefs about competitors — which itself depends on the mechanism everyone is playing under. So the entire distribution of competing bids implicit in this dataset's `Paying Price` values is an artifact of competitors' GSP-era incentives. Rerunning the same auctions under first-price wouldn't just change the payment rule — every bidder (including competitors we never directly observe) would rationally re-price, producing a different threshold distribution entirely. This is the same problem as the Lucas critique in economics: a model fit under one policy regime doesn't reliably predict behavior under a different regime, because the regime itself shapes the very behavior the model learned from.
- **First-price historical logs are actually harder to learn a price-response curve from, not easier.** A nice, under-appreciated property of GSP is that one observed `Paying Price` reveals the full counterfactual win/lose outcome at *every* bid level around it (bid above it → win, below it → lose) — a single number traces out your whole win-rate curve for that auction. First-price auctions typically don't disclose an equivalent clearing price; you only learn whether your *one* submitted bid won or lost, with no signal about what would have happened at any other bid level. Reconstructing a price-response curve from first-price logs generally requires either deliberate bid experimentation (multiple probes at different prices) or exchange-provided bid-landscape signals — this dataset's GSP structure is unusually informative by comparison.

Net: this project should be read as a faithful model of a 2013 GSP market, not a first-price one — and that's a difference in the underlying strategic environment, not just a payment-formula tweak.

### Anticipated pushback ("this is old data, markets have changed")

- **"2013 data, markets have changed"** — true, and stated upfront throughout this README. But the algorithmic contribution here (a constrained contextual bandit deciding how much to bid under a delivery target and a CTR floor) is mechanism-independent: payment rule only changes one function — spend when won — not the bandit's state, action space, or constraint logic. The bidding/evaluation code exposes this as a `mechanism` switch (`second_price`: spend = the fitted price model's output; `first_price`: spend = your own bid, no price model needed at all) — a one-line swap, which is itself evidence the core method isn't mechanism-locked. GSP is kept as the primary reported result because it's what this data can actually ground-truth; first-price mode is structurally supported but not empirically validated against real first-price logs (see below).
- **"GSP is dead, so this validates nothing about today"** — partially true, and precisely scoped rather than hand-waved: the *win condition* replays correctly under either mechanism given static competitors (see above); what doesn't transfer is the *competing-bid distribution*, because it's endogenous to which mechanism elicited it (truthful bidding under GSP vs. shaded bidding under first-price — a Lucas-critique-style regime dependency). Naming exactly what breaks is more defensible than a blanket "may not generalize."
- **"Why not use current data?"** — no public dataset with this fidelity exists for the modern market. Real bid/impression/paying-price logs are commercially sensitive — iPinYou remains one of the only released datasets with genuine ground-truth win/loss and clearing prices. That's a data-availability constraint on this whole subfield, not a corner cut specific to this project.
- **"So what's actually old here?"** — precisely: the *fitted market simulator* (CTR/win-rate/price models) is a model of a 2013 market, and is labeled as such everywhere it's used. The *bidding algorithm* is not fit to 2013 data at all — it's a general method evaluated against that simulator, the same way an RL paper evaluates a policy against a fixed benchmark environment without claiming the benchmark is the current real economy.
- **A point in this dataset's favor, not just a defense** — GSP logs are strictly more informative for offline bid-response reconstruction than first-price logs would be: one observed `Paying Price` reveals the full counterfactual win/lose curve across every bid level, while a first-price log only reveals the outcome of the one bid actually submitted. So this dataset arguably gives a *better* training ground for the offline bidding methodology than a modern first-price log would, independent of the currency question.

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

**Important caveat found in the dataset's own README (not mentioned in any secondary source we found)**: for this training data, "we have run five advertiser campaigns to get these logs with a fixed relatively high-price bidding strategy... which is for the purpose of getting enough impressions and their paying prices and is different from that of our internal live bidding algorithms." In other words, the historical bids in this log are **not** iPinYou's real, economically-optimized production bidding strategy — they're a deliberately aggressive, high-bid strategy designed to win more auctions and observe more paying prices for research purposes. Implications:
- This is actually *good* for building the market/win-rate simulator: it means broader, less-biased coverage of the paying-price distribution than a normal profit-optimizing strategy would produce (fewer systematically-missing high-price auctions).
- It means the historical bidding behavior should **not** be described as "iPinYou's real bidding strategy" in the README/naive-baseline framing — the "naive baseline" built for this project is a separate, simple strategy we define (e.g. constant bid), not a replication of what's in the log.
- The dataset's own README also states desktop display CTR is typically **0.01%-0.2%** (matches the ~0.07-0.1% figure used in planning, now confirmed from the primary source with a wider stated range).

## Verified against real downloaded data (2026-08-29)

Confirmed by direct inspection of `training2nd/bid.20130606.txt.bz2` and `imp.20130606.txt.bz2`:
- Bid log rows have exactly 21 tab-separated fields, impression/click/conversion log rows have exactly 24 — matching the schema tables above field-for-field.
- Ad Exchange is encoded as an integer (`1` = Tanx in the sample rows seen), consistent with iPinYou's Table 2, not the string labels (`adx`/`tanx`) mentioned in some secondary documentation — **use the integer encoding**, verify string labels don't also appear elsewhere in the data.
- A sample won impression showed `Bidding Price=227, Paying Price=207` (bid > payprice, DSP pays payprice) — confirms GSP win/payment logic end-to-end on real data.
- All 5 configured Season 2 advertisers (1458, 3358, 3386, 3427, 3476) are present. An early 200K-row single-day sample suggested 1458 was dominant (~60%) and 3476 the sparsest — **corrected after processing the full 7-day pipeline**: per-advertiser train-split bid counts are 3427: 12.3M, 1458: 11.8M, 3386: 11.2M, 3476: 5.8M, **3358: 2.4M (actual sparsest, by a wide margin)**. `config/config.yaml` now uses **3358** as the `warm_start_holdout` campaign instead of 3476.

## Region/city and user-profile mapping files

iPinYou also provides `region.txt` (region ID → name), `city.txt` (city ID → name), and a DAAT category ID → name mapping (English & Chinese). Place these under `data/raw/` alongside the per-advertiser folders.
