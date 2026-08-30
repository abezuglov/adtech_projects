# Data License & Attribution

This project analyzes the **iPinYou Global RTB Bidding Algorithm Competition Dataset**, released by iPinYou Inc. after its 2013 three-season DSP bidding competition. The dataset itself is **not included in this repository** (see `data/README.md` for acquisition steps) — only code that downloads, processes, and analyzes it.

## Verbatim permission notice

The dataset is documented in the following paper, which carries this notice on its first page (quoted verbatim):

> Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from Permissions@acm.org.
>
> KDD'14 August 24-27 2014, New York, NY, USA
> Copyright is held by the owner/author(s). Publication rights licensed to ACM.
> Copyright 2014 ACM 978-1-4503-2999-6/14/08 ...$15.00.
> http://dx.doi.org/10.1145/2648584.2648590

iPinYou's own paper additionally states the dataset was released "for public use" as "a great asset for computational advertising research community," for use in "academic research, consulting service, and course project."

**This project's use is academic/portfolio demonstration, non-commercial, and includes full citation — consistent with the terms above.** No claim is made to redistribute the raw dataset itself; this repository only redistributes small, derived, aggregated artifacts (e.g., summary statistics, model outputs) necessary to reproduce the published results.

## Citations

Please cite both if referencing this dataset:

- Hairen Liao, Lingxiao Peng, Zhenchuan Liu, Xuehua Shen. "iPinYou Global RTB Bidding Algorithm Competition Dataset." KDD'14, ACM, 2014. https://doi.org/10.1145/2648584.2648590
- Weinan Zhang, Shuai Yuan, Jun Wang, Xuehua Shen. "Real-Time Bidding Benchmarking with iPinYou Dataset." Technical report, UCL, 2014. https://arxiv.org/abs/1407.7073

## Dataset download

- Baidu WebDrive: http://pan.baidu.com/s/1kTwX2mF
- University College London mirror: http://data.computational-advertising.org
- Updated links maintained at: http://contest.ipinyou.com/data-release.html
- Processing tooling: https://github.com/wnzhang/make-ipinyou-data

For questions about the dataset itself, contact dsp-competition@ipinyou.com (per the original paper).

## Code license

The code in this repository (everything outside the dataset itself) is MIT-licensed — see `LICENSE`.
