"""Download the raw iPinYou Season 2 RTB dataset.

The original UCL hosts named in iPinYou's own dataset paper
(bunwell.cs.ucl.ac.uk, data.computational-advertising.org) are dead as of
this writing. This script pulls a verified-working third-party mirror from
Figshare instead. See data/README.md and DATA_LICENSE.md for details,
citation, and alternative sources (Kaggle, Hugging Face) that were
considered but not used.
"""

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

FIGSHARE_URL = "https://ndownloader.figshare.com/files/10082688"
EXPECTED_MD5 = "9ee18bf8c4dd19d1c41e6e77088367f9"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ZIP_NAME = "ipinyou.contest.dataset-season2.zip"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME

    if not zip_path.exists():
        print(f"Downloading {ZIP_NAME} (~3.6 GB) from Figshare...")
        urlretrieve(FIGSHARE_URL, zip_path)
    else:
        print(f"{zip_path} already exists, skipping download.")

    checksum = md5sum(zip_path)
    if checksum != EXPECTED_MD5:
        raise RuntimeError(
            f"MD5 mismatch for {zip_path}: got {checksum}, expected {EXPECTED_MD5}. "
            "The file may be corrupted or the mirror may have changed."
        )
    print("Checksum verified.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    print(f"Done. Raw season 2 files are under {RAW_DIR}.")
    print(
        "Next: process with wnzhang/make-ipinyou-data "
        "(https://github.com/wnzhang/make-ipinyou-data), then run "
        "scripts/make_dataset.py."
    )


if __name__ == "__main__":
    main()
