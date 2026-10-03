"""Download and extract the HUPD January-2016 sample (real USPTO applications).

Source: https://huggingface.co/datasets/HUPD/hupd (Suzgun et al., NeurIPS 2022
Datasets & Benchmarks, CC-BY-SA-4.0). ~388MB compressed, 26,808 applications.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/sample-jan-2016.tar.gz"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "sample-jan-2016.tar.gz"
EXTRACT_DIR = RAW_DIR / "hupd_jan2016"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        print(f"Archive already present at {ARCHIVE}, skipping download.")
        return

    def hook(count: int, block_size: int, total_size: int) -> None:
        done = count * block_size
        pct = min(100, done * 100 // total_size) if total_size else 0
        if count % 200 == 0 or done >= total_size:
            print(f"  {pct}% ({done / 1e6:.1f}/{total_size / 1e6:.1f} MB)", flush=True)

    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, ARCHIVE, reporthook=hook)


def extract() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.json")):
        print(f"Already extracted at {EXTRACT_DIR}, skipping.")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {EXTRACT_DIR} ...")
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(EXTRACT_DIR)


if __name__ == "__main__":
    download()
    extract()
    n = len(list((EXTRACT_DIR / "sample" / "2016").glob("*.json")))
    print(f"Done. {n} application records available at {EXTRACT_DIR / 'sample' / '2016'}")
