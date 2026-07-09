"""Helper script that prepares the Mendeley "Chilli Leaf Disease Image Dataset for
Classification" for this project.

By default this script does NOT auto-download the dataset (the dataset is hosted
on Mendeley Data and requires a free login).  Instead it:

  1. Prints step-by-step download instructions.
  2. Verifies the dataset exists at the expected location once you have placed
     it by hand.

If you DO have direct HTTP access to the dataset ZIP, you can pass --url and
the script will fetch and extract it for you.

Usage examples
--------------
    # Just verify / tell me where to put the data
    python scripts/download_data.py --data-dir data/ChilliLeaf

    # Auto-download from a direct URL
    python scripts/download_data.py --data-dir data/ChilliLeaf \
        --url https://example.com/mendeley_chilli.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


DATASET_LANDING = (
    "https://data.mendeley.com/"  # search for "Chilli Leaf Disease Mendeley"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare the Mendeley Chilli Leaf Disease dataset for this project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=Path, required=True,
        help="Where to place the extracted dataset, e.g. data/ChilliLeaf.",
    )
    p.add_argument(
        "--url", type=str, default=None,
        help="Optional direct URL of a ZIP containing the dataset.",
    )
    p.add_argument(
        "--force", action="store_true", help="Re-extract even if the dataset already looks valid.",
    )
    return p.parse_args()


def print_manual_instructions() -> None:
    print("\nManual download steps")
    print("---------------------")
    print("1. Open Mendeley Data and search for 'Chilli Leaf Disease Image Dataset'.")
    print(f"   Landing-page hint: {DATASET_LANDING}")
    print("2. Create a free Mendeley account / sign in if asked.")
    print('3. Click "Download All" (or grab the ZIP).')
    print("4. Move the ZIP anywhere on your machine.")
    print('5. Re-run this script with --url pointing at the local ZIP file, e.g.:')
    print('     python scripts/download_data.py --data-dir data/ChilliLeaf \\')
    print('                                       --url "C:\\Users\\you\\Downloads\\chilli.zip"\n')


def looks_like_dataset(root: Path) -> bool:
    if not root.exists():
        return False
    expected = {"Bacterial_Spot", "Cercospora_Leaf_Spot", "Curl_Virus",
                "Healthy_Leaf", "Nutrition_Deficiency", "Powdery_Mildew"}
    have = {p.name for p in root.iterdir() if p.is_dir()}
    return expected.issubset(have)


def fetch_and_extract(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "_download.zip"
    import urllib.request
    print(f"Downloading {url} -> {out}")
    with urllib.request.urlopen(url) as resp, open(out, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    print(f"Extracting {out} -> {dest}")
    with zipfile.ZipFile(out) as zf:
        zf.extractall(dest)
    out.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    dest: Path = args.data_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if looks_like_dataset(dest) and not args.force:
        print(f"[OK] Dataset already prepared at {dest}")
        return 0
    if args.url:
        try:
            fetch_and_extract(args.url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Could not auto-download: {exc}", file=sys.stderr)
            print_manual_instructions()
            return 1
    else:
        print(f"Dataset not found at {dest}.")
        print_manual_instructions()
        return 2
    if looks_like_dataset(dest):
        print(f"[OK] Dataset ready at {dest}")
        return 0
    print("[WARN] Extraction finished but the expected class folders are missing.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
