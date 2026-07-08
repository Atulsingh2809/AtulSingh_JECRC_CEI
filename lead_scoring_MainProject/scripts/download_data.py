#!/usr/bin/env python3
"""
Download the Lead Scoring dataset from Kaggle.

Usage:
    python scripts/download_data.py

Requires either:
  - kaggle.json in ~/.kaggle/ (or project root), OR
  - KAGGLE_USERNAME and KAGGLE_KEY environment variables.

Falls back with clear instructions if credentials are missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_CSV = RAW_DATA_DIR / "Leads.csv"
KAGGLE_DATASET = "amritachatterjee09/lead-scoring-dataset"


def _print_manual_instructions() -> None:
    print(
        """
================================================================================
  Kaggle credentials not found — manual download required
================================================================================

  1. Go to: https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset
  2. Click "Download" and extract the archive.
  3. Place the CSV file at:

       data/raw/Leads.csv

  To enable automatic download, set up Kaggle API credentials:

  Option A — kaggle.json file:
    - Create an API token at https://www.kaggle.com/settings/account
    - Save kaggle.json to ~/.kaggle/kaggle.json  (Linux/Mac)
      or %USERPROFILE%\\.kaggle\\kaggle.json  (Windows)

  Option B — environment variables:
    set KAGGLE_USERNAME=your_username
    set KAGGLE_KEY=your_api_key

  Then re-run:  python scripts/download_data.py
================================================================================
"""
    )


def _find_kaggle_json() -> Path | None:
    """Locate kaggle.json in standard or project locations."""
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        PROJECT_ROOT / "kaggle.json",
        PROJECT_ROOT / ".kaggle" / "kaggle.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return _find_kaggle_json() is not None


def _ensure_kaggle_config() -> None:
    """Copy project-local kaggle.json to ~/.kaggle if needed."""
    kaggle_json = _find_kaggle_json()
    if kaggle_json is None:
        return

    kaggle_dir = Path.home() / ".kaggle"
    dest = kaggle_dir / "kaggle.json"
    if not dest.exists() and kaggle_json != dest:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kaggle_json, dest)
        # Kaggle CLI requires restrictive permissions on Unix
        if os.name != "nt":
            dest.chmod(0o600)


def _extract_csv_from_zip(zip_path: Path) -> Path | None:
    """Extract Leads.csv from downloaded zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                extracted = RAW_DATA_DIR / Path(name).name
                with zf.open(name) as src, open(extracted, "wb") as dst:
                    dst.write(src.read())
                return extracted
    return None


def download_dataset() -> bool:
    """
    Download dataset via Kaggle CLI.

    Returns True if Leads.csv is available after this call.
    """
    if TARGET_CSV.exists():
        print(f"Dataset already present: {TARGET_CSV}")
        return True

    if not _has_kaggle_credentials():
        _print_manual_instructions()
        return False

    _ensure_kaggle_config()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {KAGGLE_DATASET} via Kaggle CLI...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                str(RAW_DATA_DIR),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print("Kaggle CLI error:")
            print(result.stderr or result.stdout)
            _print_manual_instructions()
            return False
    except FileNotFoundError:
        print("kaggle package not installed. Run: pip install kaggle")
        _print_manual_instructions()
        return False

    # Handle case where --unzip didn't produce expected filename
    if not TARGET_CSV.exists():
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if len(csv_files) == 1:
            csv_files[0].rename(TARGET_CSV)
        elif len(csv_files) > 1:
            # Prefer file named Leads.csv (case-insensitive)
            for f in csv_files:
                if f.stem.lower() == "leads":
                    f.rename(TARGET_CSV)
                    break
            else:
                print(f"Multiple CSV files found: {[f.name for f in csv_files]}")
                print(f"Please rename the correct file to: {TARGET_CSV}")
                return False
        else:
            zip_files = list(RAW_DATA_DIR.glob("*.zip"))
            for zf in zip_files:
                extracted = _extract_csv_from_zip(zf)
                if extracted and extracted.name.lower() != "leads.csv":
                    extracted.rename(TARGET_CSV)
                elif extracted:
                    break

    if TARGET_CSV.exists():
        print(f"Success! Dataset saved to: {TARGET_CSV}")
        return True

    _print_manual_instructions()
    return False


def main() -> int:
    success = download_dataset()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
