"""Load and validate the raw Leads dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import PROSPECT_ID_COL, RAW_CSV_PATH, REQUIRED_COLUMNS, TARGET_COL
from src.utils import replace_select_with_nan

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when raw data fails validation checks."""


def load_raw_data(path: Path | str | None = None, replace_select: bool = True) -> pd.DataFrame:
    """
    Load Leads.csv from disk.

    Parameters
    ----------
    path : Path or str, optional
        CSV path. Defaults to config.RAW_CSV_PATH.
    replace_select : bool
        If True, replace 'Select' with NaN immediately after load.

    Returns
    -------
    pd.DataFrame
        Raw (optionally sentinel-cleaned) lead records.
    """
    csv_path = Path(path) if path is not None else RAW_CSV_PATH

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download from Kaggle and place Leads.csv in data/raw/, "
            "or run: python scripts/download_data.py"
        )

    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)
    validate_raw_data(df)

    if replace_select:
        df = replace_select_with_nan(df)

    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Validate schema and basic data quality of the raw dataset.

    Raises
    ------
    DataValidationError
        If required columns are missing or target is invalid.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise DataValidationError("Dataset is empty.")

    if df[PROSPECT_ID_COL].duplicated().any():
        n_dup = int(df[PROSPECT_ID_COL].duplicated().sum())
        logger.warning("%d duplicate Prospect IDs found", n_dup)

    if TARGET_COL not in df.columns:
        raise DataValidationError(f"Target column '{TARGET_COL}' not found.")

    target_values = set(df[TARGET_COL].dropna().unique())
    allowed = {0, 1, "0", "1", 0.0, 1.0}
    if not target_values.issubset(allowed):
        raise DataValidationError(
            f"Target '{TARGET_COL}' has unexpected values: {sorted(target_values)}"
        )


def get_target_series(df: pd.DataFrame) -> pd.Series:
    """Extract binary target as int (0/1)."""
    return df[TARGET_COL].astype(int)


def get_prospect_ids(df: pd.DataFrame) -> pd.Series:
    """Extract Prospect ID column for joining scores back to leads."""
    return df[PROSPECT_ID_COL].copy()
