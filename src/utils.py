"""Shared utility functions for the lead scoring pipeline."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src.config import SELECT_SENTINEL

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for CLI scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def replace_select_with_nan(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Replace the literal 'Select' sentinel with NaN across categorical columns.

    In the X Education form, 'Select' means the user left a dropdown unselected.
    """
    out = df.copy()
    cols = columns if columns is not None else out.select_dtypes(include=["object"]).columns
    for col in cols:
        if col in out.columns:
            out[col] = out[col].replace(SELECT_SENTINEL, pd.NA)
    return out


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    """Map Yes/No strings to 1/0; leave numeric values unchanged."""
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.map(mapping)


def get_null_fraction(df: pd.DataFrame, column: str) -> float:
    """Return fraction of null values in a column."""
    return float(df[column].isna().mean())


def find_constant_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    """Return columns with a single unique non-null value."""
    exclude_set = set(exclude or [])
    constant = []
    for col in df.columns:
        if col in exclude_set:
            continue
        if df[col].nunique(dropna=True) <= 1:
            constant.append(col)
    return constant


def score_to_tier(score: int, hot: int = 70, warm: int = 40) -> str:
    """Map a 0–100 lead score to Hot / Warm / Cold tier."""
    if score >= hot:
        return "Hot Lead"
    if score >= warm:
        return "Warm Lead"
    return "Cold Lead"
