"""Feature engineering for X Education lead scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    SELECT_SENTINEL,
    TARGET_COL,
)
from src.utils import replace_select_with_nan

logger = logging.getLogger(__name__)

ENGAGEMENT_SCORE_COL = "engagement_score"
HAS_SPECIALIZATION_COL = "has_specialization"
HAS_OCCUPATION_COL = "has_occupation"
IS_TOP_SOURCE_COL = "is_from_top_source"

SPECIALIZATION_COL = "Specialization"
OCCUPATION_COL = "What is your current occupation"
LEAD_SOURCE_COL = "Lead Source"

ENGAGEMENT_NUMERIC_COLS = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
]
ENGAGEMENT_WEIGHTS = {
    "TotalVisits": 0.30,
    "Total Time Spent on Website": 0.50,
    "Page Views Per Visit": 0.20,
}

# Categorical columns where rare levels (<1%) are grouped into "Other"
DEFAULT_RARE_GROUP_COLS = [
    "Lead Source",
    "Lead Origin",
    "Last Activity",
    "Country",
    "Specialization",
    "City",
    "What is your current occupation",
]

RARE_LEVEL_THRESHOLD = 0.01
TOP_SOURCE_MIN_COUNT = 50
TOP_SOURCE_MIN_RATE = 0.50


@dataclass
class FeatureEngineerConfig:
    """Configurable feature engineering parameters."""

    rare_level_threshold: float = RARE_LEVEL_THRESHOLD
    rare_group_cols: list[str] = field(default_factory=lambda: list(DEFAULT_RARE_GROUP_COLS))
    engagement_weights: dict[str, float] = field(
        default_factory=lambda: dict(ENGAGEMENT_WEIGHTS)
    )
    top_source_min_count: int = TOP_SOURCE_MIN_COUNT
    top_source_min_rate: float = TOP_SOURCE_MIN_RATE


class FeatureEngineer:
    """
    Fit/transform feature engineering on raw lead data (before encoding).

    Adds engagement score, binary flags, and groups rare categorical levels.
    """

    def __init__(self, config: FeatureEngineerConfig | None = None):
        self.config = config or FeatureEngineerConfig()
        self.engagement_bounds_: dict[str, tuple[float, float]] = {}
        self.top_sources_: set[str] = set()
        self.rare_level_maps_: dict[str, set[str]] = {}
        self.is_fitted_: bool = False

    def _normalize_series(self, series: pd.Series, col: str) -> pd.Series:
        lo, hi = self.engagement_bounds_.get(col, (0.0, 1.0))
        if hi <= lo:
            return pd.Series(0.0, index=series.index)
        return (series - lo) / (hi - lo)

    def _compute_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        for col, weight in self.config.engagement_weights.items():
            if col in df.columns:
                normalized = self._normalize_series(df[col].fillna(0), col)
                score = score + weight * normalized
        return score.clip(0.0, 1.0)

    def _fit_engagement_bounds(self, df: pd.DataFrame) -> None:
        for col in ENGAGEMENT_NUMERIC_COLS:
            if col not in df.columns:
                continue
            values = df[col].fillna(0)
            self.engagement_bounds_[col] = (float(values.min()), float(values.max()))

    def _fit_top_sources(self, df: pd.DataFrame) -> None:
        if LEAD_SOURCE_COL not in df.columns or TARGET_COL not in df.columns:
            self.top_sources_ = set()
            return

        stats = (
            df.groupby(LEAD_SOURCE_COL)[TARGET_COL]
            .agg(["mean", "count"])
            .reset_index()
        )
        mask = (stats["count"] >= self.config.top_source_min_count) & (
            stats["mean"] >= self.config.top_source_min_rate
        )
        self.top_sources_ = set(stats.loc[mask, LEAD_SOURCE_COL].astype(str))
        logger.info("Top converting lead sources (fit): %s", sorted(self.top_sources_))

    def _fit_rare_levels(self, df: pd.DataFrame) -> None:
        n = len(df)
        for col in self.config.rare_group_cols:
            if col not in df.columns:
                continue
            freq = df[col].value_counts(normalize=True, dropna=True)
            rare = set(freq[freq < self.config.rare_level_threshold].index.astype(str))
            self.rare_level_maps_[col] = rare

    def _group_rare_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, rare_levels in self.rare_level_maps_.items():
            if col not in out.columns:
                continue
            series = out[col].copy()
            non_null_mask = series.notna()
            if non_null_mask.any():
                str_vals = series.loc[non_null_mask].astype(str)
                rare_mask = str_vals.isin(rare_levels)
                rare_indices = rare_mask[rare_mask].index
                series.loc[rare_indices] = "Other"
            out[col] = series
        return out

    def _add_binary_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if SPECIALIZATION_COL in out.columns:
            spec = out[SPECIALIZATION_COL]
            out[HAS_SPECIALIZATION_COL] = (
                spec.notna() & (spec.fillna("").astype(str) != SELECT_SENTINEL)
            ).astype(int)
        else:
            out[HAS_SPECIALIZATION_COL] = 0

        if OCCUPATION_COL in out.columns:
            occ = out[OCCUPATION_COL]
            out[HAS_OCCUPATION_COL] = (
                occ.notna() & (occ.fillna("").astype(str) != SELECT_SENTINEL)
            ).astype(int)
        else:
            out[HAS_OCCUPATION_COL] = 0

        if LEAD_SOURCE_COL in out.columns:
            out[IS_TOP_SOURCE_COL] = (
                out[LEAD_SOURCE_COL].astype(str).isin(self.top_sources_)
            ).astype(int)
        else:
            out[IS_TOP_SOURCE_COL] = 0

        return out

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Learn normalization bounds, top sources, and rare level mappings."""
        work = replace_select_with_nan(df)
        self._fit_engagement_bounds(work)
        self._fit_top_sources(work)
        self._fit_rare_levels(work)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        if not self.is_fitted_:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")

        work = replace_select_with_nan(df.copy())
        work = self._group_rare_levels(work)
        work = self._add_binary_flags(work)
        work[ENGAGEMENT_SCORE_COL] = self._compute_engagement_score(work)
        return work

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def engineered_numeric_cols(self) -> list[str]:
        return [
            ENGAGEMENT_SCORE_COL,
            HAS_SPECIALIZATION_COL,
            HAS_OCCUPATION_COL,
            IS_TOP_SOURCE_COL,
        ]


def apply_feature_engineering(
    df: pd.DataFrame, engineer: FeatureEngineer | None = None, fit: bool = True
) -> tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience wrapper: fit (optional) and transform raw lead data.

    Returns (engineered_df, fitted FeatureEngineer).
    """
    if engineer is None:
        engineer = FeatureEngineer()
    if fit:
        engineer.fit(df)
    return engineer.transform(df), engineer
