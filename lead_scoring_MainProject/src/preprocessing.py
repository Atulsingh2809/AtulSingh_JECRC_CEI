"""
Data cleaning and preprocessing for X Education lead scoring.

Supports two model variants:
  - Model A: pre-sales-contact features only (no data leakage) — primary deployment
  - Model B: all features including sales-assigned fields — reference/comparison only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BINARY_YES_NO_COLS,
    CONSTANT_COLS,
    DEFAULT_CATEGORICAL_FILL,
    DEFAULT_MISSING_CATEGORY,
    HIGH_CARDINALITY_COLS,
    HIGH_NULL_THRESHOLD,
    ID_COLS,
    LEAKAGE_COLS,
    LOW_CARDINALITY_MAX,
    NUMERIC_COLS,
    PROSPECT_ID_COL,
    TARGET_COL,
)
from src.feature_engineering import FeatureEngineer
from src.utils import find_constant_columns, get_null_fraction, replace_select_with_nan, yes_no_to_binary

logger = logging.getLogger(__name__)

ModelVariant = Literal["A", "B"]


@dataclass
class PreprocessorConfig:
    """Configurable preprocessing behaviour."""

    variant: ModelVariant = "A"
    high_null_threshold: float = HIGH_NULL_THRESHOLD
    categorical_fill: str = DEFAULT_CATEGORICAL_FILL
    missing_category: str = DEFAULT_MISSING_CATEGORY
    low_cardinality_max: int = LOW_CARDINALITY_MAX
    drop_constant_cols: bool = True
    constant_cols: list[str] = field(default_factory=lambda: list(CONSTANT_COLS))


class LeadPreprocessor:
    """
    Fit/transform preprocessor for lead scoring features.

    Usage
    -----
    >>> prep = LeadPreprocessor(variant="A")
    >>> X, y, ids = prep.fit_transform(df)
    >>> X_new = prep.transform(df_new)
    """

    def __init__(self, config: PreprocessorConfig | None = None, variant: ModelVariant = "A"):
        if config is None:
            config = PreprocessorConfig(variant=variant)
        elif config.variant != variant and variant != "A":
            config.variant = variant
        self.config = config

        # Fitted state
        self.feature_columns_: list[str] = []
        self.dropped_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.categorical_modes_: dict[str, str] = {}
        self.high_null_columns_: set[str] = set()
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.one_hot_columns_: list[str] = []
        self.one_hot_categories_: dict[str, list[str]] = {}
        self.binary_columns_: list[str] = []
        self.frequency_encoded_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.is_fitted_: bool = False
        self.feature_engineer_ = FeatureEngineer()

    @property
    def variant(self) -> ModelVariant:
        return self.config.variant

    def _columns_to_exclude(self, df: pd.DataFrame) -> list[str]:
        """Columns removed before feature engineering."""
        exclude = set(ID_COLS + [TARGET_COL, PROSPECT_ID_COL])
        if self.config.variant == "A":
            exclude.update(LEAKAGE_COLS)
        return [c for c in exclude if c in df.columns]

    def _apply_feature_engineering(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        work = replace_select_with_nan(df)
        if fit:
            self.feature_engineer_.fit(work)
        return self.feature_engineer_.transform(work)

    def _get_feature_frame(self, df: pd.DataFrame, fit_fe: bool = False) -> pd.DataFrame:
        """Sentinel cleaning, feature engineering, drop IDs/target/leakage."""
        work = self._apply_feature_engineering(df, fit=fit_fe)
        exclude = self._columns_to_exclude(work)
        features = work.drop(columns=exclude, errors="ignore").copy()
        return features

    def _detect_constant_columns(self, df: pd.DataFrame) -> list[str]:
        """Known constant cols plus any newly detected single-value columns."""
        known = [c for c in self.config.constant_cols if c in df.columns]
        detected = find_constant_columns(df, exclude=[TARGET_COL, PROSPECT_ID_COL])
        combined = list(dict.fromkeys(known + detected))
        return combined

    def _split_column_types(self, df: pd.DataFrame) -> None:
        """Classify columns into numeric, binary, frequency-encoded, and one-hot."""
        self.binary_columns_ = [c for c in BINARY_YES_NO_COLS if c in df.columns]
        engineered = self.feature_engineer_.engineered_numeric_cols
        self.numeric_columns_ = [
            c
            for c in list(NUMERIC_COLS) + engineered
            if c in df.columns and c not in self.config.constant_cols
        ]

        remaining = [
            c
            for c in df.columns
            if c not in self.binary_columns_ + self.numeric_columns_
        ]

        self.frequency_encoded_columns_ = [
            c for c in HIGH_CARDINALITY_COLS if c in remaining
        ]
        self.one_hot_columns_ = [
            c
            for c in remaining
            if c not in self.frequency_encoded_columns_
            and df[c].nunique(dropna=True) <= self.config.low_cardinality_max
        ]

        # Any leftover high-cardinality columns get frequency encoding
        for c in remaining:
            if (
                c not in self.frequency_encoded_columns_
                and c not in self.one_hot_columns_
                and df[c].nunique(dropna=True) > self.config.low_cardinality_max
            ):
                self.frequency_encoded_columns_.append(c)

    def _fit_imputers(self, df: pd.DataFrame) -> None:
        """Learn median/mode/Missing-category imputation values."""
        self.high_null_columns_ = set()
        for col in df.columns:
            null_frac = get_null_fraction(df, col)
            if null_frac > self.config.high_null_threshold:
                self.high_null_columns_.add(col)

        for col in self.numeric_columns_:
            if col in df.columns:
                self.numeric_medians_[col] = float(df[col].median())

        cat_cols = self.one_hot_columns_ + self.frequency_encoded_columns_
        for col in cat_cols:
            if col in self.high_null_columns_:
                continue
            mode = df[col].mode(dropna=True)
            self.categorical_modes_[col] = (
                str(mode.iloc[0]) if len(mode) else self.config.categorical_fill
            )

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation rules."""
        out = df.copy()
        for col in self.numeric_columns_:
            if col in out.columns:
                out[col] = out[col].fillna(self.numeric_medians_.get(col, 0.0))

        cat_cols = self.one_hot_columns_ + self.frequency_encoded_columns_
        for col in cat_cols:
            if col not in out.columns:
                continue
            if col in self.high_null_columns_:
                out[col] = out[col].fillna(self.config.missing_category)
            else:
                out[col] = out[col].fillna(
                    self.categorical_modes_.get(col, self.config.categorical_fill)
                )
        return out

    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.binary_columns_:
            if col in out.columns:
                out[col] = yes_no_to_binary(out[col]).fillna(0).astype(int)
        return out

    def _fit_frequency_encoders(self, df: pd.DataFrame) -> None:
        for col in self.frequency_encoded_columns_:
            if col not in df.columns:
                continue
            counts = df[col].value_counts(normalize=True)
            self.freq_maps_[col] = counts.to_dict()

    def _apply_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.frequency_encoded_columns_:
            if col not in out.columns:
                continue
            mapping = self.freq_maps_.get(col, {})
            global_mean = float(np.mean(list(mapping.values()))) if mapping else 0.0
            out[col] = out[col].map(mapping).fillna(global_mean)
        return out

    def _fit_one_hot(self, df: pd.DataFrame) -> None:
        for col in self.one_hot_columns_:
            if col not in df.columns:
                continue
            self.one_hot_categories_[col] = sorted(
                df[col].dropna().astype(str).unique().tolist()
            )

    def _apply_one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [df.drop(columns=self.one_hot_columns_, errors="ignore")]
        for col in self.one_hot_columns_:
            if col not in df.columns:
                continue
            categories = self.one_hot_categories_.get(col, [])
            col_series = df[col].astype(str)
            for cat in categories[1:]:  # drop_first: skip first category
                new_name = f"{col}_{cat}"
                parts.append((col_series == cat).astype(int).rename(new_name))
        return pd.concat(parts, axis=1)

    def fit(self, df: pd.DataFrame) -> LeadPreprocessor:
        """Learn preprocessing parameters from training data."""
        features = self._get_feature_frame(df, fit_fe=True)

        if self.config.drop_constant_cols:
            self.dropped_columns_ = self._detect_constant_columns(features)
            features = features.drop(columns=self.dropped_columns_, errors="ignore")

        self._split_column_types(features)
        self._fit_imputers(features)
        self._fit_frequency_encoders(features)
        self._fit_one_hot(features)

        transformed = self._transform_features(features)
        self.feature_columns_ = transformed.columns.tolist()
        self.is_fitted_ = True

        logger.info(
            "Fitted preprocessor variant=%s: %d features, dropped %d constant cols, "
            "excluded %d leakage cols",
            self.config.variant,
            len(self.feature_columns_),
            len(self.dropped_columns_),
            len(LEAKAGE_COLS) if self.config.variant == "A" else 0,
        )
        return self

    def _transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply all encoding steps to a feature-only dataframe."""
        if self.config.drop_constant_cols and self.dropped_columns_:
            features = features.drop(columns=self.dropped_columns_, errors="ignore")

        features = self._impute(features)
        features = self._encode_binary(features)
        features = self._apply_frequency_encoding(features)
        features = self._apply_one_hot(features)

        # Keep only numeric columns for modeling
        for col in features.columns:
            if features[col].dtype == object:
                features[col] = pd.to_numeric(features[col], errors="coerce")
        features = features.fillna(0.0)

        if self.is_fitted_:
            features = features.reindex(columns=self.feature_columns_, fill_value=0.0)

        return features

    def transform(
        self, df: pd.DataFrame, return_ids: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """
        Transform raw leads into model-ready features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or partially cleaned lead data.
        return_ids : bool
            If True, also return Prospect ID series.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, pd.Series)
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        ids = df[PROSPECT_ID_COL].copy() if PROSPECT_ID_COL in df.columns else None
        features = self._get_feature_frame(df)
        X = self._transform_features(features)

        if return_ids and ids is not None:
            return X, ids
        return X

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Fit preprocessor and return (X, y, prospect_ids).

        Rows with missing target are dropped.
        """
        work = df.copy()
        if TARGET_COL not in work.columns:
            raise ValueError(f"Target column '{TARGET_COL}' required for fit_transform.")

        mask = work[TARGET_COL].notna()
        work = work.loc[mask]

        y = work[TARGET_COL].astype(int)
        ids = work[PROSPECT_ID_COL].copy() if PROSPECT_ID_COL in work.columns else pd.Series(
            index=work.index, dtype=str
        )

        self.fit(work)
        X = self.transform(work)
        return X, y, ids

    def save(self, path: str | Any) -> None:
        """Persist fitted preprocessor to disk."""
        joblib.dump(self, path)
        logger.info("Saved preprocessor to %s", path)

    @classmethod
    def load(cls, path: str | Any) -> LeadPreprocessor:
        """Load a fitted preprocessor from disk."""
        return joblib.load(path)

    def get_leakage_documentation(self) -> str:
        """Human-readable note on leakage columns for this variant."""
        if self.config.variant == "A":
            return (
                "Model A excludes post-contact sales-assigned columns to avoid data leakage: "
                + ", ".join(LEAKAGE_COLS)
            )
        return (
            "Model B includes all features including sales-assigned fields "
            "(Tags, Lead Quality, Last Notable Activity, Asymmetrique indices/scores). "
            "Use for reference only — not for scoring brand-new inbound leads."
        )


def build_preprocessor(variant: ModelVariant = "A") -> LeadPreprocessor:
    """Factory for Model A or Model B preprocessor."""
    return LeadPreprocessor(config=PreprocessorConfig(variant=variant))


def preprocess_and_save(
    df: pd.DataFrame,
    variant: ModelVariant = "A",
    output_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, LeadPreprocessor]:
    """
    Full preprocessing pipeline: fit, transform, optionally save processed CSV.

    Returns (X, y, prospect_ids, fitted_preprocessor).
    """
    prep = build_preprocessor(variant=variant)
    X, y, ids = prep.fit_transform(df)

    if output_path:
        processed = X.copy()
        processed[PROSPECT_ID_COL] = ids.values
        processed[TARGET_COL] = y.values
        processed.to_csv(output_path, index=False)
        logger.info("Saved processed data to %s", output_path)

    return X, y, ids, prep


if __name__ == "__main__":
    from src.config import PROCESSED_A_PATH, PROCESSED_B_PATH, PREPROCESSOR_A_PATH, PREPROCESSOR_B_PATH
    from src.data_loader import load_raw_data
    from src.utils import setup_logging

    setup_logging()
    raw = load_raw_data()

    for variant, proc_path, prep_path in [
        ("A", PROCESSED_A_PATH, PREPROCESSOR_A_PATH),
        ("B", PROCESSED_B_PATH, PREPROCESSOR_B_PATH),
    ]:
        X, y, ids, prep = preprocess_and_save(raw, variant=variant, output_path=proc_path)
        prep.save(prep_path)
        print(
            f"Model {variant}: X={X.shape}, positive_rate={y.mean():.3f}, "
            f"features={len(prep.feature_columns_)}"
        )
        print(f"  {prep.get_leakage_documentation()}")
