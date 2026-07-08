"""Unit tests for feature engineering."""

import pandas as pd
import pytest

from src.feature_engineering import (
    ENGAGEMENT_SCORE_COL,
    HAS_OCCUPATION_COL,
    HAS_SPECIALIZATION_COL,
    IS_TOP_SOURCE_COL,
    FeatureEngineer,
    apply_feature_engineering,
)


@pytest.fixture
def fe_sample_df():
    return pd.DataFrame(
        {
            "Lead Source": ["Reference", "Google", "Rare Source", "Reference"],
            "Specialization": ["Finance", "Select", None, "Marketing"],
            "What is your current occupation": ["Student", "Unemployed", None, "Working Professional"],
            "TotalVisits": [10, 0, 2, 5],
            "Total Time Spent on Website": [500, 0, 100, 200],
            "Page Views Per Visit": [3.0, 0.0, 2.0, 2.5],
            "Converted": [1, 0, 0, 1],
            "City": ["Mumbai"] * 4,
        }
    )


class TestFeatureEngineer:
    def test_engagement_score_in_valid_range(self, fe_sample_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(fe_sample_df)
        assert ENGAGEMENT_SCORE_COL in out.columns
        assert out[ENGAGEMENT_SCORE_COL].between(0, 1).all()

    def test_binary_flags_created(self, fe_sample_df):
        fe = FeatureEngineer()
        out = fe.fit_transform(fe_sample_df)
        assert out[HAS_SPECIALIZATION_COL].tolist() == [1, 0, 0, 1]
        assert out[HAS_OCCUPATION_COL].tolist() == [1, 1, 0, 1]

    def test_top_source_flag(self, fe_sample_df):
        fe = FeatureEngineer()
        fe.fit(fe_sample_df)
        fe.top_sources_ = {"Reference"}
        out = fe.transform(fe_sample_df)
        assert out[IS_TOP_SOURCE_COL].tolist() == [1, 0, 0, 1]

    def test_rare_levels_grouped_to_other(self):
        df = pd.DataFrame(
            {
                "Lead Source": ["Google"] * 199 + ["TinySource"],
                "TotalVisits": [1] * 200,
                "Total Time Spent on Website": [10] * 200,
                "Page Views Per Visit": [1.0] * 200,
                "Converted": [0] * 200,
            }
        )
        fe = FeatureEngineer()
        out = fe.fit_transform(df)
        assert out.loc[199, "Lead Source"] == "Other"

    def test_transform_before_fit_raises(self, fe_sample_df):
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            fe.transform(fe_sample_df)

    def test_apply_feature_engineering_wrapper(self, fe_sample_df):
        out, engineer = apply_feature_engineering(fe_sample_df)
        assert engineer.is_fitted_
        assert ENGAGEMENT_SCORE_COL in out.columns
