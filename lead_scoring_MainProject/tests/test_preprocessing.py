"""Unit tests for preprocessing pipeline."""

import pandas as pd
import pytest

from src.config import LEAKAGE_COLS, PROSPECT_ID_COL, TARGET_COL
from src.data_loader import DataValidationError, load_raw_data, validate_raw_data
from src.preprocessing import LeadPreprocessor, build_preprocessor
from src.utils import replace_select_with_nan, score_to_tier


@pytest.fixture
def sample_raw_df():
    """Minimal valid raw lead records for unit tests."""
    return pd.DataFrame(
        {
            PROSPECT_ID_COL: ["id-1", "id-2", "id-3"],
            "Lead Number": [1, 2, 3],
            "Lead Origin": ["API", "Landing Page Submission", "API"],
            "Lead Source": ["Organic Search", "Direct Traffic", "Olark Chat"],
            "Do Not Email": ["No", "Yes", "No"],
            "Do Not Call": ["No", "No", "Yes"],
            TARGET_COL: [1, 0, 1],
            "TotalVisits": [5, 0, 3],
            "Total Time Spent on Website": [100, 0, 50],
            "Page Views Per Visit": [2.0, 0.0, 1.5],
            "Last Activity": ["Email Opened", "Page Visited on Website", "Email Opened"],
            "Country": ["India", "Select", None],
            "Specialization": ["Select", "Finance", "Marketing"],
            "How did you hear about X Education": ["Select", "Student of Somebody", None],
            "What is your current occupation": ["Unemployed", "Student", "Working Professional"],
            "What matters most to you in choosing a course": [
                "Better Career Prospects",
                "Better Career Prospects",
                "Skill Development",
            ],
            "Search": ["No", "No", "Yes"],
            "Magazine": ["No", "No", "No"],
            "Newspaper Article": ["No", "No", "No"],
            "X Education Forums": ["No", "No", "No"],
            "Newspaper": ["No", "No", "No"],
            "Digital Advertisement": ["No", "No", "No"],
            "Through Recommendations": ["No", "Yes", "No"],
            "Receive More Updates About Our Courses": ["No", "No", "No"],
            "Tags": [None, "Interested in other courses", None],
            "Lead Quality": [None, "Low in Relevance", None],
            "Update me on Supply Chain Content": ["No", "No", "No"],
            "Get updates on DM Content": ["No", "No", "No"],
            "Lead Profile": ["Select", "Potential Lead", "Select"],
            "City": ["Mumbai", "Select", "Delhi"],
            "Asymmetrique Activity Index": [None, "02.Medium", None],
            "Asymmetrique Profile Index": [None, "02.Medium", None],
            "Asymmetrique Activity Score": [None, 15, None],
            "Asymmetrique Profile Score": [None, 15, None],
            "I agree to pay the amount through cheque": ["No", "No", "No"],
            "A free copy of Mastering The Interview": ["No", "No", "No"],
            "Last Notable Activity": [None, "Email Opened", None],
        }
    )


class TestSelectHandling:
    def test_select_replaced_with_nan(self, sample_raw_df):
        cleaned = replace_select_with_nan(sample_raw_df)
        assert cleaned.loc[0, "Country"] == "India"
        assert pd.isna(cleaned.loc[1, "Country"])
        assert pd.isna(cleaned.loc[0, "Specialization"])
        assert cleaned.loc[1, "Specialization"] == "Finance"


class TestDataLoader:
    def test_validate_missing_columns_raises(self):
        df = pd.DataFrame({PROSPECT_ID_COL: ["a"], TARGET_COL: [1]})
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_raw_data(df)

    def test_load_real_dataset_if_present(self):
        try:
            df = load_raw_data()
        except FileNotFoundError:
            pytest.skip("Leads.csv not available")
        assert len(df) == 9240
        assert TARGET_COL in df.columns


class TestPreprocessor:
    def test_model_a_excludes_leakage_columns(self, sample_raw_df):
        prep = build_preprocessor(variant="A")
        X, y, ids = prep.fit_transform(sample_raw_df)
        for col in LEAKAGE_COLS:
            assert not any(col in c for c in X.columns)
        assert len(y) == 3
        assert list(ids) == ["id-1", "id-2", "id-3"]

    def test_model_b_has_more_or_equal_features(self, sample_raw_df):
        prep_a = build_preprocessor(variant="A")
        prep_b = build_preprocessor(variant="B")
        X_a, _, _ = prep_a.fit_transform(sample_raw_df)
        X_b, _, _ = prep_b.fit_transform(sample_raw_df)
        assert X_b.shape[1] >= X_a.shape[1]

    def test_transform_before_fit_raises(self, sample_raw_df):
        prep = build_preprocessor(variant="A")
        with pytest.raises(RuntimeError, match="must be fitted"):
            prep.transform(sample_raw_df)

    def test_output_is_numeric(self, sample_raw_df):
        prep = build_preprocessor(variant="A")
        X, _, _ = prep.fit_transform(sample_raw_df)
        assert X.dtypes.apply(lambda t: t.kind in "iuf").all()

    def test_constant_columns_dropped(self, sample_raw_df):
        prep = build_preprocessor(variant="A")
        X, _, _ = prep.fit_transform(sample_raw_df)
        assert "Magazine" not in X.columns
        assert not any("Magazine" in c for c in X.columns)

    def test_save_and_load_roundtrip(self, sample_raw_df, tmp_path):
        prep = build_preprocessor(variant="A")
        prep.fit(sample_raw_df)
        path = tmp_path / "prep.joblib"
        prep.save(path)
        loaded = LeadPreprocessor.load(path)
        X1 = prep.transform(sample_raw_df)
        X2 = loaded.transform(sample_raw_df)
        pd.testing.assert_frame_equal(X1, X2)


class TestScoreTier:
    def test_tier_boundaries(self):
        assert score_to_tier(85) == "Hot Lead"
        assert score_to_tier(70) == "Hot Lead"
        assert score_to_tier(55) == "Warm Lead"
        assert score_to_tier(40) == "Warm Lead"
        assert score_to_tier(20) == "Cold Lead"
