"""Unit tests for lead prediction and scoring module."""

import pandas as pd
import pytest
from pathlib import Path

from src.config import BEST_MODEL_PATH, PREPROCESSOR_A_PATH
from src.predict import LeadPredictor, clean_feature_name


def test_clean_feature_name():
    """Test feature name formatting for user-friendly UI presentation."""
    assert clean_feature_name("engagement_score") == "Engagement Score (Visits & Time)"
    assert clean_feature_name("has_specialization") == "Has Specialization Listed"
    assert clean_feature_name("What is your current occupation_Working Professional") == "Occupation: Working Professional"
    assert clean_feature_name("Lead Source_Google") == "Lead Source: Google"
    assert clean_feature_name("Total Time Spent on Website") == "Total Time Spent On Website"


@pytest.fixture
def predictor_model_a():
    """Fixture to retrieve Model A LeadPredictor if files exist, else skip."""
    if not BEST_MODEL_PATH.is_file() or not PREPROCESSOR_A_PATH.is_file():
        pytest.skip("Trained Model A or Preprocessor files not found. Skipping prediction tests.")
    return LeadPredictor(variant="A")


@pytest.fixture
def mock_lead():
    """Sample raw lead inputs for prediction testing."""
    return {
        "Lead Origin": "Landing Page Submission",
        "Lead Source": "Direct Traffic",
        "TotalVisits": 5,
        "Total Time Spent on Website": 450,
        "Page Views Per Visit": 2.5,
        "What is your current occupation": "Working Professional",
        "Specialization": "Finance Management",
        "Do Not Email": "No",
        "Do Not Call": "No"
    }


class TestLeadPredictor:
    def test_predict_single_returns_correct_keys(self, predictor_model_a, mock_lead):
        res = predictor_model_a.predict_single(mock_lead)
        
        # Check output structure
        assert "variant" in res
        assert "probability" in res
        assert "lead_score" in res
        assert "tier" in res
        assert "factors" in res
        
        # Check value bounds
        assert res["variant"] == "A"
        assert 0.0 <= res["probability"] <= 1.0
        assert 0 <= res["lead_score"] <= 100
        assert res["tier"] in ["Hot Lead", "Warm Lead", "Cold Lead"]
        
        # Check explanations
        assert "positive" in res["factors"]
        assert "negative" in res["factors"]
        assert isinstance(res["factors"]["positive"], list)
        assert isinstance(res["factors"]["negative"], list)

    def test_predict_single_tier_boundaries(self, predictor_model_a):
        # High probability mock leads (e.g. Working Professional with high time spent)
        hot_lead = {
            "Lead Origin": "Lead Add Form",
            "Lead Source": "Reference",
            "TotalVisits": 10,
            "Total Time Spent on Website": 1500,
            "Page Views Per Visit": 3.0,
            "What is your current occupation": "Working Professional",
            "Specialization": "Marketing Management",
            "Do Not Email": "No",
            "Do Not Call": "No"
        }
        res = predictor_model_a.predict_single(hot_lead)
        assert res["lead_score"] >= 70
        assert res["tier"] == "Hot Lead"

        # Low probability mock leads
        cold_lead = {
            "Lead Origin": "API",
            "Lead Source": "Olark Chat",
            "TotalVisits": 0,
            "Total Time Spent on Website": 0,
            "Page Views Per Visit": 0,
            "What is your current occupation": "Unemployed",
            "Specialization": "Select",
            "Do Not Email": "Yes",
            "Do Not Call": "No"
        }
        res = predictor_model_a.predict_single(cold_lead)
        assert res["lead_score"] < 40
        assert res["tier"] == "Cold Lead"

    def test_predict_batch_adds_columns(self, predictor_model_a, mock_lead):
        df = pd.DataFrame([mock_lead] * 3)
        res_df = predictor_model_a.predict_batch(df)
        
        assert len(res_df) == 3
        assert "Lead Score" in res_df.columns
        assert "Score Tier" in res_df.columns
        
        # Verify predictions
        assert res_df["Lead Score"].between(0, 100).all()
        assert res_df["Score Tier"].isin(["Hot Lead", "Warm Lead", "Cold Lead"]).all()
