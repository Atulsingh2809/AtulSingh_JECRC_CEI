"""
Lead scoring and prediction module.
Loads trained models and preprocessors to score single leads or batch datasets.
Provides explainability (positive/negative contributors) for single predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_A_PATH,
    PREPROCESSOR_B_PATH,
    MODELS_DIR,
    HOT_THRESHOLD,
    WARM_THRESHOLD
)
from src.utils import score_to_tier
from src.preprocessing import LeadPreprocessor

logger = logging.getLogger(__name__)

# Complete list of 37 raw columns from X Education Lead dataset
RAW_LEAD_COLUMNS = [
    "Prospect ID", "Lead Number", "Lead Origin", "Lead Source", "Do Not Email",
    "Do Not Call", "Converted", "TotalVisits", "Total Time Spent on Website",
    "Page Views Per Visit", "Last Activity", "Country", "Specialization",
    "How did you hear about X Education", "What is your current occupation",
    "What matters most to you in choosing a course", "Search", "Magazine",
    "Newspaper Article", "X Education Forums", "Newspaper", "Digital Advertisement",
    "Through Recommendations", "Receive More Updates About Our Courses", "Tags",
    "Lead Quality", "Update me on Supply Chain Content", "Get updates on DM Content",
    "Lead Profile", "City", "Asymmetrique Activity Index", "Asymmetrique Profile Index",
    "Asymmetrique Activity Score", "Asymmetrique Profile Score",
    "I agree to pay the amount through cheque", "A free copy of Mastering The Interview",
    "Last Notable Activity"
]


def clean_feature_name(name: str) -> str:
    """Format technical feature names into user-friendly UI strings."""
    # Handle engineered flags
    if name == "engagement_score":
        return "Engagement Score (Visits & Time)"
    if name == "has_specialization":
        return "Has Specialization Listed"
    if name == "has_occupation":
        return "Has Occupation Listed"
    if name == "is_from_top_source":
        return "From High-Converting Lead Source"
    
    # Handle standard columns
    if "_" in name:
        parts = name.split("_")
        category = parts[0]
        value = "_".join(parts[1:])
        
        if "occupation" in category.lower():
            category = "Occupation"
        elif "lead origin" in category.lower():
            category = "Lead Origin"
        elif "lead source" in category.lower():
            category = "Lead Source"
        elif "specialization" in category.lower():
            category = "Specialization"
        elif "city" in category.lower():
            category = "City"
        elif "last activity" in category.lower():
            category = "Last Activity"
            
        return f"{category}: {value}"
    
    # Capitalize spacing for readable presentation
    return name.replace("_", " ").title()


class LeadPredictor:
    """
    Predictor for scoring leads and explaining predictions.
    Defaults to Model A (pre-sales features only).
    """

    def __init__(self, variant: str = "A"):
        self.variant = variant.upper()
        
        # Determine paths
        if self.variant == "A":
            self.model_path = BEST_MODEL_PATH
            self.preprocessor_path = PREPROCESSOR_A_PATH
        else:
            self.model_path = MODELS_DIR / "model_b.pkl"
            self.preprocessor_path = PREPROCESSOR_B_PATH
            
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train models first.")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}. Preprocess data first.")
            
        logger.info("Loading LeadPredictor variant=%s...", self.variant)
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        
    def predict_single(self, lead_dict: dict[str, any]) -> dict[str, any]:
        """
        Score a single lead and explain the key scoring factors.
        
        Parameters
        ----------
        lead_dict : dict
            Dictionary containing lead features.
            
        Returns
        -------
        dict
            Contains probability, score, tier, and explainability factors.
        """
        # Convert lead dict to DataFrame and align columns
        df = pd.DataFrame([lead_dict])
        for col in RAW_LEAD_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[RAW_LEAD_COLUMNS]
        
        # Transform lead
        X = self.preprocessor.transform(df)
        
        # Run prediction
        prob = float(self.model.predict_proba(X)[0, 1])
        score = int(round(prob * 100))
        tier = score_to_tier(score, hot=HOT_THRESHOLD, warm=WARM_THRESHOLD)
        
        # Compute feature contributions using model-agnostic feature perturbation (LOO)
        explanations = self._explain_prediction(X, prob)
        
        return {
            "variant": self.variant,
            "probability": prob,
            "lead_score": score,
            "tier": tier,
            "factors": explanations
        }
        
    def _explain_prediction(self, X_sample: pd.DataFrame, base_prob: float) -> dict[str, list[dict[str, any]]]:
        """Compute feature contributions using perturbation (Leave-One-Out)."""
        contributions = []
        
        for col in X_sample.columns:
            # Determine baseline reference value for this feature
            if col in self.preprocessor.numeric_columns_:
                ref_val = float(self.preprocessor.numeric_medians_.get(col, 0.0))
            else:
                ref_val = 0.0  # reference for low-cardinality, binary, and one-hot
                
            val = float(X_sample.loc[0, col])
            if abs(val - ref_val) < 1e-4:
                continue
                
            # Perturb feature
            X_perturbed = X_sample.copy()
            X_perturbed.loc[0, col] = ref_val
            
            perturbed_prob = float(self.model.predict_proba(X_perturbed)[0, 1])
            change = base_prob - perturbed_prob
            
            # Record significant changes
            if abs(change) >= 0.005:  # threshold: >= 0.5% probability change
                contributions.append({
                    "feature": col,
                    "clean_name": clean_feature_name(col),
                    "value": val,
                    "change": change
                })
                
        # Sort contributions
        positive_factors = [c for c in contributions if c["change"] > 0]
        positive_factors = sorted(positive_factors, key=lambda x: x["change"], reverse=True)[:3]
        
        negative_factors = [c for c in contributions if c["change"] < 0]
        negative_factors = sorted(negative_factors, key=lambda x: x["change"])[:3]
        
        return {
            "positive": positive_factors,
            "negative": negative_factors
        }
        
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame batch of leads.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame matching raw lead schema.
            
        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with added score and tier columns.
        """
        # Align columns
        work = df.copy()
        for col in RAW_LEAD_COLUMNS:
            if col not in work.columns:
                work[col] = np.nan
        
        # Preprocess features
        X = self.preprocessor.transform(work)
        
        # Score
        probs = self.model.predict_proba(X)[:, 1]
        scores = np.round(probs * 100).astype(int)
        tiers = [score_to_tier(s, hot=HOT_THRESHOLD, warm=WARM_THRESHOLD) for s in scores]
        
        out = df.copy()
        out["Lead Score"] = scores
        out["Score Tier"] = tiers
        
        return out


if __name__ == "__main__":
    from src.utils import setup_logging
    
    setup_logging()
    
    # Quick test prediction
    try:
        predictor = LeadPredictor(variant="A")
        
        test_lead = {
            "Lead Origin": "Landing Page Submission",
            "Lead Source": "Direct Traffic",
            "TotalVisits": 8.0,
            "Total Time Spent on Website": 1200.0,
            "Page Views Per Visit": 4.0,
            "Specialization": "Marketing Management",
            "What is your current occupation": "Working Professional",
            "Do Not Email": "No"
        }
        
        result = predictor.predict_single(test_lead)
        print("Test Single Prediction:")
        print(f"  Score: {result['lead_score']}")
        print(f"  Tier:  {result['tier']}")
        print("  Positive factors:")
        for factor in result['factors']['positive']:
            print(f"    - {factor['clean_name']}: +{factor['change']*100:.1f}%")
        print("  Negative factors:")
        for factor in result['factors']['negative']:
            print(f"    - {factor['clean_name']}: {factor['change']*100:.1f}%")
            
    except Exception as e:
        logger.exception("Prediction test failed: %s", e)
