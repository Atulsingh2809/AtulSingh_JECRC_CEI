"""
Train and compare machine learning models for Lead Scoring.
Supported variants:
- Model A: Pre-sales-contact features only (default).
- Model B: Includes sales-assigned features (for comparison).
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Try to import XGBoost and LightGBM if installed
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from src.config import (
    PROCESSED_A_PATH,
    PROCESSED_B_PATH,
    BEST_MODEL_PATH,
    MODELS_DIR,
    TARGET_COL,
    PROSPECT_ID_COL,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_COMPARISON_PATH
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def load_processed_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed CSV and split into features X and target y."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run preprocessing first: python -m src.preprocessing"
        )
    logger.info("Loading processed data from %s", path)
    df = pd.read_csv(path)
    
    # Drop identifier and target columns to get features
    X = df.drop(columns=[PROSPECT_ID_COL, TARGET_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)
    
    return X, y


def get_candidate_models() -> dict[str, any]:
    """Return dictionary of candidate classifiers."""
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, 
            n_estimators=150, 
            max_depth=10, 
            class_weight="balanced"
        )
    }
    
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=5,
            eval_metric="logloss",
            use_label_encoder=False
        )
        
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=5,
            verbosity=-1
        )
        
    return models


def train_and_select_best(variant: str, data_path: Path) -> tuple[any, list[dict[str, any]]]:
    """
    Evaluate candidate models on cross-validation and train the best model.
    """
    X, y = load_processed_data(data_path)
    
    # Stratified train/test split to preserve held-out test data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Variant %s split: Train shape: %s, Test shape: %s, Train target balance: %s",
        variant, X_train.shape, X_test.shape, y_train.value_counts(normalize=True).to_dict()
    )
    
    models = get_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    comparison_results = []
    best_model_name = None
    best_roc_auc = -1.0
    best_model_cv = None
    
    for name, clf in models.items():
        logger.info("Evaluating %s via 5-fold Cross-Validation...", name)
        scores = cross_validate(
            clf, X_train, y_train, 
            cv=cv, 
            scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
            n_jobs=-1
        )
        
        mean_roc_auc = float(np.mean(scores["test_roc_auc"]))
        mean_f1 = float(np.mean(scores["test_f1"]))
        mean_prec = float(np.mean(scores["test_precision"]))
        mean_rec = float(np.mean(scores["test_recall"]))
        mean_acc = float(np.mean(scores["test_accuracy"]))
        
        logger.info(
            "%s results: ROC AUC=%0.4f, F1=%0.4f, Precision=%0.4f, Recall=%0.4f, Accuracy=%0.4f",
            name, mean_roc_auc, mean_f1, mean_prec, mean_rec, mean_acc
        )
        
        comparison_results.append({
            "variant": variant,
            "model_name": name,
            "cv_roc_auc": mean_roc_auc,
            "cv_f1": mean_f1,
            "cv_precision": mean_prec,
            "cv_recall": mean_rec,
            "cv_accuracy": mean_acc
        })
        
        if mean_roc_auc > best_roc_auc:
            best_roc_auc = mean_roc_auc
            best_model_name = name
            best_model_cv = clf
            
    logger.info("--> Selected best model for Variant %s: %s (CV ROC AUC = %0.4f)", variant, best_model_name, best_roc_auc)
    
    # Train the selected model on the FULL training split
    logger.info("Fitting best model %s on train split...", best_model_name)
    best_model = best_model_cv
    best_model.fit(X_train, y_train)
    
    return best_model, comparison_results


def main() -> None:
    setup_logging()
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    variants = [
        ("A", PROCESSED_A_PATH, MODELS_DIR / "model_a.pkl"),
        ("B", PROCESSED_B_PATH, MODELS_DIR / "model_b.pkl")
    ]
    
    all_comparison_rows = []
    
    for variant, data_path, model_save_path in variants:
        logger.info("=" * 60)
        logger.info("TRAINING MODEL VARIANT %s", variant)
        logger.info("=" * 60)
        
        best_clf, metrics_list = train_and_select_best(variant, data_path)
        all_comparison_rows.extend(metrics_list)
        
        # Save fitted model
        joblib.dump(best_clf, model_save_path)
        logger.info("Saved Variant %s model to %s", variant, model_save_path)
        
        # If it's Model A (the deployable one), also save it to best_model.pkl
        if variant == "A":
            joblib.dump(best_clf, BEST_MODEL_PATH)
            logger.info("Saved Variant A model as the deployable best model at %s", BEST_MODEL_PATH)
            
    # Save comparison dataframe to reports
    comp_df = pd.DataFrame(all_comparison_rows)
    comp_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info("Saved cross-validation model comparison report to %s", MODEL_COMPARISON_PATH)


if __name__ == "__main__":
    main()
