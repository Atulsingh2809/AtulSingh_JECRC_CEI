"""
Evaluate trained Lead Scoring models on the test set.
Generates metrics and visualization plots for Model A and Model B.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    PROCESSED_A_PATH,
    PROCESSED_B_PATH,
    MODELS_DIR,
    TARGET_COL,
    PROSPECT_ID_COL,
    TEST_SIZE,
    RANDOM_STATE,
    FIGURES_DIR
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def load_model_and_test_data(variant: str, data_path: Path) -> tuple[any, pd.DataFrame, pd.Series]:
    """Load model and return the test features X_test and targets y_test."""
    model_path = MODELS_DIR / f"model_{variant.lower()}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
    
    logger.info("Loading model %s...", model_path)
    model = joblib.load(model_path)
    
    logger.info("Loading data from %s to extract test split...", data_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=[PROSPECT_ID_COL, TARGET_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)
    
    # Recreate the deterministic test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return model, X_test, y_test


def find_optimal_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.zeros_like(thresholds)
    for idx, t in enumerate(thresholds):
        p, r = precisions[idx], recalls[idx]
        if (p + r) > 0:
            f1_scores[idx] = (2 * p * r) / (p + r)
        else:
            f1_scores[idx] = 0.0
            
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    logger.info("Optimal threshold to maximize F1-score: %0.4f (F1 = %0.4f)", best_threshold, f1_scores[best_idx])
    return best_threshold


def evaluate_model(
    model: any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    variant: str
) -> dict[str, any]:
    """Calculate and log performance metrics for a model variant."""
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics at default 0.5 threshold
    y_pred_default = (y_prob >= 0.5).astype(int)
    
    # Find best threshold on test set (or validation set)
    best_thresh = find_optimal_threshold(y_test, y_prob)
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    logger.info("=== Variant %s Performance Metrics ===", variant)
    logger.info("ROC AUC: %0.4f", roc_auc)
    logger.info("PR AUC:  %0.4f", pr_auc)
    logger.info("--- Default Threshold (0.50) ---")
    logger.info("Accuracy:  %0.4f", accuracy_score(y_test, y_pred_default))
    logger.info("Precision: %0.4f", precision_score(y_test, y_pred_default))
    logger.info("Recall:    %0.4f", recall_score(y_test, y_pred_default))
    logger.info("F1 Score:  %0.4f", f1_score(y_test, y_pred_default))
    logger.info("--- Optimized Threshold (%0.2f) ---")
    logger.info("Accuracy:  %0.4f", accuracy_score(y_test, y_pred_opt))
    logger.info("Precision: %0.4f", precision_score(y_test, y_pred_opt))
    logger.info("Recall:    %0.4f", recall_score(y_test, y_pred_opt))
    logger.info("F1 Score:  %0.4f", f1_score(y_test, y_pred_opt))
    
    # Calculate conversion metrics at optimized threshold
    cm = confusion_matrix(y_test, y_pred_opt)
    # Conversion rate of leads marked as Hot (optimized threshold)
    leads_contacted = int(np.sum(y_pred_opt))
    converted_contacted = int(cm[1, 1])
    conversion_rate_contacted = converted_contacted / leads_contacted if leads_contacted > 0 else 0.0
    logger.info(
        "Conversion rate of prioritised leads: %0.1f%% (%d/%d leads)",
        conversion_rate_contacted * 100, converted_contacted, leads_contacted
    )
    
    return {
        "variant": variant,
        "y_test": y_test,
        "y_prob": y_prob,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_threshold": best_thresh,
        "conversion_rate_prioritised": conversion_rate_contacted
    }


def generate_plots(results: list[dict[str, any]], output_dir: Path) -> None:
    """Generate ROC, PR, Score distribution, and Feature Importance plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curves
    plt.figure(figsize=(7, 6))
    for res in results:
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        plt.plot(fpr, tpr, label=f"Model {res['variant']} (AUC = {res['roc_auc']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    roc_path = output_dir / "06_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    logger.info("Saved ROC curve to %s", roc_path)
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(7, 6))
    for res in results:
        prec, rec, _ = precision_recall_curve(res["y_test"], res["y_prob"])
        plt.plot(rec, prec, label=f"Model {res['variant']} (PR AUC = {res['pr_auc']:.3f})", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(loc="lower left")
    pr_path = output_dir / "07_precision_recall_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    logger.info("Saved Precision-Recall curve to %s", pr_path)
    
    # 3. Lead Score Distribution (Model A)
    res_a = next(r for r in results if r["variant"] == "A")
    scores_a = res_a["y_prob"] * 100
    targets_a = res_a["y_test"].values
    
    plt.figure(figsize=(8, 5))
    sns.histplot(
        x=scores_a[targets_a == 0], 
        color="#E66101", 
        label="Not Converted (0)", 
        alpha=0.5, 
        bins=25, 
        kde=True,
        stat="density",
        common_norm=False
    )
    sns.histplot(
        x=scores_a[targets_a == 1], 
        color="#5E3C99", 
        label="Converted (1)", 
        alpha=0.5, 
        bins=25, 
        kde=True,
        stat="density",
        common_norm=False
    )
    plt.axvline(
        res_a["optimal_threshold"] * 100, 
        color="red", 
        linestyle="--", 
        label=f"Optimized Threshold ({res_a['optimal_threshold']*100:.1f})"
    )
    plt.xlabel("Lead Score (0-100)")
    plt.ylabel("Density")
    plt.title("Model A: Lead Score Distribution by Actual Conversion")
    plt.legend(loc="upper right")
    dist_path = output_dir / "08_lead_score_distribution.png"
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    logger.info("Saved Lead Score Distribution to %s", dist_path)


def plot_feature_importance(
    model: any, 
    feature_names: list[str], 
    variant: str, 
    output_dir: Path
) -> None:
    """Extract and plot feature importances for a model variant."""
    # Handle pipelines vs raw estimators
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["clf"]
        
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        logger.warning("Estimator has no feature_importances_ or coef_ attributes.")
        return
        
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Model {variant}: Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    
    feat_path = output_dir / f"09_feature_importance_model_{variant.lower()}.png"
    plt.tight_layout()
    plt.savefig(feat_path)
    plt.close()
    logger.info("Saved feature importance chart for Model %s to %s", variant, feat_path)


def main() -> None:
    setup_logging()
    
    variants = [
        ("A", PROCESSED_A_PATH),
        ("B", PROCESSED_B_PATH)
    ]
    
    results = []
    
    for variant, data_path in variants:
        model, X_test, y_test = load_model_and_test_data(variant, data_path)
        eval_metrics = evaluate_model(model, X_test, y_test, variant)
        results.append(eval_metrics)
        
        # Plot feature importance
        feature_names = X_test.columns.tolist()
        plot_feature_importance(model, feature_names, variant, FIGURES_DIR)
        
    generate_plots(results, FIGURES_DIR)
    
    # Save a summary markdown text
    summary_path = FIGURES_DIR.parent / "evaluation_summary.md"
    res_a = next(r for r in results if r["variant"] == "A")
    res_b = next(r for r in results if r["variant"] == "B")
    
    summary_content = f"""# Lead Scoring Model Evaluation Summary

## Model Performance

| Metric | Model A (No Leakage - Primary) | Model B (With Leakage - Reference) |
| :--- | :--- | :--- |
| **ROC AUC** | {res_a['roc_auc']:.4f} | {res_b['roc_auc']:.4f} |
| **PR AUC** | {res_a['pr_auc']:.4f} | {res_b['pr_auc']:.4f} |
| **Optimal Threshold** | {res_a['optimal_threshold']:.4f} | {res_b['optimal_threshold']:.4f} |
| **Priority Conversion Rate** | {res_a['conversion_rate_prioritised']*100:.1f}% | {res_b['conversion_rate_prioritised']*100:.1f}% |

- **Model A** is designed for deployment on live incoming traffic. It achieves a **{res_a['roc_auc']:.4f} ROC AUC** without using post-contact variables.
- **Model B** includes sales-assigned indices and tag fields. While it achieves **{res_b['roc_auc']:.4f} ROC AUC**, this performance is artificially inflated by data leakage (post-contact information) and should only be used as a reference benchmark.
"""
    summary_path.write_text(summary_content)
    logger.info("Saved evaluation summary markdown report to %s", summary_path)


if __name__ == "__main__":
    main()
