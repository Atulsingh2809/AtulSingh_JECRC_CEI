"""Paths, constants, and configuration for the lead scoring pipeline."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CSV_PATH = RAW_DATA_DIR / "Leads.csv"
PROCESSED_A_PATH = PROCESSED_DATA_DIR / "processed_model_a.csv"
PROCESSED_B_PATH = PROCESSED_DATA_DIR / "processed_model_b.csv"

# Model & report paths
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_A_PATH = MODELS_DIR / "preprocessor_a.joblib"
PREPROCESSOR_B_PATH = MODELS_DIR / "preprocessor_b.joblib"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Target column
TARGET_COL = "Converted"

# Identifier columns (dropped from features, kept for output)
ID_COLS = ["Prospect ID", "Lead Number"]
PROSPECT_ID_COL = "Prospect ID"

# Potential data-leakage columns (sales-assigned post-contact)
# Model A excludes these; Model B includes them for comparison only.
LEAKAGE_COLS = [
    "Tags",
    "Lead Quality",
    "Last Notable Activity",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
]

# Known constant / near-zero-variance columns (verified via EDA)
CONSTANT_COLS = [
    "Magazine",
    "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "I agree to pay the amount through cheque",
]

# Numeric feature columns
NUMERIC_COLS = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
]

# High-cardinality categoricals → frequency encoding when used
HIGH_CARDINALITY_COLS = ["Tags", "Specialization", "City"]

# One-hot encode categoricals with at most this many unique values (after cleaning)
LOW_CARDINALITY_MAX = 10

# Binary Yes/No columns (excluding target)
BINARY_YES_NO_COLS = [
    "Do Not Email",
    "Do Not Call",
    "Search",
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Through Recommendations",
    "A free copy of Mastering The Interview",
]

# Imputation defaults (configurable via PreprocessorConfig)
DEFAULT_CATEGORICAL_FILL = "Unknown"
DEFAULT_MISSING_CATEGORY = "Missing"

# Lead scoring tiers
HOT_THRESHOLD = 70
WARM_THRESHOLD = 40

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# High-null threshold → use "Missing" category instead of mode imputation
HIGH_NULL_THRESHOLD = 0.40

# Categorical sentinel replaced with NaN
SELECT_SENTINEL = "Select"

# Required columns for raw data validation
REQUIRED_COLUMNS = [
    PROSPECT_ID_COL,
    "Lead Number",
    "Lead Origin",
    "Lead Source",
    TARGET_COL,
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
]
