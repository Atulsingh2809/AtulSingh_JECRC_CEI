# X Education Lead Scoring

> **Status:** Project scaffold complete. Place `Leads.csv` in `data/raw/` to continue.

## Business Context

X Education generates many leads through online channels but only **~30% convert** into paying customers. Sales effort is spread equally across all leads, wasting time on low-potential prospects.

This project builds a data-driven **Lead Score (0–100)** for every prospect so the sales team can prioritize **Hot Leads** first. Target: help push conversion rate toward **80%** by focusing on the highest-scoring leads.

## Dataset

- **Source:** [Kaggle — Lead Scoring Dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)
- **Size:** 9,240 rows × 37 columns
- **Target:** `Converted` (1 = converted, 0 = not converted)

### Getting the Data

**Option 1 — Manual download (recommended if no Kaggle API key):**

1. Download from [Kaggle](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)
2. Place the file at: `data/raw/Leads.csv`

**Option 2 — Kaggle CLI:**

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

Requires `kaggle.json` in `~/.kaggle/` or `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Repository Structure

```
lead-scoring-x-education/
├── data/raw/              # Leads.csv (gitignored)
├── data/processed/        # Cleaned datasets (gitignored)
├── notebooks/             # EDA, feature engineering, modeling
├── src/                   # Core ML pipeline modules
├── models/                # Trained model artifacts (gitignored)
├── app/                   # Streamlit dashboard + FastAPI
├── tests/                 # Unit tests
├── scripts/               # Data download utilities
└── reports/figures/       # EDA charts
```

## Next Steps

After adding `data/raw/Leads.csv`:

1. Run EDA notebook: `notebooks/01_eda.ipynb`
2. Train models: `python -m src.train`
3. Launch Streamlit app: `streamlit run app/streamlit_app.py`

_Full documentation will be completed as the pipeline is built._

## License

MIT — see [LICENSE](LICENSE).
