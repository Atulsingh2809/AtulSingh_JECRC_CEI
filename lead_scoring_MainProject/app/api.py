"""
FastAPI REST API for X Education Lead Scoring.
Endpoints:
- GET /health: Health check.
- POST /predict: Predict lead score and get key contributing factors.
- POST /predict-batch: Upload a CSV of leads, return scored CSV.
"""

from __future__ import annotations

import logging
import io
from typing import Any, Dict
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.predict import LeadPredictor

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="X Education Lead Scoring API",
    description="REST API for predicting lead conversion scores and identifying hot prospects.",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LeadPredictor instances lazily to save resources on startup
_predictors: Dict[str, LeadPredictor] = {}


def get_predictor(variant: str = "A") -> LeadPredictor:
    """Helper to retrieve or initialize a LeadPredictor for the given variant."""
    var = variant.upper()
    if var not in ["A", "B"]:
        raise HTTPException(status_code=400, detail="Invalid variant. Must be 'A' or 'B'.")
    
    if var not in _predictors:
        try:
            _predictors[var] = LeadPredictor(variant=var)
        except Exception as e:
            logger.error("Failed to load predictor for variant %s: %s", var, e)
            raise HTTPException(
                status_code=500, 
                detail=f"Model files for Variant {var} not found or corrupted. Train model first."
            )
    return _predictors[var]


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "description": "X Education Lead Scoring API is running."}


@app.post("/predict", tags=["Scoring"])
def predict_single_lead(
    payload: Dict[str, Any],
    variant: str = Query("A", description="Model variant to use: 'A' (primary) or 'B' (reference).")
) -> Dict[str, Any]:
    """
    Predict conversion score and tier for a single lead.
    
    Accepts raw lead features as key-value pairs (using Leads.csv column names).
    Returns score, tier, and top positive/negative contribution factors.
    """
    predictor = get_predictor(variant)
    try:
        results = predictor.predict_single(payload)
        return results
    except Exception as e:
        logger.exception("Error during single lead prediction")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", tags=["Scoring"])
async def predict_batch_leads(
    file: UploadFile = File(..., description="CSV file of leads matching raw dataset schema."),
    variant: str = Query("A", description="Model variant to use: 'A' (primary) or 'B' (reference).")
) -> StreamingResponse:
    """
    Upload a CSV file containing leads, annotate them with scores/tiers, and return the scored CSV file.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
        
    predictor = get_predictor(variant)
    
    try:
        # Read uploaded CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
            
        logger.info("Batch scoring %d leads using Variant %s...", len(df), variant)
        scored_df = predictor.predict_batch(df)
        
        # Save scored DataFrame to buffer
        stream = io.StringIO()
        scored_df.to_csv(stream, index=False)
        response_content = stream.getvalue()
        
        # Return CSV stream as downloadable response
        return StreamingResponse(
            io.BytesIO(response_content.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scored_leads_variant_{variant.lower()}.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during batch lead prediction")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
