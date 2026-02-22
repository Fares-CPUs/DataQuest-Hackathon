from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import solution  # <- your solution.py (preprocess/load_model/predict)

app = FastAPI(title="Insurance Bundle Recommender API", version="1.0")

# Load model once at startup (fast requests)
MODEL = None

@app.on_event("startup")
def _startup():
    global MODEL
    MODEL = solution.load_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None}


class PredictRequest(BaseModel):
    # Option A: send rows as list of dicts (easy from frontend)
    rows: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]  # [{"User_ID": "...", "Purchased_Coverage_Bundle": 3}, ...]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = pd.DataFrame(req.rows)
        # Reuse your exact judge pipeline logic
        df2 = solution.preprocess(df)
        out = solution.predict(df2, MODEL)

        # Enforce the required columns
        if not {"User_ID", "Purchased_Coverage_Bundle"}.issubset(out.columns):
            raise ValueError("predict() must return User_ID and Purchased_Coverage_Bundle")

        # Convert to JSON-friendly format
        return {"predictions": out[["User_ID", "Purchased_Coverage_Bundle"]].to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))