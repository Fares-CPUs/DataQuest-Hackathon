from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import solution  # must be in same folder as this file

app = FastAPI(title="Insurance Bundle Recommender API", version="1.0")

MODEL = None

@app.on_event("startup")
def _startup():
    global MODEL
    MODEL = solution.load_model()

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None}

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = pd.DataFrame(req.rows)
        df2 = solution.preprocess(df)
        out = solution.predict(df2, MODEL)





        if not {"User_ID", "Purchased_Coverage_Bundle"}.issubset(out.columns):
            raise ValueError("predict() must return User_ID and Purchased_Coverage_Bundle")

        preds = out[["User_ID", "Purchased_Coverage_Bundle"]].copy()
        preds = preds.replace([float("inf"), float("-inf")], pd.NA).where(pd.notna(preds), None)

        return {"predictions": preds.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
