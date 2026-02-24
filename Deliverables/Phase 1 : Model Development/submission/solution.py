from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

_BUNDLE: Dict[str, Any] | None = None

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    global _BUNDLE
    if _BUNDLE is None:
        import joblib
        _BUNDLE = joblib.load(Path(__file__).resolve().parent / "model.joblib")

    b = _BUNDLE
    out = df.copy()
    out["User_ID"] = out["User_ID"].astype(str)

    missing_token = b["missing_token"]
    month_map = b["month_map"]
    freq_maps = b["freq_maps"]

    def _num(s): return pd.to_numeric(s, errors="coerce")
    def _key(v):
        if pd.isna(v): return missing_token
        try: return str(int(v))
        except Exception: return str(v)

    out["Total_Dependents"] = (
        _num(out.get("Adult_Dependents")).fillna(0)
        + _num(out.get("Child_Dependents")).fillna(0)
        + _num(out.get("Infant_Dependents")).fillna(0)
    ).astype("float32")

    income = _num(out.get("Estimated_Annual_Income")).fillna(0)
    out["Income_Log"] = np.log1p(income).astype("float32")

    prev = _num(out.get("Previous_Claims_Filed")).fillna(0)
    yrs = _num(out.get("Years_Without_Claims")).fillna(0)
    amend = _num(out.get("Policy_Amendments_Count")).fillna(0)
    out["Risk_Score"] = (prev - yrs + amend).astype("float32")

    month_num = out.get("Policy_Start_Month").map(month_map).fillna(0).astype(int)
    out["Month_sin"] = np.sin(2*np.pi*month_num/12).astype("float32")
    out["Month_cos"] = np.cos(2*np.pi*month_num/12).astype("float32")

    if "Broker_ID" in out.columns:
        keys = out["Broker_ID"].map(_key)
        out["Broker_ID_freq"] = keys.map(freq_maps.get("Broker_ID",{})).fillna(0).astype("float32")

    feature_order = b["feature_order"]
    cat_cols = b["cat_cols"]
    encoders = b["encoders"]
    num_cols = b["num_cols"]
    num_medians = b["num_medians"]

    for col in feature_order:
        if col not in out.columns:
            out[col] = pd.NA

    X = out[feature_order].copy()

    for c in cat_cols:
        s = X[c].astype("string").fillna(missing_token)
        X[c] = s.map(encoders.get(c,{})).fillna(-1).astype("int32")

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(num_medians.get(c,0)).astype("float32")

    return pd.concat([out[["User_ID"]], X], axis=1)

def load_model() -> Any:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    import joblib
    _BUNDLE = joblib.load(Path(__file__).resolve().parent / "model.joblib")
    return _BUNDLE

def predict(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    b = model
    clf = b["model"]
    feature_order = b["feature_order"]
    user_ids = df["User_ID"].astype(str).to_numpy()
    X_mat = df[feature_order].to_numpy(dtype=np.float32, copy=False)
    preds = clf.predict(X_mat)
    preds = np.clip(np.asarray(preds).reshape(-1).astype(int),0,9)
    return pd.DataFrame({"User_ID":user_ids,"Purchased_Coverage_Bundle":preds})
