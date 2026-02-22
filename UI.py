from __future__ import annotations

import io
import json
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Insurance Bundle Recommender", layout="centered")

st.title("Insurance Bundle Recommender")
st.caption("Demo UI — sends requests to the FastAPI backend (/predict).")

# ---- Settings
DEFAULT_API_URL = "http://127.0.0.1:8000"  # change if deployed

with st.sidebar:
    st.header("Backend")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL).rstrip("/")
    st.write("Health check:")
    if st.button("Ping /health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(f"Failed: {e}")

st.divider()

tab1, tab2 = st.tabs(["Single prediction (Form)", "Batch prediction (CSV)"])


def call_predict(api_url: str, rows: list[dict]) -> pd.DataFrame:
    """Call FastAPI /predict and return dataframe of predictions."""
    payload = {"rows": rows}
    r = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    if r.status_code != 200:
        # FastAPI returns {"detail": "..."}
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"/predict failed ({r.status_code}): {detail}")

    data = r.json()
    preds = data.get("predictions", [])
    return pd.DataFrame(preds)


with tab1:
    st.subheader("Single customer")

    st.write("Fill what you know. Leave unknown fields empty; backend should handle missing columns.")
    with st.form("single_form"):
        user_id = st.text_input("User_ID", value="USR_000001")

        # Put your most important fields here (edit to match your dataset)
        col1, col2 = st.columns(2)
        with col1:
            employment_status = st.text_input("Employment_Status", value="")
            region_code = st.text_input("Region_Code", value="")
            deductible_tier = st.text_input("Deductible_Tier", value="")
        with col2:
            payment_schedule = st.text_input("Payment_Schedule", value="")
            broker_agency_type = st.text_input("Broker_Agency_Type", value="")
            acquisition_channel = st.text_input("Acquisition_Channel", value="")

        # High-card IDs (optional)
        col3, col4 = st.columns(2)
        with col3:
            broker_id = st.text_input("Broker_ID", value="")
        with col4:
            employer_id = st.text_input("Employer_ID", value="")

        # Add any numeric fields you have (examples)
        st.markdown("**Optional numeric fields (edit/expand):**")
        n1, n2, n3 = st.columns(3)
        with n1:
            annual_premium = st.number_input("Annual_Premium (if exists)", min_value=0.0, value=0.0, step=10.0)
        with n2:
            age = st.number_input("Age (if exists)", min_value=0, value=0, step=1)
        with n3:
            dependents = st.number_input("Dependents (if exists)", min_value=0, value=0, step=1)

        submitted = st.form_submit_button("Predict bundle")

    if submitted:
        row = {
            "User_ID": user_id,
            "Employment_Status": employment_status or None,
            "Region_Code": region_code or None,
            "Deductible_Tier": deductible_tier or None,
            "Payment_Schedule": payment_schedule or None,
            "Broker_Agency_Type": broker_agency_type or None,
            "Acquisition_Channel": acquisition_channel or None,
            "Broker_ID": broker_id or None,
            "Employer_ID": employer_id or None,
            "Annual_Premium": float(annual_premium) if annual_premium else None,
            "Age": int(age) if age else None,
            "Dependents": int(dependents) if dependents else None,
        }

        # remove None keys (optional)
        row = {k: v for k, v in row.items() if v is not None and v != ""}

        with st.spinner("Calling /predict ..."):
            try:
                df_pred = call_predict(api_url, [row])
                st.success("Prediction returned.")
                st.dataframe(df_pred, use_container_width=True)
            except Exception as e:
                st.error(str(e))


with tab2:
    st.subheader("Batch from CSV")

    st.write("Upload a CSV with a **User_ID** column (and any feature columns).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head(20), use_container_width=True)

        if "User_ID" not in df.columns:
            st.error("CSV must contain a 'User_ID' column.")
        else:
            if st.button("Predict for CSV"):
                rows = df.to_dict(orient="records")
                with st.spinner(f"Predicting {len(rows)} rows..."):
                    try:
                        df_pred = call_predict(api_url, rows)
                        st.success("Done.")
                        st.dataframe(df_pred, use_container_width=True)

                        # Merge predictions back to original if you want
                        merged = df.merge(df_pred, on="User_ID", how="left")

                        csv_bytes = merged.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results CSV",
                            data=csv_bytes,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(str(e))