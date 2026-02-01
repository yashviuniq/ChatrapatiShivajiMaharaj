"""Streamlit dashboard for JijaArogyaCare: hospital-themed no-show risk viewer.

- Loads a saved pipeline (`models/rf_pipeline.joblib`)
- Allows generating sample data or uploading CSV
- Shows risk-level indicators: Green/Yellow/Red
- Displays reason codes (from SHAP) for selected patient
"""
from __future__ import annotations

import os
from typing import Optional

import joblib
import pandas as pd
import numpy as np
import shap
import streamlit as st

from data_gen import generate_synthetic_data
from explainer import explain_dataset, load_pipeline
from ocr_engine import extract_ehr_data
from doctor_finder import find_nearby_doctors

MODEL_PATH = "models/rf_pipeline.joblib"


def _dept_to_appointment(dept: str) -> str:
    if not dept:
        return "PrimaryCare"
    dept = dept.lower()
    if any(k in dept for k in ["cardio", "cardiac", "heart", "cardiology"]):
        return "Specialist"
    if any(k in dept for k in ["derma", "skin", "dermatology"]):
        return "Specialist"
    if any(k in dept for k in ["pedia", "child"]):
        return "PrimaryCare"
    return "PrimaryCare"


def _make_patient_from_ocr(extracted: dict, defaults: dict = None) -> dict:
    defaults = defaults or {}
    age = extracted.get("Age") or defaults.get("age", 40)
    dept = extracted.get("Department") or defaults.get("appointment_type", "PrimaryCare")
    meds = extracted.get("Medication", [])

    patient = {
        "age": int(age) if age is not None else defaults.get("age", 40),
        "lead_time": defaults.get("lead_time", 7),
        "past_no_shows": defaults.get("past_no_shows", 0),
        "distance_km": defaults.get("distance_km", 5.0),
        "gender": defaults.get("gender", "M"),
        "appointment_type": _dept_to_appointment(dept),
        "weather": defaults.get("weather", "Sunny"),
        "medications": meds,
    }

    return patient


RISK_THRESHOLDS = {
    "green": 0.4,
    "yellow": 0.7,  # >= yellow is yellow; >= red is red
}


def risk_label(prob: float) -> tuple[str, str]:
    if prob >= RISK_THRESHOLDS["yellow"]:
        return "High", "red"
    if prob >= RISK_THRESHOLDS["green"]:
        return "Medium", "orange"
    return "Low", "green"


@st.cache_data
def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def style_badge(label: str, color: str) -> str:
    return f"<span style='background:{color};color:white;padding:6px 10px;border-radius:6px;font-weight:600'>{label}</span>"


def render_doctor_finder(patient_df: pd.DataFrame, api_key: str, key_suffix: str = "main"):
    """Render the recommended doctors section with unique state keys."""
    if st.button("Find Nearest Doctor", key=f"find_doctor_btn_{key_suffix}"):
        st.session_state[f"find_doctor_{key_suffix}"] = True

    if st.session_state.get(f"find_doctor_{key_suffix}", False):
         # Extract specialty
        specialty = "General Physician"
        if "appointment_type" in patient_df.columns:
             specialty = patient_df["appointment_type"].iloc[0]
        
        if specialty == "PrimaryCare":
            specialty = "General Physician"
            
        st.subheader("👨‍⚕️ Recommended Doctors")
        st.markdown(f"**Department:** {specialty}")
        st.caption(f"Powered by Geoapify")
        
        # Mock location (Pune) for demo
        lat, lon = 18.5204, 73.8567 
        
        with st.spinner(f"Searching for {specialty} near you..."):
            doctors = find_nearby_doctors(lat, lon, specialty=specialty, api_key=api_key)
            
        if doctors:
            for i, doc in enumerate(doctors):
                # Mock availability (randomize slightly for demo)
                avail = "Available Today: 4:00 PM" if i % 2 == 0 else "Next Slot: Tomorrow 10:00 AM"
                st.markdown(f"""
                <div style='padding:10px; border:1px solid #ddd; border-radius:10px; margin-bottom:10px'>
                    <b>{doc['name']}</b> <span style='float:right; color:green; font-size:0.8em'>{avail}</span><br>
                    <span style='color:grey'>{doc['specialty']}</span> • {doc['distance']}<br>
                    <small>{doc['address']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No doctors found nearby.")


def main():
    st.set_page_config(page_title="JijaArogyaCare — No-Show Risk", layout="wide", page_icon="🏥")

    st.markdown("# 🏥 JijaArogyaCare — No-Show Risk Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Data")
        uploaded = st.file_uploader("Upload patient CSV (optional)", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            n = st.slider("Generate sample patients", 10, 5000, 200)
            if st.button("Generate sample data"):
                df = generate_synthetic_data(n_samples=n)
            else:
                st.info("Upload a CSV or press 'Generate sample data' to create a dataset.")
                df = None

    # Sidebar: Upload EHR / Prescription to auto-populate a prediction form
    with st.sidebar:
        st.header("EHR / Prescription OCR")
        ehr_file = st.file_uploader("Upload EHR/Prescription image", type=["png", "jpg", "jpeg", "tiff"], help="Upload an image of a prescription or brief EHR note")
        extracted = None
        if ehr_file is not None:
            try:
                extracted = extract_ehr_data(ehr_file)
                st.success("EHR parsed successfully")
                st.write(extracted)
            except Exception as e:
                st.error(f"Failed to parse EHR: {e}")

        st.markdown("---")
        st.subheader("One-off prediction form")
        # default inputs can be prefilled with extracted data
        defaults = {}
        if extracted:
            defaults = {"age": extracted.get("Age"), "appointment_type": extracted.get("Department")}
        age_val = st.number_input("Age", min_value=0, max_value=120, value=defaults.get("age") or 40)
        lead_val = st.number_input("Lead Time (days)", min_value=0, max_value=365, value=7)
        hist_val = st.number_input("History of no-shows", min_value=0, max_value=20, value=0)
        dist_val = st.number_input("Distance to clinic (km)", min_value=0.0, max_value=500.0, value=5.0)
        gender_val = st.selectbox("Gender", ["M", "F", "Other"], index=0)
        appt_type = st.selectbox("Appointment Type", ["PrimaryCare", "Specialist", "Lab", "Imaging"], index=0)
        weather_val = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"], index=0)

        if st.button("Predict from EHR/inputs"):
            st.session_state["predict_clicked"] = True
            
        if st.session_state.get("predict_clicked", False):
            # create patient row matching pipeline's expected columns
            patient_row = {
                "lead_time": int(lead_val),
                "distance_km": float(dist_val),
                "past_no_shows": int(hist_val),
                "age": int(age_val),
                "gender": gender_val,
                "appointment_type": appt_type if not extracted or not extracted.get("Department") else _dept_to_appointment(extracted.get("Department")),
                "weather": weather_val,
            }
            patient_df = pd.DataFrame([patient_row])
            try:
                prob = float(load_model().predict_proba(patient_df)[:, 1][0])
                label, color = risk_label(prob)
                st.markdown(f"**Predicted no-show probability:** {prob*100:.1f}%")
                st.markdown(style_badge(label, color), unsafe_allow_html=True)

                expl = explain_dataset(load_model(), patient_df, top_k=3)
                reasons = expl.loc[0, "top_reasons"]
                st.markdown("**Top Reasons (feature, contribution)**")
                for feat, val in reasons:
                    sign = "+" if val > 0 else "-"
                    st.write(f"{feat}: {sign}{abs(val):.3f}")

                # Weather-driven alert for severe conditions
                distance_val = patient_row.get("distance_km", 0)
                weather_flag = any("weather" in str(f).lower() or "storm" in str(f).lower() for f, _ in reasons)
                if label == "High" and weather_flag and distance_val is not None and distance_val > 10:
                    st.warning("Note: Patient likely to miss due to severe weather conditions + travel distance. Suggesting Telehealth conversion.")
                
                # Doctor Finder in Sidebar
                # We need api_key_input. Let's assume it's defined in main or we use the default.
                # Ideally, we move API key input to sidebar top.
                # For this patch, I will use the default key directly or check session state if I move logic.
                # I will define api_key locally here to be safe, defaulting to the provided one.
                default_key = "dc0cf78c14ec43ed8203f975bedefec0"
                render_doctor_finder(patient_df, api_key=default_key, key_suffix="sidebar")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown("---")
        st.caption("Upload EHR to auto-populate fields and run a one-off prediction")

    with col2:
        st.subheader("Model")
        if os.path.exists(MODEL_PATH):
            st.success("Model found")
            if st.button("Reload model"):
                load_model.clear()
                model = load_model()
                st.experimental_rerun()
        else:
            st.warning("Model not found. Run `python model_trainer.py` first.")

    if df is None:
        return

    # Keep a copy of original
    original = df.copy()

    model = load_model()

    # Select features expected by model
    expected = model.named_steps["preprocess"].feature_names_in_
    missing = [f for f in expected if f not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    X = df[expected]
    probs = model.predict_proba(X)[:, 1]
    df["no_show_prob"] = probs

    # Risk category
    df["risk"], df["risk_color"] = zip(*[risk_label(p) for p in probs])

    st.subheader("Predictions")
    st.markdown("**Summary**")
    counts = df["risk"].value_counts().to_dict()
    c1, c2, c3 = st.columns(3)
    c1.markdown(style_badge("Low", "green"), unsafe_allow_html=True)
    c1.caption(f"{counts.get('Low',0)} patients")
    c2.markdown(style_badge("Medium", "orange"), unsafe_allow_html=True)
    c2.caption(f"{counts.get('Medium',0)} patients")
    c3.markdown(style_badge("High", "red"), unsafe_allow_html=True)
    c3.caption(f"{counts.get('High',0)} patients")

    st.subheader("Admin Console")
    
    # Toggle for Admin View
    admin_mode = st.checkbox("Enable Admin View (High Risk / Telehealth)", value=False)

    if admin_mode:
        st.info("Admin Mode: Filter needy patients and switch to Online appointments.")
        
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            risk_filter = st.multiselect("Filter by Risk", ["High", "Medium", "Low"], default=["High"])
        with filter_col2:
            mode_filter = st.multiselect("Filter by Mode", ["Online", "Offline"], default=["Offline"])
            
        # Filter Logic
        filtered_df = df.copy()
        if risk_filter:
            filtered_df = filtered_df[filtered_df["risk"].isin(risk_filter)]
        # Determine mode column name
        col_mode = "appointment_mode"
        if col_mode not in filtered_df.columns:
             if "Appointment_Mode" in filtered_df.columns:
                 col_mode = "Appointment_Mode"
             else:
                 filtered_df[col_mode] = "Offline"

        if mode_filter:
            filtered_df = filtered_df[filtered_df[col_mode].isin(mode_filter)]

        st.write(f"Showing {len(filtered_df)} patients")

        # Display Custom Table for Admins
        for i, row in filtered_df.iterrows():
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 2])
                c1.write(f"**#{i}**")
                c2.write(f"Risk: {style_badge(row['risk'], row['risk_color'])} ({row['no_show_prob']*100:.1f}%)", unsafe_allow_html=True)
                c3.write(f"Dept: {row['appointment_type']}")
                
                # Mode Toggle (Mock)
                current_mode = row.get(col_mode, "Offline")
                new_mode = c4.selectbox(f"Mode #{i}", ["Offline", "Online"], index=0 if current_mode=="Offline" else 1, key=f"mode_{i}")
                
                # Action
                if new_mode == "Online":
                    # Generate Google Meet Link (Mock)
                    meet_link = f"https://meet.google.com/new?authuser=0&patient={i}"
                    c5.markdown(f"[Start Google Meet]({meet_link})", unsafe_allow_html=True)
                else:
                    if st.button(f"Send Reminder #{i}", key=f"remind_{i}"):
                        st.toast(f"✅ Reminder sent to Patient #{i}!", icon="📩")
                        # Mock backend call result
                        c5.success("Sent!")
                    else:
                        c5.caption("Patient visiting clinic")
                
                st.markdown("---")
                
    else:
        # Standard Table
        st.dataframe(df.head(200))

    st.subheader("Inspect individual patient")
    idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0)
    patient = X.iloc[[idx]]

    # Allow overriding or selecting weather for the inspected patient
    default_weather = None
    if "weather" in patient.columns:
        default_weather = patient.iloc[0]["weather"]
    else:
        # try original dataset casing
        default_weather = original.iloc[idx]["weather"] if "weather" in original.columns else "Sunny"

    weather_choice = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"], index=["Sunny", "Cloudy", "Rainy", "Stormy"].index(default_weather) if default_weather in ["Sunny", "Cloudy", "Rainy", "Stormy"] else 0)

    # create a modified patient row with selected weather
    patient_mod = patient.copy()
    patient_mod = patient_mod.reset_index(drop=True)
    patient_mod.loc[0, "weather"] = weather_choice

    # compute probability with potentially updated weather
    prob = float(model.predict_proba(patient_mod)[:, 1][0])
    label, color = risk_label(prob)

    st.markdown(f"**Predicted no-show probability:** {prob*100:.1f}%")
    st.markdown(style_badge(label, color), unsafe_allow_html=True)

    # Explain
    if st.button("Explain this prediction"):
        st.session_state["explain_clicked"] = True
        
    if st.session_state.get("explain_clicked", False):
        st.info("Computing SHAP reason codes...")
        expl = explain_dataset(model, patient_mod, top_k=3)
        reasons = expl.loc[0, "top_reasons"]

        st.markdown("**Top Reasons (feature, contribution)**")
        for feat, val in reasons:
            sign = "+" if val > 0 else "-"
            st.write(f"{feat}: {sign}{abs(val):.3f}")
            # Specific reasoning callout for Weather
            if "weather" in str(feat).lower() or "storm" in str(feat).lower() or "rain" in str(feat).lower():
                st.info(f"**Weather Impact**: {feat} is increasing the risk by {abs(val):.3f}. Consider rescheduling if travel is difficult.")

        # Weather-driven alert: if high risk and weather is a contributor and distance>10km
        distance_val = None
        if "distance_km" in original.columns:
            distance_val = original.loc[idx, "distance_km"]
        elif "Distance_to_Clinic" in original.columns:
            distance_val = original.loc[idx, "Distance_to_Clinic"]

        weather_flag = any("weather" in str(f).lower() or "storm" in str(f).lower() for f, _ in reasons)
        if label == "High" and weather_flag and distance_val is not None and distance_val > 10:
            st.warning("Note: Patient likely to miss due to severe weather conditions + travel distance. Suggesting Telehealth conversion.")

        # Try to show a waterfall
        try:
            preproc = model.named_steps["preprocess"]
            X_trans = preproc.transform(patient_mod)
            explainer = shap.TreeExplainer(model.named_steps["clf"])
            
            # Calculate SHAP values
            shap_output = explainer.shap_values(X_trans)
            
            # --- Robust SHAP Value Extraction ---
            sv = None
            if isinstance(shap_output, list):
                # Binary classification usually returns [class_0, class_1]
                if len(shap_output) > 1:
                     sv = shap_output[1]
                else:
                     sv = shap_output[0]
            elif isinstance(shap_output, np.ndarray):
                # Could be (samples, features) or (samples, features, classes)
                if shap_output.ndim == 3:
                    sv = shap_output[:, :, 1]
                else:
                    sv = shap_output
            else:
                 sv = shap_output

            # Flatten to (features,) for single sample
            if isinstance(sv, np.ndarray) and sv.ndim == 2:
                sv = sv[0]
            # ------------------------------------

            # --- Robust Base Value Extraction ---
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                 if len(base_val) > 1:
                     base_val = base_val[1]
                 else:
                     base_val = base_val[0]
            # ------------------------------------
            
            # Plot
            st.pyplot(shap.plots._waterfall.waterfall_legacy(
                base_val, 
                sv, 
                feature_names=explainer.feature_names if hasattr(explainer, 'feature_names') else None, 
                show=False
            ))
            
        except Exception as e:
            st.error(f"SHAP Error Details: {e}")
            # Optional: st.write(f"Debug: expected_value shape={np.shape(explainer.expected_value)}")

    st.markdown("---")
    st.markdown("---")
    
    # Api Key Input
    api_key_input = "dc0cf78c14ec43ed8203f975bedefec0"
    if st.checkbox("Override API Key"):
        api_key_input = st.text_input("Enter Geoapify Key", type="password", value=api_key_input)
        
    render_doctor_finder(patient_mod, api_key=api_key_input, key_suffix="main")

    st.markdown("---")
    st.caption("JijaArogyaCare · A professional no-show risk assistant for hospitals")


if __name__ == "__main__":
    main()
