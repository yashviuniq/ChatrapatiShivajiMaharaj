"""Train a RandomForest model for no-show prediction and save pipeline with joblib.

Produces: models/rf_pipeline.joblib

Usage:
    python model_trainer.py --data data/patients.csv --out models/rf_pipeline.joblib

Copilot Chat Prompt:
"Write the training pipeline for a Random Forest Classifier. Include:

Train-test split (80/20).

Handling of imbalanced data (use class_weight='balanced').

Model evaluation printing: Accuracy, Precision, Recall, and F1-Score.

Export the final model to a folder named models/."
"""
from __future__ import annotations

import argparse
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Union, Optional
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["lead_time", "distance_km", "past_no_shows", "age"]
CATEGORICAL_FEATURES = ["gender", "appointment_type", "weather"]
TARGET = "no_show"


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42, save_csv: bool = False, csv_path: str = "data/synthetic_patients.csv") -> pd.DataFrame:
    """Generate synthetic patient records for training and demo.

    Columns:
    - Appointment_ID: unique string
    - Age: integer
    - Lead_Time: days since booking (strong positive effect on No-Show)
    - History_of_No_Show: integer 0-5 (strong positive effect on No-Show)
    - Distance_to_Clinic: kilometers (float)
    - Status: 'Show' or 'No-Show' target

    """
    rng = np.random.default_rng(seed)

    # Demographics and features
    age = np.clip(rng.normal(45, 16, size=n_samples).round().astype(int), 0, 100)

    # Lead time with heavy-tailed behavior; higher lead_time increases likelihood of forgetting
    lead_time = np.clip(rng.poisson(14, size=n_samples) + rng.integers(0, 7, size=n_samples), 0, 365)

    # History of prior no-shows (0-5)
    history = np.clip(rng.poisson(0.8, size=n_samples), 0, 5)

    # Distance to clinic: log-normal distribution to simulate skew
    distance = np.clip(rng.lognormal(mean=np.log(5), sigma=0.6, size=n_samples), 0.1, 200).round(2)

    # Weather
    weather = rng.choice(["Sunny", "Rainy", "Stormy"], size=n_samples, p=[0.7, 0.25, 0.05])

    # Strong drivers for no-show: history and lead_time
    coef_history = 1.4
    coef_lead = 0.07
    coef_age = -0.01
    coef_distance = 0.02
    intercept = -2.0

    linear = coef_history * history + coef_lead * lead_time + coef_age * age + coef_distance * distance + intercept
    # weather effects
    linear += np.where(weather == "Rainy", 0.15, 0.0)
    linear += np.where((weather == "Stormy") & (distance > 10), 1.6, 0.0)

    prob_no_show = 1.0 / (1.0 + np.exp(-linear))

    status = np.where(rng.random(n_samples) < prob_no_show, "No-Show", "Show")

    appointment_ids = [f"APT{100000 + i}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "Appointment_ID": appointment_ids,
            "Age": age,
            "Lead_Time": lead_time,
            "History_of_No_Show": history,
            "Distance_to_Clinic": distance,
            "weather": weather,
            "Status": status,
        }
    )

    if save_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

    return df


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ]
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")

    pipeline = Pipeline([("preprocess", preproc), ("clf", clf)])

    return pipeline


def train_and_save(data_path: str, out_path: str) -> None:
    df = pd.read_csv(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipeline, out_path)

    print(f"Model saved to {out_path}")
    print(f"Test Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}, F1-Score: {f1:.4f}")


def train_random_forest(
    data_path: str = "data/synthetic_patients.csv",
    model_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
):
    """Train a RandomForest on numeric features with class_weight='balanced'.

    Expects a CSV with columns: Appointment_ID, Age, Lead_Time, History_of_No_Show, Distance_to_Clinic, Status
    """
    df = pd.read_csv(data_path)

    # Map column alternatives for flexibility
    if "Status" not in df.columns and "no_show" in df.columns:
        df["Status"] = df["no_show"].map({0: "Show", 1: "No-Show"})

    required = ["Age", "Lead_Time", "History_of_No_Show", "Distance_to_Clinic", "weather", "Status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for simple training: {missing}")

    X = df[["Age", "Lead_Time", "History_of_No_Show", "Distance_to_Clinic"]].copy()
    # Map weather to numeric categories for simple RF
    X["weather"] = df["weather"].map({"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 3})
    y = df["Status"].map({"Show": 0, "No-Show": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Simple RandomForest evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "random_forest.joblib")
    joblib.dump(clf, out_path)
    print(f"Exported simple RandomForest to {out_path}")

    return {"model": clf, "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}}


def _map_feature_to_reason(feature_name: str) -> str:
    """Convert a feature name to a human-readable reason label."""
    name = feature_name.lower()
    if "lead" in name:
        return "Long Lead Time"
    if "history" in name or "no_show" in name or "past" in name:
        return "Past No-Show History"
    if "distance" in name:
        return "Far Distance to Clinic"
    if "age" in name:
        return "Age"
    if "gender" in name:
        # keep specific gender if present
        tokens = feature_name.split("_")
        if len(tokens) > 1:
            return f"Gender: {tokens[-1]}"
        return "Gender"
    if "appointment" in name or "specialist" in name or "primary" in name or "lab" in name:
        tokens = feature_name.split("_")
        return f"Appointment Type: {tokens[-1]}" if len(tokens) > 1 else "Appointment Type"
    if "weather" in name or "storm" in name or "rain" in name or "cloud" in name:
        if "storm" in name:
            return "Severe Weather (Storm)"
        if "rain" in name:
            return "Inclement Weather (Rain)"
        return "Weather Condition"

    # fallback
    return feature_name


def get_prediction_reason(
    patient_data: Union[dict, pd.Series, pd.DataFrame],
    model_path: Optional[str] = None,
    top_k: int = 2,
    risk_threshold: float = 0.7,
) -> Dict[str, Union[float, str, List[Tuple[str, float]]]]:
    """Return actionable reasons (top contributors) for a patient's no-show probability.

    Args:
        patient_data: single patient as dict, Series, or single-row DataFrame
        model_path: optional path to a saved model; falls back to known locations
        top_k: number of top contributing factors to return
        risk_threshold: threshold to label 'High' risk

    Returns:
        Dict with keys: prob (float), risk_label (str), top_reasons (list of (label, contribution))
    """
    # Load model: try given path or common locations
    possible = [model_path] if model_path else ["models/rf_pipeline.joblib", "models/random_forest.joblib", "models/random_forest.pkl", "models/random_forest.joblib"]
    model = None
    last_err = None
    for p in possible:
        if not p:
            continue
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                break
            except Exception as e:
                last_err = e
                continue
    if model is None:
        raise FileNotFoundError(f"No model found in {possible}. Last error: {last_err}")

    # Normalize input into a single-row DataFrame
    if isinstance(patient_data, dict):
        X = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        X = patient_data.to_frame().T
    elif isinstance(patient_data, pd.DataFrame):
        if len(patient_data) == 0:
            raise ValueError("Empty DataFrame passed to get_prediction_reason")
        X = patient_data.iloc[[0]]
    else:
        raise TypeError("patient_data must be dict, pandas.Series, or single-row pandas.DataFrame")

    # Determine if model is a pipeline with 'preprocess' + 'clf' or a raw RandomForest
    if hasattr(model, "named_steps") and "preprocess" in model.named_steps and "clf" in model.named_steps:
        preproc = model.named_steps["preprocess"]
        clf = model.named_steps["clf"]

        # Attempt to get expected columns
        try:
            expected = list(preproc.feature_names_in_)
        except Exception:
            # fallback to transformers
            expected = []
            for name, trans, cols in preproc.transformers_:
                expected.extend(list(cols))

        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns required by model preprocessor: {missing}")

        X_proc = X[expected]
        X_trans = preproc.transform(X_proc)
        probs = model.predict_proba(X_proc)[:, 1]
        prob = float(probs[0])

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_trans)
        # Robust extraction
        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        else:
            vals = shap_vals
        vals = np.array(vals)
        if vals.ndim == 2:
            sv = vals[0]
        elif vals.ndim == 1:
            sv = vals
        else:
            sv = vals.flatten()

        # Feature names after preprocessing
        try:
            feature_names = list(preproc.get_feature_names_out())
        except Exception:
            # try to construct names for numeric + categorical
            feature_names = list(expected)

    else:
        # assume simple RandomForest trained on numeric columns
        clf = model
        expected = ["Age", "Lead_Time", "History_of_No_Show", "Distance_to_Clinic", "weather"]
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns for simple RF model: {missing}")
        # ensure same column order used during training (weather becomes numeric mapping)
        X_proc = X[expected].copy()
        if X_proc["weather"].dtype == object:
            X_proc["weather"] = X_proc["weather"].map({"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 3})
        X_arr = X_proc.values
        probs = clf.predict_proba(X_arr)[:, 1]
        prob = float(probs[0])

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_arr)
        # Robust extraction
        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        else:
            vals = shap_vals
        vals = np.array(vals)
        if vals.ndim == 2:
            sv = vals[0]
        elif vals.ndim == 1:
            sv = vals
        else:
            sv = vals.flatten()

        feature_names = expected

    # Identify top positive contributors
    idx_sorted = np.argsort(-sv)
    top = []
    for j in idx_sorted:
        if len(top) >= top_k:
            break
        contribution = float(sv[j])
        if contribution <= 0:
            # prefer positive contributors for 'High Risk'
            continue
        fname = feature_names[j] if j < len(feature_names) else f"f{j}"
        label = _map_feature_to_reason(fname)
        top.append((label, round(contribution, 6)))

    # If not enough positive contributors, fill with top absolute contributors
    if len(top) < top_k:
        for j in idx_sorted:
            if len(top) >= top_k:
                break
            contribution = float(sv[j])
            fname = feature_names[j] if j < len(feature_names) else f"f{j}"
            label = _map_feature_to_reason(fname)
            if (label, round(contribution,6)) not in top:
                top.append((label, round(contribution, 6)))

    # Risk label
    if prob >= risk_threshold:
        risk_label = "High"
    elif prob >= 0.4:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    return {"prob": prob, "risk_label": risk_label, "top_reasons": top}



def main(args: argparse.Namespace) -> None:
    # Optionally generate a synthetic dataset first
    if getattr(args, "generate_data", False):
        print("Generating synthetic data...")
        generate_synthetic_data(n_samples=args.n_samples, seed=args.seed, save_csv=True, csv_path=args.data)

    if getattr(args, "simple_train", False):
        train_random_forest(data_path=args.data, model_dir=os.path.dirname(args.out) or "models")
    else:
        train_and_save(args.data, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest and save the pipeline")
    parser.add_argument("--data", type=str, default="data/patients.csv")
    parser.add_argument("--out", type=str, default="models/rf_pipeline.joblib")
    parser.add_argument("--generate-data", dest="generate_data", action="store_true", help="Generate synthetic data before training")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of synthetic records to create when --generate-data is set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simple-train", dest="simple_train", action="store_true", help="Run simple RandomForest training on synthetic columns")
    args = parser.parse_args()

    # Demo: if running directly and a model exists, show explanation for first synthetic patient
    if args.generate_data:
        generate_synthetic_data(n_samples=args.n_samples, seed=args.seed, save_csv=True, csv_path=args.data)

    main(args)

    # Try a demo explanation if a model exists
    demo_model_paths = ["models/random_forest.joblib", "models/random_forest.pkl", "models/rf_pipeline.joblib"]
    demo_model = None
    for p in demo_model_paths:
        if os.path.exists(p):
            try:
                demo_model = p
                break
            except Exception:
                continue

    if demo_model and os.path.exists(args.data):
        df_demo = pd.read_csv(args.data)
        if len(df_demo) > 0:
            sample = df_demo.iloc[0]
            print("Demo patient:")
            print(sample.to_dict())
            try:
                reason = get_prediction_reason(sample)
                print("Explanation:", reason)
            except Exception as e:
                print("Could not compute explanation:", e)
    else:
        print("No model or data available for demo explanation. Run with --generate-data and --simple-train to create them.")
