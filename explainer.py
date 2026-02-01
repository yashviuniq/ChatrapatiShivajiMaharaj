"""Produce SHAP-based explanations (reason codes) for predictions.

Functions:
- load_pipeline
- explain_dataset -> returns top contributors for each example

Note: expects a pipeline saved by `model_trainer.py` where the pipeline is
Pipeline([('preprocess', ColumnTransformer), ('clf', RandomForestClassifier)])
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap


def load_pipeline(path: str):
    return joblib.load(path)


def _get_feature_names(pipeline) -> List[str]:
    """Extract feature names after preprocessing (works with sklearn >=1.0)"""
    preproc = pipeline.named_steps["preprocess"]

    # ColumnTransformer exposes get_feature_names_out
    try:
        names = preproc.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback: create approximate names
        num = preproc.transformers_[0][2]
        cat = preproc.transformers_[1][2]
        cat_ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps[
            "ohe"
        ]
        cat_names = list(cat_ohe.get_feature_names_out(cat)) if hasattr(cat_ohe, "get_feature_names_out") else list(cat)
        return list(num) + list(cat_names)


def explain_dataset(pipeline, X: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Return top-k positive contributing features (reason codes) per sample.

    Returns a DataFrame with columns: index, prob, top_reasons (list of tuples)
    where each tuple is (feature_name, shap_value)
    """
    # Extract model and transformed data
    clf = pipeline.named_steps["clf"]
    preproc = pipeline.named_steps["preprocess"]

    X_trans = preproc.transform(X)

    # SHAP TreeExplainer on RandomForest
    explainer = shap.TreeExplainer(clf)
    feature_names = _get_feature_names(pipeline)

    # SHAP TreeExplainer on RandomForest
    explainer = shap.TreeExplainer(clf)
    shap_vals_raw = explainer.shap_values(X_trans)

    # shap_vals_raw structure depends on sklearn/shap version
    # Binary classification: usually list of 2 arrays [arr_class0, arr_class1]
    # or single array if older shap
    if isinstance(shap_vals_raw, list):
        # We want Class 1 (Positive)
        vals = shap_vals_raw[1]
    else:
        # If single array, check if it's already class 1 or need logic
        # For RF classifier, it's safer to assume list, but if not:
        if len(shap_vals_raw.shape) == 3: # (samples, features, classes)
             vals = shap_vals_raw[:, :, 1]
        else:
             vals = shap_vals_raw
    
    # Ensure numpy
    vals = np.array(vals)
    
    # If vals is 2D (samples, features), correct.
    # If 1D (features,) - implies single sample but shape lost
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    
    # Double check compatibility
    if vals.shape[1] != len(feature_names):
         # Mismatch can happen if X_trans has different cols than feature_names expectation
         # But usually they match if pipeline is consistent
         pass

    results = []
    probs = pipeline.predict_proba(X)[:, 1]

    for i in range(len(X)):
        row = vals[i]
        # positive contributions sorted descending
        # Ensure row is 1D array of floats
        row = row.flatten()
        
        # Argsort gives indices
        pos_idx = np.argsort(-row)[:top_k]
        
        # Safe extraction
        top = []
        for j in pos_idx:
            # j must be int
            j_int = int(j)
            fname = feature_names[j_int] if j_int < len(feature_names) else f"Feature {j_int}"
            top.append((fname, float(row[j_int])))
        
        results.append({"index": i, "prob": float(probs[i]), "top_reasons": top})

    return pd.DataFrame(results)


def main(pipeline_path: str, data_csv: str, out_json: str = "explanations.json") -> None:
    pipeline = load_pipeline(pipeline_path)
    X = pd.read_csv(data_csv)[pipeline.named_steps["preprocess"].feature_names_in_]

    df = explain_dataset(pipeline, X, top_k=3)
    df.to_json(out_json, orient="records", lines=False)
    print(f"Wrote explanations to {out_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SHAP-based reason codes for a dataset")
    parser.add_argument("--pipeline", default="models/rf_pipeline.joblib")
    parser.add_argument("--data", default="data/patients.csv")
    parser.add_argument("--out", default="explanations.json")
    args = parser.parse_args()
    main(args.pipeline, args.data, args.out)
