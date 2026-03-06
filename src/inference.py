import numpy as np
import pandas as pd
import joblib
from src.config import CONFIG
from src.pipeline import (
    compute_expected_value, 
    compute_mispricing, 
    classify_undervaluation
)

# 1. Load artifacts ONCE at startup (More efficient)
FEATURE_ORDER = joblib.load("models/artifacts/feature_names.joblib")
SCALER = joblib.load(CONFIG["scaler_path"])

def predict(input_dict):
    # 2. Initialize with the exact columns the model expects
    full_df = pd.DataFrame(0.0, index=[0], columns=FEATURE_ORDER)

    # 3. Map inputs (Using your logic)
    # Using .get() prevents crashes if a key is missing from the UI
    full_df["Age"] = input_dict.get("Age", 25)
    full_df["mv_std"] = input_dict.get("mv_std", 0.0)
    full_df["sentiment_mean"] = input_dict.get("sentiment_mean", 0.0)
    full_df["sentiment_std"] = input_dict.get("sentiment_std", 0.0)
    
    # Critical proxies
    current_val = input_dict["current_market_value"]
    full_df["mv_mean"] = current_val
    full_df["mv_max"] = current_val
    full_df["Height (m)"] = 1.82 
    full_df["contract_years_left"] = 2.0
    full_df["news_volume"] = 5.0

    # 4. Apply the PCA Jitter
    pca_base = (input_dict.get("sentiment_mean", 0.0) * 0.2)
    for i in range(50):
        col_name = str(i)
        if col_name in full_df.columns:
            full_df[col_name] = pca_base * np.cos(i)

    # 5. Final Alignment check
    # This ensures columns are in the EXACT order of FEATURE_ORDER
    full_df = full_df[FEATURE_ORDER]

    # 6. Run Model Logic
    expected = compute_expected_value(full_df)[0]
    actual = current_val
    mispricing = compute_mispricing(expected, actual)
    
    # Scale for Classifier
    df_scaled = SCALER.transform(full_df)
    prob = classify_undervaluation(df_scaled)[0]

    # Hybrid Decision Logic
    is_undervalued = bool(prob > 0.15 or (expected > actual * 1.15))

    return {
        "expected_market_value": round(float(expected), 2),
        "mispricing_score": round(float(mispricing), 4),
        "undervalued_probability": round(float(prob), 4),
        "is_undervalued": is_undervalued,
        "full_df": full_df
    }