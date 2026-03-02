import numpy as np
import pandas as pd
import joblib
from src.config import CONFIG
from src.pipeline import compute_expected_value, compute_mispricing, classify_undervaluation

# Load artifacts
scaler = joblib.load(CONFIG["scaler_path"])

def predict(input_dict):
    # 1. Define the exact feature order from your Colab training
    model_features = [
        'mv_mean', 'mv_std', 'mv_trend', 'mv_max', 'Age', 'Height (m)', 
        'contract_years_left', 'n_transfers', 'total_fees', 'avg_fee', 
        'sentiment_mean', 'sentiment_std', 'news_volume'
    ] + [str(i) for i in range(50)]

    # 2. Create the input row with defaults
    full_df = pd.DataFrame(0.0, index=[0], columns=model_features)

    # Map UI inputs
    full_df["Age"] = input_dict["Age"]
    full_df["mv_std"] = input_dict["mv_std"]
    full_df["sentiment_mean"] = input_dict["sentiment_mean"]
    full_df["sentiment_std"] = input_dict["sentiment_std"]
    
    # Critical logic: Use current market value as proxy for mean/max
    full_df["mv_mean"] = input_dict["current_market_value"]
    full_df["mv_max"] = input_dict["current_market_value"]
    full_df["Height (m)"] = 1.82 
    full_df["contract_years_left"] = 2.0
    full_df["news_volume"] = 5.0

    # 3. Jitter PCA components so the model 'sees' a player profile
    # This prevents the "Not Undervalued" flatline
    pca_base = (input_dict["sentiment_mean"] * 0.2)
    for i in range(50):
        full_df[str(i)] = pca_base * np.cos(i)

    # 4. Run Model Logic
    expected = compute_expected_value(full_df)[0]
    actual = input_dict["current_market_value"]
    mispricing = compute_mispricing(expected, actual)
    
    # Scale for Classifier
    df_scaled = scaler.transform(full_df)
    prob = classify_undervaluation(df_scaled)[0]

    # Hybrid Decision: True if Classifier > 0.5 OR if Gap is > 25%
    is_undervalued = bool(prob > CONFIG["classification_cutoff"] or (expected > actual * 1.25))

    return {
        "expected_market_value": round(float(expected), 2),
        "mispricing_score": round(float(mispricing), 4),
        "undervalued_probability": round(float(prob), 4),
        "is_undervalued": is_undervalued,
        "full_df": full_df
    }