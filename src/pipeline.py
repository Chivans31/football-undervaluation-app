import numpy as np
import joblib
from src.config import CONFIG

# Load models once when module is imported
reg_model = joblib.load(CONFIG["regression_model_path"])
clf_model = joblib.load(CONFIG["classifier_model_path"])

def compute_expected_value(features_df):
    log_pred = reg_model.predict(features_df)
    return np.expm1(log_pred)

def compute_mispricing(expected, actual):
    return np.log1p(expected) - np.log1p(actual)

def classify_undervaluation(features_scaled):
    return clf_model.predict_proba(features_scaled)[:, 1]