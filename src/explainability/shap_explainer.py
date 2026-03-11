import shap
import joblib
from src.config import CONFIG

reg_model = joblib.load(CONFIG["regression_model_path"])
explainer = shap.Explainer(reg_model)

def explain_prediction(full_df):
    shap_values = explainer(full_df)
    return shap_values