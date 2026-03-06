import shap
import joblib
from src.config import CONFIG

reg_model = joblib.load(CONFIG["regression_model_path"])
explainer = shap.TreeExplainer(reg_model)

def explain_prediction(full_df):
    shap_values = shap.TreeExplainer(reg_model)

    if len(shap_values.shape) ==2:
        return shap_values[0]

    return shap_values