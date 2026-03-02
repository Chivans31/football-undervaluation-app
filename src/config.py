import json
import os

# Get absolute path to the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# Update paths in config to be absolute
for key in ["regression_model_path", "classifier_model_path", "scaler_path", "pca_path"]:
    CONFIG[key] = os.path.join(BASE_DIR, CONFIG[key])