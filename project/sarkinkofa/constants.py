import os
from importlib import resources as res_import


# Load resources
with res_import.path("sarkinkofa", "resources") as res_path:
    MODEL_DIR = os.path.join(res_path, "models")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# ANPD_MODEL_DIR = os.path.join(MODEL_DIR, "anpd")
# ANPR_MODEL_DIR = os.path.join(MODEL_DIR, "anpr")
