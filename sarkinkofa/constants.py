import os
from importlib import resources as res_import


# Load resources
with res_import.path("sarkinkofa", "resources") as res_path:
    MODEL_DIR = os.path.join(res_path, "models")
