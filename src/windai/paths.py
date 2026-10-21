import os
from pathlib import Path

def get_data_root(path=None):
    if path is None or path is "":
        env_data_root = os.environ.get("WINDAI_DATA_ROOT")
        if env_data_root is None:
            root = Path(__file__).resolve().parents[2]
            data_root = root / "data" / "raw" / "kushtriv" / "Desktop" / "windai_dataset"
            err_message = "Ensure the folder windai_dataset can be found under data/raw/kushtriv/Desktop/"
        else:
            data_root = Path(env_data_root)
            err_message = f"set WINDAI_DATA_ROOT to the path to 'windai_dataset'. Currently: {env_data_root}"
    else:
        data_root = Path(path)
        err_message = f"Provided path '{path}' is invalid, doesn't exist or is a file"
        
    if not data_root.is_dir():
        raise FileNotFoundError(err_message)
    return data_root