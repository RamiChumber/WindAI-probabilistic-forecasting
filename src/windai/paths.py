import os
from pathlib import Path

def get_data_root(path=None):
    if not path:
        env_data_root = os.environ.get("WINDAI_DATA_ROOT")
        if not env_data_root:
            root = Path(__file__).resolve().parents[2]
            data_root = root / "data" / "raw" / "kushtriv" / "Desktop" / "windai_dataset"
            err_message = "Ensure the folder 'windai_dataset' can be found under data/raw/kushtriv/Desktop/ or set the env variable WINDAI_DATA_ROOT"
        else:
            data_root = Path(env_data_root)
            err_message = f"Set the env variable WINDAI_DATA_ROOT to the path to 'windai_dataset', currently: {env_data_root}"
    else:
        data_root = Path(path)
        err_message = f"Provided path '{path}' is invalid, doesn't exist or is a file"
        
    if not data_root.is_dir():
        raise FileNotFoundError(err_message)
    return data_root

def list_files(subdir=""):
    data_root = get_data_root()
    target_dir = data_root / subdir
    if target_dir.is_file():
        raise NotADirectoryError(f"{target_dir} is a file, not a directory")
    elif not target_dir.is_dir():
        raise FileNotFoundError(f"Cannot find directory with path {target_dir}")
    files = [x for x in target_dir.iterdir() if x.is_file() and not x.name.startswith(".")]
    return sorted(files)