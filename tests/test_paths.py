import pytest
from pathlib import Path
import re
from windai.paths import get_data_root, list_files

# Tests for get_data_root()
def test_path(tmp_path):
    assert get_data_root(tmp_path) == tmp_path

def test_bad_path(tmp_path):
    bad_path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match=f"Provided path '{re.escape(str(bad_path))}' is invalid, doesn't exist or is a file"):
        get_data_root(bad_path)
        
def test_missing_env(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    default_path = root / "data" / "raw" / "kushtriv" / "Desktop" / "windai_dataset"
    monkeypatch.delenv("WINDAI_DATA_ROOT", raising=False)
    assert get_data_root() == default_path
    
def test_empty_env(monkeypatch):
    monkeypatch.setenv("WINDAI_DATA_ROOT", "")
    root = Path(__file__).resolve().parents[1]
    default_path = root / "data" / "raw" / "kushtriv" / "Desktop" / "windai_dataset"
    assert get_data_root() == default_path
    
def test_invalid_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WINDAI_DATA_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match=f"Set the env variable WINDAI_DATA_ROOT to the path to 'windai_dataset', currently: {re.escape(str(tmp_path / "missing"))}"):
        get_data_root()
        
def test_valid_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WINDAI_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path
    
# Tests for list_files()
@pytest.fixture
def data_root(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setenv("WINDAI_DATA_ROOT", str(root))
    d = root / "sub" / "nested"
    d.mkdir(parents=True)
    (root / "b.parquet").touch()
    (root / "a.csv").touch()
    (root / "._a.csv").touch()
    (root / ".hidden").touch()
    (root / "sub" / "c.csv").touch()
    (root / "sub" / "nested" / "d.csv").touch()
    return root

def test_only_visible_files(data_root):
    assert set(list_files()) == {data_root / "a.csv", data_root / "b.parquet"}

def test_sorted(data_root):
    assert list_files() == [data_root / "a.csv", data_root / "b.parquet"]

def test_subdirs_excluded(data_root):
    assert data_root / "sub" not in list_files()

def test_nested_subdir(data_root):
    assert list_files("sub/nested") == [data_root / "sub" / "nested" / "d.csv"]

def test_not_recursive(data_root):
    assert data_root / "sub" / "nested" / "d.csv" not in list_files("sub")
    assert data_root / "sub" / "c.csv" in list_files("sub")

def test_missing_subdir(data_root):
    with pytest.raises(FileNotFoundError, match=f"Cannot find directory with path {re.escape(str(data_root / "missing"))}"):
        list_files("missing")

def test_subdir_is_file(data_root):
    with pytest.raises(NotADirectoryError, match=f"{re.escape(str(data_root / "a.csv"))} is a file, not a directory"):
        list_files("a.csv")

def test_absolute_path(data_root, tmp_path):
    o = tmp_path / "other"
    o.mkdir()
    (tmp_path / "other" / "some_file.txt").touch()
    assert list_files(tmp_path / "other") == [tmp_path / "other" / "some_file.txt"]