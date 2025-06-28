import pytest
from pathlib import Path
import importlib.util

# Dynamically load the config module
spec = importlib.util.spec_from_file_location("config", "src/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def test_paths_exist():
    assert isinstance(config.ROOT_DIR, Path)
    assert config.DATA_DIR == config.ROOT_DIR / "data"
    assert config.RAW_DATA_PATH == config.DATA_DIR / "raw" / "gravity_speed_data.csv"
    assert config.PROCESSED_DATA_PATH == config.DATA_DIR / "processed" / "gravity_speed_data_processed.csv"
    assert config.MODELS_DIR == config.ROOT_DIR / "models"
    assert config.RF_MODEL_PATH == config.MODELS_DIR / "rf_model.joblib"
    assert config.REPORTs_DIR == config.ROOT_DIR / "reports"
    assert config.FIGURES_DIR == config.REPORTs_DIR / "figures"

def test_constants_types():
    assert isinstance(config.TARGET_COL, str)
    assert isinstance(config.RANDOM_STATE, int)
    assert isinstance(config.TEST_SIZE, float)
    assert isinstance(config.PERM_N_REPEATS, int)
    assert isinstance(config.PERM_SCORING, str)
    assert isinstance(config.PLOT_STYLE, str)
    assert isinstance(config.DPI, int)
