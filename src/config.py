from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "gravity_speed_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "gravity_speed_data_processed.csv"

MODELS_DIR = ROOT_DIR / "models"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"

REPORTs_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTs_DIR / "figures"


TARGET_COL = "combined_dilation_theory"

RANDOM_STATE = 42
TEST_SIZE = 0.2

PERM_N_REPEATS = 10
PERM_SCORING = "neg_mean_squared_error"

PLOT_STYLE = "seaborn-v0_8-muted"
DPI = 300