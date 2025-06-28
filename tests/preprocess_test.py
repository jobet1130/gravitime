import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocess import (
    load_data,
    clean_data,
    add_physics_features,
    add_log_features,
    add_polynomial_features,
    save_processed_data,
)

# ---------------------------
# 🔧 Fixtures
# ---------------------------
@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "mass_kg": [5.97e24, 1.0e30],
        "radius_m": [6.371e6, 7e8],
        "velocity_m_s": [1e4, 2e7],
    })


# ---------------------------
# ✅ Tests
# ---------------------------
def test_load_data_success(tmp_path):
    sample = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    file_path = tmp_path / "sample.csv"
    sample.to_csv(file_path, index=False)
    loaded = load_data(file_path)
    pd.testing.assert_frame_equal(loaded, sample)


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data(Path("nonexistent_file.csv"))


def test_clean_data():
    df = pd.DataFrame({
        "radius_m": [100, -1, 0],
        "velocity_m_s": [10, 20, -5],
        "other": [1, 2, 3]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["radius_m"] > 0
    assert cleaned.iloc[0]["velocity_m_s"] >= 0


def test_add_physics_features(raw_df):
    df = add_physics_features(raw_df.copy())
    expected_cols = [
        "gravitational_dilation", "velocity_dilation", "combined_dilation",
        "time_far_s", "time_near_s", "time_difference_s"
    ]
    for col in expected_cols:
        assert col in df.columns
        assert np.all((df[col] >= 0) & (df[col] <= 1))


def test_add_log_features(raw_df):
    df = add_log_features(raw_df.copy())
    log_cols = [f"log1p_{col}" for col in raw_df.columns]
    for col in log_cols:
        assert col in df.columns
        assert np.all(df[col] > 0)


def test_add_polynomial_features(raw_df):
    df_poly = add_polynomial_features(raw_df.copy(), exclude=["mass_kg"])
    # Check if number of features increased
    assert df_poly.shape[1] > raw_df.shape[1]
    # Check for interaction or squared terms
    poly_feature_cols = df_poly.columns.difference(raw_df.columns)
    assert any("^" in col or "*" in col for col in poly_feature_cols)


def test_save_processed_data(tmp_path, raw_df):
    path = tmp_path / "processed.csv"
    save_processed_data(raw_df, path)
    assert path.exists()
    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(raw_df.reset_index(drop=True), loaded)
