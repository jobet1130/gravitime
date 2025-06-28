import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from .physics import (
    gravitational_time_dilation,
    velocity_time_dilation,
    combined_time_dilation,
    time_difference
)

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"❌ Raw data not found at: {path}")
    df = pd.read_csv(path)
    print(f"📂 Loaded raw data: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df[(df["radius_m"] > 0) & (df["velocity_m_s"] >= 0)]
    print(f"🧹 Cleaned data: {df.shape}")
    return df

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df["gravitational_dilation"] = gravitational_time_dilation(df["mass_kg"].values, df["radius_m"].values)
    df["velocity_dilation"] = velocity_time_dilation(df["velocity_m_s"].values)
    df["combined_dilation"] = combined_time_dilation(df["gravitational_dilation"], df["velocity_dilation"])
    df["time_far_s"] = df["combined_dilation"]
    df["time_near_s"] = df["velocity_dilation"]
    df["time_difference_s"] = time_difference(df["time_far_s"], df["time_near_s"])
    print("🧪 Physics-based features added.")
    return df

def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    log_cols = [col for col in numeric_cols if (df[col] > 0).all()]
    if not log_cols:
        print("⚠️ No positive-only numeric columns found for log1p transformation.")
        return df

    log_transformer = FunctionTransformer(np.log1p, validate=True)
    log_data = log_transformer.fit_transform(df[log_cols])
    log_df = pd.DataFrame(log_data, columns=[f"log1p_{col}" for col in log_cols], index=df.index)
    df = pd.concat([df, log_df], axis=1)
    print(f"📈 Added {len(log_cols)} log-transformed features.")
    return df

def add_polynomial_features(df: pd.DataFrame, exclude: list = None, degree: int = 2) -> pd.DataFrame:
    exclude = exclude or []
    base_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(exclude)
    if base_cols.empty:
        print("⚠️ No numeric columns available for polynomial features.")
        return df

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_array = poly.fit_transform(df[base_cols])
    poly_feature_names = poly.get_feature_names_out(base_cols)
    poly_df = pd.DataFrame(poly_array, columns=poly_feature_names, index=df.index)

    # Drop original columns to avoid duplication
    new_features = poly_df.drop(columns=base_cols, errors="ignore")
    df = pd.concat([df, new_features], axis=1)
    print(f"🔧 Added {new_features.shape[1]} polynomial features (degree={degree}).")
    return df

def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to: {path}")

def preprocess():
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = add_physics_features(df)
    df = add_log_features(df)
    df = add_polynomial_features(df, exclude=["mass_kg", "radius_m", "velocity_m_s"])
    save_processed_data(df, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    preprocess()
