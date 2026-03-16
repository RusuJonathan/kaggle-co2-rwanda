import pandas as pd
import numpy as np
from typing import Dict, List

from src.data.data_loader import load_yaml, preprocessing_path

preprocessing_config = load_yaml(preprocessing_path)
KEY_EMISSION_COLS = preprocessing_config["KEY_EMISSION_COLS"]

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives calendar and cyclical features from year and week_no.

    week_no (1–52) is mapped to sin/cos encodings so the model sees
    week 52 and week 1 as adjacent (circular continuity).
    Rwanda's dry season runs approximately June–September (weeks 22–39).
    """
    df = df.copy()

    df["month"] = ((df["week_no"] - 1) // 4 + 1).clip(1, 12).astype(int)
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
    })

    df["week_sin"] = np.sin(2 * np.pi * df["week_no"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_no"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_dry_season"] = ((df["week_no"] >= 22) & (df["week_no"] <= 39)).astype(int)

    df["year_normalized"] = df["year"] - df["year"].min()

    return df


def add_spatial_features(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Bins latitude and longitude and creates a location_id encoding
    so tree models can pick up zone-level patterns.
    """
    df = df.copy()

    df["lat_bin"] = pd.cut(df["latitude"],  bins=n_bins, labels=False)
    df["lon_bin"] = pd.cut(df["longitude"], bins=n_bins, labels=False)
    df["location_id"] = (
        df["latitude"].round(2).astype(str) + "_" +
        df["longitude"].round(2).astype(str)
    )
    df["location_id_enc"] = df.groupby("location_id").ngroup()

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pollutant ratio features (NO2/SO2, CO/NO2, HCHO/NO2),
    total pollutant load, and average cloud fraction across sensors.
    """
    df = df.copy()

    so2 = df["SulphurDioxide_SO2_column_number_density"].replace(0, np.nan)
    no2 = df["NitrogenDioxide_NO2_column_number_density"].replace(0, np.nan)
    co  = df["CarbonMonoxide_CO_column_number_density"].replace(0, np.nan)
    hcho = df["Formaldehyde_tropospheric_HCHO_column_number_density"].replace(0, np.nan)

    df["NO2_SO2_ratio"]  = no2 / so2
    df["CO_NO2_ratio"]   = co  / no2
    df["HCHO_NO2_ratio"] = hcho / no2

    key_cols = [
        "SulphurDioxide_SO2_column_number_density",
        "CarbonMonoxide_CO_column_number_density",
        "NitrogenDioxide_NO2_column_number_density",
        "Formaldehyde_tropospheric_HCHO_column_number_density",
    ]
    df["total_pollutant_load"] = df[key_cols].sum(axis=1, min_count=1)

    cloud_frac_cols = [c for c in df.columns if "cloud_fraction" in c]
    if cloud_frac_cols:
        df["cloud_mean"] = df[cloud_frac_cols].mean(axis=1)

    return df



def log_transform_features(df: pd.DataFrame,
                            features: List[str]) -> pd.DataFrame:
    """
    Applies log1p to highly right-skewed atmospheric concentration columns.
    Uses clip(lower=0) first to handle any negative artefacts from satellite
    retrieval algorithms.
    """
    df = df.copy()
    for col in features:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


class LocationStats:
    """
    Computes per-location (lat/lon) median and standard deviation of
    each key emission column.
    """
    def __init__(self, key_cols: List[str] = KEY_EMISSION_COLS):
        self.key_cols = key_cols
        self._stats: Dict[str, pd.DataFrame] = {}

    def fit(self, df: pd.DataFrame) -> "LocationStats":
        df = df.copy()
        df["_loc"] = (
            df["latitude"].round(2).astype(str) + "_" +
            df["longitude"].round(2).astype(str)
        )
        for col in self.key_cols:
            if col not in df.columns:
                continue
            agg = df.groupby("_loc")[col].agg(["median", "std"]).rename(
                columns={"median": f"{col}_loc_median", "std": f"{col}_loc_std"}
            )
            self._stats[col] = agg
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_loc"] = (
            df["latitude"].round(2).astype(str) + "_" +
            df["longitude"].round(2).astype(str)
        )
        for col, stats_df in self._stats.items():
            df = df.merge(stats_df, how="left", left_on="_loc", right_index=True)
        df = df.drop(columns=["_loc"], errors="ignore")
        return df

if __name__ == "__main__":
    print(KEY_EMISSION_COLS)