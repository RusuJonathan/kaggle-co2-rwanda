import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from src.preprocessing.preprocessing_utils import (
    add_temporal_features,
    add_spatial_features,
    add_ratio_features,
    log_transform_features,
    LocationStats,
)
from src.data.data_loader import load_yaml, preprocessing_path

config = load_yaml(preprocessing_path)


class DropIdColumns(BaseEstimator, TransformerMixin):
    """Drops identifier columns that carry no signal"""

    def __init__(self, id_cols=None):
        self.id_cols = id_cols or config.get("ID_COLS", [])

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.id_cols if c in X.columns]
        return X.drop(columns=cols_to_drop)


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds calendar and cyclical temporal features from year and week_no"""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_temporal_features(X)


class SpatialFeatureTransformer(BaseEstimator, TransformerMixin):
    """Discretises lat/lon and adds a location integer encoding"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_spatial_features(X, n_bins=self.n_bins)


class PollutantRatioTransformer(BaseEstimator, TransformerMixin):
    """Adds physically-motivated ratio and composite pollutant features"""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_ratio_features(X)


class LocationStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Computes per-location median/std of key emission columns.
    Fit must be called only on training data to prevent leakage
    """

    def __init__(self):
        self._loc_stats = LocationStats()

    def fit(self, X, y=None):
        self._loc_stats.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._loc_stats.transform(X)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Applies log1p to highly skewed sensor columns"""

    def __init__(self, features=None):
        self.features = features or config.get("LOG_FEATURES", [])

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return log_transform_features(X, self.features)


class DropStringColumns(BaseEstimator, TransformerMixin):
    """
    Drops any remaining object/string columns that sklearn estimators
    cannot handle
    """

    def fit(self, X, y=None):
        self._string_cols = X.select_dtypes(include="object").columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self._string_cols, errors="ignore")


class SensorImputer(BaseEstimator, TransformerMixin):
    """
    Applies median imputation to all remaining numerical columns.
    Wrapped as a custom transformer to preserve pandas column names
    """

    def __init__(self, strategy: str = "median"):
        self.strategy = strategy
        self._imputer = SimpleImputer(strategy=strategy)

    def fit(self, X, y=None):
        self._imputer.fit(X.select_dtypes(include="number"))
        self._num_cols = X.select_dtypes(include="number").columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._num_cols] = self._imputer.transform(X[self._num_cols])
        return X
