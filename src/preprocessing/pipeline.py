from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config

from src.preprocessing.transformers import (
    DropIdColumns,
    TemporalFeatureTransformer,
    SpatialFeatureTransformer,
    PollutantRatioTransformer,
    LocationStatsTransformer,
    LogTransformer,
    DropStringColumns,
    SensorImputer,
)

set_config(transform_output="pandas")

def build_data_preparation_pipeline() -> Pipeline:
    """
    Full preprocessing pipeline applied identically to train and test data.

    Step-by-step:
      1. DropIdColumns          — removes the ID_LAT_LON_YEAR_WEEK identifier
      2. TemporalFeatureTransformer — month, season, sin/cos encodings, dry season flag
      3. SpatialFeatureTransformer  — lat/lon bins, location integer encoding
      4. PollutantRatioTransformer  — NO2/SO2, CO/NO2, HCHO/NO2, total load
      5. LocationStatsTransformer   — per-location median/std (fit on train only)
      6. LogTransformer             — log1p on skewed atmospheric cols
      7. DropStringColumns          — removes any residual object columns
      8. SensorImputer              — median imputation of remaining NaNs
      9. StandardScaler             — zero-mean, unit-variance normalisation
    """
    return Pipeline(steps=[
        ("drop_ids",          DropIdColumns()),
        ("temporal",          TemporalFeatureTransformer()),
        ("spatial",           SpatialFeatureTransformer(n_bins=10)),
        ("ratios",            PollutantRatioTransformer()),
        ("location_stats",    LocationStatsTransformer()),
        ("log_transform",     LogTransformer()),
        ("drop_strings",      DropStringColumns()),
        ("imputer",           SensorImputer(strategy="median")),
        ("scaler",            StandardScaler()),
    ])