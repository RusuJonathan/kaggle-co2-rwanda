# 🌍 Predict CO₂ Emissions in Rwanda

[![Kaggle](https://img.shields.io/badge/Kaggle-PS3E20-blue?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s3e20)
[![Data](https://img.shields.io/badge/Data-Sentinel--5P%20Satellite-orange)](https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Stacking ensemble with Optuna hyperparameter optimisation trained on **ESA Sentinel-5P satellite observations** to predict weekly CO₂ emissions across Rwanda (2019–2022).

---

## 📊 Results

| Metric | Score              |
|---|--------------------|
| CV RMSE (5-fold) | **~142**           |
| Kaggle Public Score (RMSE) | Competition Closed |

---

## 🛰️ Problem & Context

Accurate carbon monitoring is critical for climate policy and energy transition planning. Ground-based sensors are sparse across Africa, making satellite-derived estimates essential. This project uses open-source data from the **TROPOMI instrument aboard Sentinel-5P**

Each observation represents a **geographic cell × week × year** combination with ~75 atmospheric features across 7 sensor products:

| Sensor Product | Key Variable | Physical Meaning |
|---|---|---|
| L3_SO2 | `SO2_column_number_density` | Sulphur dioxide — industrial / volcanic |
| L3_CO | `CO_column_number_density` | Carbon monoxide — combustion tracer |
| L3_NO2 | `tropospheric_NO2_column_number_density` | Nitrogen dioxide — traffic + industry |
| L3_HCHO | `HCHO_column_number_density` | Formaldehyde — VOC oxidation |
| L3_AER_AI | `absorbing_aerosol_index` | UV aerosol — biomass burning |
| L3_O3 | `O3_column_number_density` | Ozone — photochemical activity |
| L3_CLOUD | `cloud_fraction` | Cloud cover — data quality proxy |

---

## 📁 Project Structure

```
co2-emissions-rwanda/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── config/
│   │   ├── preprocessing_config.yaml   # Column groups, feature lists, log targets
│   │   ├── model_config.yaml           # HPO search spaces per model
│   │   └── best_hyperparams.yaml       # Saved best hyperparameters (auto-generated)
│   ├── data/
│   │   └── data_loader.py              # CSV loading, YAML config loader, path constants
│   ├── preprocessing/
│   │   ├── pipeline.py                 # Full sklearn preprocessing pipeline builder
│   │   ├── transformers.py             # Custom sklearn transformers
│   │   └── preprocessing_utils.py      # Feature engineering functions
│   └── models/
│       ├── model.py                    # Main Model class (fit / predict / optimize)
│       ├── models_builders.py          # Pipeline & stacking model factory functions
│       └── hpo_tuner.py                # Optuna HPO objective & study runner
├── notebooks/
│   └── eda_co2_rwanda.ipynb            # Exploratory Data Analysis (11 sections)
├── main.py                             # Entry point: optimize → fit → predict → submit
└── README.md
```

---

## 🧠 Approach

### 1 — Preprocessing Pipeline

Each base model is wrapped in a unified `sklearn.Pipeline` with these steps in order:

| Step | Transformer | What it does |
|---|---|---|
| 1 | `DropIdColumns` | Removes `ID_LAT_LON_YEAR_WEEK` identifier |
| 2 | `TemporalFeatureTransformer` | `month`, `season`, `week_sin/cos`, `month_sin/cos`, `is_dry_season`, `year_normalized` |
| 3 | `SpatialFeatureTransformer` | `lat_bin`, `lon_bin`, `location_id_enc` (integer-encoded lat/lon key) |
| 4 | `PollutantRatioTransformer` | `NO2/SO2`, `CO/NO2`, `HCHO/NO2` ratios; `total_pollutant_load`, `cloud_mean` |
| 5 | `LocationStatsTransformer` | Per-location median & std for key emission columns — **fit on train only** |
| 6 | `LogTransformer` | `log1p` on the 5 most skewed concentration columns |
| 7 | `DropStringColumns` | Removes residual object columns after encoding |
| 8 | `SensorImputer` | Median imputation for satellite NaN gaps (orbital misses, clouds) |
| 9 | `StandardScaler` | Zero-mean, unit-variance normalisation |
| 10 | `TransformedTargetRegressor` | `log1p` on target at train time, `expm1` at predict time |

### 2 — Stacking Ensemble

A `StackingRegressor` combines six base learners, each with its own independent preprocessing pipeline:

| Model | Library | Role |
|---|---|---|
| XGBoost | xgboost | Gradient boosting on decision trees |
| LightGBM | lightgbm | Fast gradient boosting, strong on high-dimensional data |
| CatBoost | catboost | Gradient boosting with ordered boosting |
| Random Forest | scikit-learn | Bagging ensemble, reduces variance |

**Meta-learner:** `ElasticNet` — regularised linear combination of base model out-of-fold predictions.

### 3 — Hyperparameter Optimisation

All base model hyperparameters are tuned with **Optuna**:

- **300 trials** per base model
- **5-fold cross-validation** with `neg_root_mean_squared_error` objective
- **TPE sampler** with `MedianPruner` to terminate unpromising trials early
- Search spaces defined in `model_config.yaml` (IntUniform, LogUniform, Uniform, Categorical)
- Best parameters auto-saved to `best_hyperparams.yaml`

### 4 — Key Feature Engineering Insights (from EDA)

| Feature | Motivation |
|---|---|
| `week_sin` / `week_cos` | Circular encoding — week 52 and week 1 are adjacent in Rwanda's cycle |
| `location_id_enc` | Geographic zones have persistent emission baselines |
| `loc_median` stats | Per-location historical median captures long-run spatial patterns |
| `NO2_SO2_ratio` | Distinguishes traffic-dominated (high NO₂) from industrial (high SO₂) zones |
| `CO_NO2_ratio` | Combustion completeness proxy |
| `log1p` on concentrations | All key pollutant measurements are right-skewed; log-normalisation improves tree splits |



---

## 📦 Dependencies

Managed with **Poetry** (`pyproject.toml`):

```
scikit-learn  xgboost  lightgbm  catboost
optuna  pandas  numpy  pyyaml
matplotlib  seaborn  scipy  shap
```

---

## 📓 EDA Highlights

Key findings from `notebooks/eda_co2_rwanda.ipynb` (11 sections):

- `emission` is heavily right-skewed → `log1p` transformation via `TransformedTargetRegressor`
- **NO₂ tropospheric column** is the single strongest predictor of CO₂ emission level
- Strong **spatial clustering** — some geographic zones persistently emit 3–5× more than others
- **Pollutant ratios** (NO₂/SO₂, CO/NO₂) carry information independent of raw concentrations

---

## 📄 License

This project is licensed under the MIT License.
