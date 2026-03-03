import pandas as pd
from typing import Dict
from pathlib import Path

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.models_builders import (
    build_pipeline,
    set_base_model_params,
    set_stacking_model_params,
    build_stacking_model,
)
from src.models.hpo_tuner import (
    run_hyperparameter_optimization,
    save_best_hyperparameters,
)
from src.data.data_loader import load_yaml, load_data, train_path, model_config_path


class Model(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble with Optuna-tuned base learners.

    Base models (each wrapped in its own preprocessing pipeline):
      - ElasticNet
      - SVR
      - XGBoost
      - LightGBM
      - CatBoost
      - Random Forest

    Meta-learner: ElasticNet (chosen for its regularisation properties
    on the potentially correlated base model predictions).

    Target is log-transformed inside each pipeline via
    TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1).
    """

    def __init__(self, config: Dict, n_trials: int = 300):
        self.config  = config
        self.n_trials = n_trials
        self._build_pipeline = build_pipeline
        self._run_hyperparameter_optimization = run_hyperparameter_optimization
        self._build_stacking_model = build_stacking_model
        self.set_base_model_params = set_base_model_params
        self.set_stacking_model_params = set_stacking_model_params
        self.save_best_hyperparameters = save_best_hyperparameters
        self._init_models()

    def _init_models(self):
        self.elasticnet  = ("elasticnet",self._build_pipeline(ElasticNet,{"max_iter": 1000}))
        self.svr = ("svr",self._build_pipeline(SVR))
        self.xgboost = ("xgboost",self._build_pipeline(XGBRegressor,{"verbosity": 0}))
        self.lgbm = ("lgbm",self._build_pipeline(LGBMRegressor,{"verbosity": -1}))
        self.catboost = ("catboost",self._build_pipeline(CatBoostRegressor, {"verbose": 0}))
        self.randomforest = ("random_forest",self._build_pipeline(RandomForestRegressor,{"n_jobs": -1}))

        self.estimators = [
            self.xgboost,
            self.catboost,
            self.randomforest,
            self.lgbm,
        ]

        self.stacking_model = StackingRegressor(
            estimators=self.estimators,
            final_estimator=ElasticNet(max_iter=10000),
        )

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> None:
        for model_name, pipeline in self.estimators:
            print(f"\n🔍 Optimising {model_name} ({self.n_trials} trials)...")
            study = self._run_hyperparameter_optimization(
                model=pipeline,
                X=X,
                y=y,
                model_config=self.config[model_name],
                n_trials=self.n_trials,
            )
            self.set_base_model_params(pipeline, study.best_params)
            self.save_best_hyperparameters(study=study, model_name=model_name)
            print(f"Best RMSE: {study.best_value:.4f}")

    def load_hyperparameters(self, hyperparameter_config: Dict) -> "Model":
        for model_name, pipeline in self.estimators:
            if model_name in hyperparameter_config:
                self.set_base_model_params(pipeline, hyperparameter_config[model_name])
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.stacking_model.predict(X)

    def get_all_params(self) -> Dict:
        all_params = {}
        for name, pipeline in self.stacking_model.estimators:
            all_params[name] = pipeline.named_steps["model"].regressor.get_params()
        all_params["meta_model"] = self.stacking_model.final_estimator.get_params()
        return all_params

if __name__ == "__main__":
    config  = load_yaml(model_config_path)
    df      = load_data(train_path)
    x_train = df.drop(columns=["emission"])
    y_train = df["emission"]

    model = Model(config=config, n_trials=5)
    model.optimize_hyperparameters(x_train, y_train)

    scores = cross_val_score(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring="neg_root_mean_squared_error",
        cv=5,
    )
    print(f"\nCV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")