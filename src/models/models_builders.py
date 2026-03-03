from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor
import numpy as np
from typing import Dict, List, Tuple, Any

from src.preprocessing.pipeline import build_data_preparation_pipeline

def build_pipeline(
        model_class: Any,
        param: Dict[str, Any] = {}) -> Pipeline:

    return Pipeline(steps=[
        ("preprocessing", build_data_preparation_pipeline()),
        ("model", TransformedTargetRegressor(
            regressor=model_class(**param),
            func=np.log1p,
            inverse_func=np.expm1,
        ))
    ])

def build_stacking_model(
        final_model_estimator: Any,
        estimators: List[Tuple[str, Any]],
        param: Dict[str, Any] = {}) -> StackingRegressor:

    final_estimator = final_model_estimator(**param)
    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
    )


def set_base_model_params(
        pipeline: Pipeline,
        param: Dict[str, Any]) -> None:
    pipeline.named_steps["model"].regressor.set_params(**param)


def set_stacking_model_params(
        stacking: StackingRegressor,
        param: Dict[str, Any]) -> None:
    stacking.final_estimator.set_params(**param)