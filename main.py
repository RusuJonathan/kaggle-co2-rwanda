import pandas as pd
from src.models.model import Model
from sklearn.model_selection import cross_val_score
from src.data.data_loader import (
    load_yaml,
    load_data,
    train_path,
    test_path,
    model_config_path,
    best_hyperparams_path,
    target,
)

#The competition is closed and submissions are no longer accepted, so I will evaluate using only a 5-fold cross-validation.

df = load_data(train_path)

config = load_yaml(model_config_path)

x_train = df.drop(columns=[target])
y_train = df[target]

model = Model(config=config, n_trials=50)

#model.optimize_hyperparameters(x_train, y_train)

best_hyperparams = load_yaml(best_hyperparams_path)

model.load_hyperparameters(hyperparameter_config=best_hyperparams)

scores = cross_val_score(estimator=model,
                         X=x_train,
                         y=y_train,
                         scoring="neg_root_mean_squared_error",
                         cv=5)

print(f"CV Mean: {scores.mean():.4f} ± {scores.std():.4f}")