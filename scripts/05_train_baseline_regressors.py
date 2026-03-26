import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_baseline_regressors():
    X_train = np.load("data/modeling/X_train.npy")
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")

    y_train = np.load("data/modeling/y_train_returns.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    param_grids = {
        "Linear_Regression": {
            "model_class": LinearRegression,
            "params": [{}]
        },
        "Ridge_Regression": {
            "model_class": Ridge,
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1.0},
                {"alpha": 10.0},
            ]
        },
        "Random_Forest": {
            "model_class": RandomForestRegressor,
            "params": [
                {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1},
                {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1},
            ]
        },
        "Gradient_Boosting": {
            "model_class": GradientBoostingRegressor,
            "params": [
                {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
                {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3, "random_state": 42},
            ]
        }
    }

    results = {}

    for model_name, config in param_grids.items():
        best_mae = float("inf")
        best_params = None

        for params in config["params"]:
            model = config["model_class"](**params)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_preds)

            if val_mae < best_mae:
                best_mae = val_mae
                best_params = params

        final_model = config["model_class"](**best_params)
        final_model.fit(X_train_full, y_train_full)
        test_preds = final_model.predict(X_test)

        results[model_name] = {
            "best_params": best_params,
            "val_mae": best_mae,
            "test_mae": mean_absolute_error(y_test, test_preds),
            "test_rmse": mean_squared_error(y_test, test_preds) ** 0.5,
            "test_r2": r2_score(y_test, test_preds)
        }

    print("Final baseline models trained on train + val")
    print(results)

if __name__ == "__main__":
    train_baseline_regressors()