import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_baseline_regressors():

    # Load preprocessed train, validation, and test splits
    X_train = np.load("data/modeling/X_train.npy")
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")
    y_train = np.load("data/modeling/y_train_returns.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")

    # Define simple search spaces for each model and validate the set used for tuning to avoid test leakage
    param_grids = {
        "Ridge_Regression": {
            "model_class": Ridge,
            "params": [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1.0},
                {"alpha": 10.0},
            ]
        },
        "RF_Regressor": {
            "model_class": RandomForestRegressor,
            "params": [
                {"n_estimators": 100, "max_depth": 5, "random_state": 42},
                {"n_estimators": 200, "max_depth": 10, "random_state": 42},
                {"n_estimators": 200, "max_depth": 15, "random_state": 42},
                {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 10, "random_state": 42},
            ]
        },
        "GB_Regressor": {
            "model_class": GradientBoostingRegressor,
            "params": [
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
                {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "random_state": 42},
                {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "random_state": 42},
                {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.01, "random_state": 42},
            ]
        },
    }

    # Include a basic linear model as a benchmark reference
    baselines = {"Linear_Regression": LinearRegression()}
    

    # Ensure output directories exist before saving anything
    os.makedirs("models/baselines", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    results = []
    tuning_logs = []

    print(f"Training & Tuning Baseline Regressors")
    print("-" * 70)

    # Train linear regression without tuning as a baseline check
    lr = baselines["Linear_Regression"]
    lr.fit(X_train, y_train)
    y_pred_test = lr.predict(X_test)
    y_pred_val = lr.predict(X_val)
    
    # Evaluate performance across validation and test sets
    val_mae = mean_absolute_error(y_val, y_pred_val)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_dir_acc = np.mean((y_pred_test > 0) == (y_test > 0))
    
    # Save trained model for later use
    joblib.dump(lr, "models/baselines/Linear_Regression.pkl")
    results.append({
        "Model": "Linear_Regression", "Best_Params": "N/A",
        "Val_MAE": val_mae, "Test_MAE": test_mae, "Test_RMSE": test_rmse,
        "Test_R2": test_r2, "Test_DirAcc": test_dir_acc
    })
    print(f"Linear_Regression  | Val MAE: {val_mae:.6f} | Test MAE: {test_mae:.6f} | DirAcc: {test_dir_acc:.2%}")

    # Loop through each model and perform simple validation-based tuning
    for model_name, config in param_grids.items():
        print(f"\nTuning {model_name}")
        best_val_mae = float('inf')
        best_model = None
        best_params = None
        
        for params in config["params"]:
            model = config["model_class"](**params)
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            
            tuning_logs.append({
                "model": model_name,
                "params": str(params),
                "val_mae": val_mae
            })
            
            print(f"  Params: {params} -> Val MAE: {val_mae:.6f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model = model
                best_params = params
        
        # Evaluate best model on unseen test data
        y_pred_test = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        test_dir_acc = np.mean((y_pred_test > 0) == (y_test > 0))
        
        # Save the best model for this configuration
        joblib.dump(best_model, f"models/baselines/{model_name}.pkl")

        # Store final evaluation results
        results.append({
            "Model": model_name, "Best_Params": str(best_params),
            "Val_MAE": best_val_mae, "Test_MAE": test_mae, "Test_RMSE": test_rmse,
            "Test_R2": test_r2, "Test_DirAcc": test_dir_acc
        })
        
        print(f"  >> Best: {best_params}")
        print(f"  >> Test MAE: {test_mae:.6f} | RMSE: {test_rmse:.6f} | R2: {test_r2:.4f} | DirAcc: {test_dir_acc:.2%}")


    # Save final results for reporting and analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/results/baseline_regression_results.csv", index=False)
    

    # Save full tuning history for reproducibility
    tuning_df = pd.DataFrame(tuning_logs)
    tuning_df.to_csv("data/results/hyperparameter_tuning_log.csv", index=False)
    
    print("All baseline models trained, tuned, and saved.")
    print(results_df.to_string(index=False))
    return results_df

if __name__ == "__main__":
    train_baseline_regressors()