import json
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import joblib 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    # Init DagsHub MLflow
    dagshub.init(
        repo_owner="har-neptune",
        repo_name="world-happiness-mlflow",
        mlflow=True
    )

    # Load data
    X_train = np.load("namadataset_preprocessing/X_train.npy")
    X_test = np.load("namadataset_preprocessing/X_test.npy")
    y_train = np.load("namadataset_preprocessing/y_train.npy")
    y_test = np.load("namadataset_preprocessing/y_test.npy")

    # Parameter grid (sederhana & aman)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }

    best_rmse = float("inf")
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=f"rf_tuning_{params['n_estimators']}_{params['max_depth']}"):
            model = RandomForestRegressor(
                random_state=42,
                **params
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log params & metrics
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Track best
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = params

    # Final run for best model
    with mlflow.start_run(run_name="rf_best_model"):
        mlflow.log_params(best_params)

        y_pred_best = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred_best)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred_best)
        r2 = r2_score(y_test, y_pred_best)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        joblib.dump(best_model, "model.pkl")
        mlflow.log_artifact("model.pkl")

        # Artefact 1: Feature Importance
        plt.figure(figsize=(8, 5))
        importances = best_model.feature_importances_
        plt.barh(range(len(importances)), importances)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature Index")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()

        mlflow.log_artifact("feature_importance.png")

        # Artefact 2: Prediction vs Actual
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred_best, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediction vs Actual")
        plt.tight_layout()
        plt.savefig("prediction_vs_actual.png")
        plt.close()

        mlflow.log_artifact("prediction_vs_actual.png")

        # Artefact 3 (opsional tapi bagus): Best params
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        mlflow.log_artifact("best_params.json")


if __name__ == "__main__":
    main()
