import numpy as np
import mlflow
import dagshub

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    # Init DagsHub MLflow
    dagshub.init(
        repo_owner="har-neptune",
        repo_name="world-happiness-mlflow",
        mlflow=True
    )

    # Load preprocessed data
    X_train = np.load("namadataset_preprocessing/X_train.npy")
    X_test = np.load("namadataset_preprocessing/X_test.npy")
    y_train = np.load("namadataset_preprocessing/y_train.npy")
    y_test = np.load("namadataset_preprocessing/y_test.npy")

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    with mlflow.start_run(run_name="rf_baseline"):
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)


        # Manual logging
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
