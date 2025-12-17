import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    X_train = np.load("worldhappiness_preprocessing/X_train.npy")
    X_test = np.load("worldhappiness_preprocessing/X_test.npy")
    y_train = np.load("worldhappiness_preprocessing/y_train.npy")
    y_test = np.load("worldhappiness_preprocessing/y_test.npy")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    # LOGGING TANPA START_RUN & SET_EXPERIMENT
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        model, 
        artifact_path="model",
        registered_model_name="CI World Hapiness Training")


if __name__ == "__main__":
    main()
