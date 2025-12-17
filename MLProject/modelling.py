import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # WAJIB AUTLOG
    mlflow.sklearn.autolog()

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

    print("RMSE:", rmse)
    print("R2:", r2)

if __name__ == "__main__":
    main()
