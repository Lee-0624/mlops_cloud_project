import lightgbm as lgb, pandas as pd, mlflow, os, numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("weather_24h")

def load_feature_df() -> pd.DataFrame:
    path = sorted(Path("/tmp").glob("feature_*.parquet"))[-1]
    return pd.read_parquet(path)

def main():
    df = load_feature_df()
    X = df.drop(columns=["target_temp"])
    y = df["target_temp"]
    split = int(len(df)*0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    with mlflow.start_run():
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {"objective": "regression", "metric": "rmse"}
        model = lgb.train(params, lgb_train, valid_sets=[lgb_val],
                          early_stopping_rounds=30)

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = mean_squared_error(y_val, preds, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.lightgbm.log_model(model, artifact_path="model")
        print("RMSE:", rmse)

if __name__ == "__main__":
    main()
