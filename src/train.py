import lightgbm as lgb, pandas as pd, mlflow, os, numpy as np
from sklearn.metrics import mean_squared_error
from s3_utils import download_latest_from_s3

# MLflow tracking URI 설정
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print(f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")

# 실험 생성 또는 설정
experiment_name = "weather_24h"
try:
    # 기존 실험이 있는지 확인
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # 실험이 없으면 생성
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"새로운 실험 '{experiment_name}' 생성됨 (ID: {experiment_id})")
    else:
        print(f"기존 실험 '{experiment_name}' 사용 (ID: {experiment.experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    
    # 디버깅: 모든 실험 목록 출력
    client = mlflow.MlflowClient()
    all_experiments = client.search_experiments()
    print(f"현재 존재하는 모든 실험들: {[exp.name for exp in all_experiments]}")
    
except Exception as e:
    print(f"실험 설정 중 오류 발생: {e}")
    raise

def load_feature_df() -> pd.DataFrame:
    """S3에서 가장 최신 전처리된 데이터를 로드"""
    bucket_name = "mlflow"
    return download_latest_from_s3(bucket_name, "preprocess/preprocess_{}.parquet")

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
                          callbacks=[lgb.early_stopping(30)])

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        mlflow.log_metric("rmse", rmse)

        # 모델을 아티팩트로 로깅
        model_info = mlflow.lightgbm.log_model(model, artifact_path="model")
        
        # 모델을 Model Registry에 등록
        model_name = "seoul_temp"
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
        
        print(f"모델 훈련 완료. RMSE: {rmse}")
        print(f"모델이 '{model_name}'으로 등록되었습니다.")

if __name__ == "__main__":
    main()
