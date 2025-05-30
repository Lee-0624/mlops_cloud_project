import lightgbm as lgb, pandas as pd, mlflow, os, numpy as np
from sklearn.metrics import mean_squared_error
from s3_utils import download_latest_from_s3

# MLflow tracking URI 설정
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print(f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")


def run():
    """날씨 관측 데이터로 기온과 습도 예측 모델 학습"""
    # 실험 설정
    set_experiment()

    # 데이터 로드
    df = load_feature_df()
    
    # 타겟 변수 생성 (24시간 후 예측)
    df["ta_target"] = df["ta"].shift(-24)   # 24h ahead 기온
    df["hm_target"] = df["hm"].shift(-24)   # 24h ahead 습도
    
    # 결측값이 있는 행 제거
    df = df.dropna()

    # 피처와 타겟 분리
    X = df.drop(columns=["ta_target", "hm_target"])
    y_temp = df["ta_target"]
    y_humid = df["hm_target"]
    
    print(f"🤖 날씨 관측 데이터로 모델 학습 시작")
    print(f"피처 수: {X.shape[1]}, 데이터 수: {len(X)}")

    with mlflow.start_run():
        # 기온 예측 모델 학습
        print("🌡️ 기온 예측 모델 학습 중...")
        model_temp, rmse_temp = train_lightgbm(X, y_temp, "기온")
        
        # 습도 예측 모델 학습  
        print("💧 습도 예측 모델 학습 중...")
        model_humid, rmse_humid = train_lightgbm(X, y_humid, "습도")
        
        # 평균 RMSE 계산 (전체 모델 성능 지표)
        avg_rmse = (rmse_temp + rmse_humid) / 2
        
        # MLflow 메트릭 로깅
        mlflow.log_metric("rmse_temp", rmse_temp)
        mlflow.log_metric("rmse_humid", rmse_humid)
        mlflow.log_metric("rmse", avg_rmse)  # 기존 평가 로직과 호환성을 위해 평균 RMSE 사용

        # 기온 모델을 주 모델로 아티팩트 로깅 (기존 코드와 호환성)
        model_info = mlflow.lightgbm.log_model(model_temp, artifact_path="model_temp")
        
        # 습도 모델도 별도로 로깅
        mlflow.lightgbm.log_model(model_humid, artifact_path="model_humid")
        
        # 모델을 Model Registry에 등록
        model_name = "seoul_temp"
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_temp"
        mlflow.register_model(model_uri, model_name)
        
        # 습도 모델도 별도 이름으로 등록
        humid_model_name = "seoul_humid"
        humid_model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_humid"
        mlflow.register_model(humid_model_uri, humid_model_name)
        
        print(f"🎉 모델 훈련 완료!")
        print(f"기온 모델 RMSE: {rmse_temp:.3f}")
        print(f"습도 모델 RMSE: {rmse_humid:.3f}")
        print(f"평균 RMSE: {avg_rmse:.3f}")
        print(f"모델이 '{model_name}', '{humid_model_name}'으로 등록되었습니다.")


def train_lightgbm(X, y, target_name):
    """8:2 holdout 방식으로 모델 학습"""
    # 8:2 비율로 데이터 분할 (시간순)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"📊 {target_name} 데이터 분할: 학습 {len(X_train)}개 / 테스트 {len(X_test)}개")
    
    params = dict(
        objective="regression",
        learning_rate=0.01,
        num_leaves=10,
        n_estimators=2000,
        min_child_samples=5,
        verbose=-1
    )

    # 모델 학습
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # 테스트 데이터 평가
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    print(f"🔍 {target_name} 모델 테스트 성능:")
    print(f"   테스트 RMSE: {rmse:.3f}")
    print(f"   실제값 범위: {y_test.min():.1f} ~ {y_test.max():.1f}")
    print(f"   예측값 범위: {pred.min():.1f} ~ {pred.max():.1f}")
    
    return model, rmse


def load_feature_df() -> pd.DataFrame:
    """S3에서 가장 최신 날씨 관측 전처리된 데이터를 로드"""
    bucket_name = "mlflow"
    return download_latest_from_s3(bucket_name, "preprocess/preprocess_{}.parquet")


def set_experiment():
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


if __name__ == "__main__":
    run()
