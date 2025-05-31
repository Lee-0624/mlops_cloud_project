from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import mlflow, pandas as pd, os, numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from src.s3_utils import download_latest_from_s3


# MLflow 추적 URI 설정
mlflow.set_tracking_uri("http://localhost:5000")

app = FastAPI()

# 정적 파일 서비스 설정 (반드시 FastAPI 선언 후에 위치)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:5500"] 등 HTML 실행 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static 파일 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

_temp_model = None
_humid_model = None

@app.get("/")
def read_root():
    """루트 경로에서 index.html 반환"""
    return FileResponse('static/index.html')

def load_models():
    """프로덕션 모델들 로드"""
    global _temp_model, _humid_model
    try:
        # 기온 예측 모델 로드
        _temp_model = mlflow.pyfunc.load_model("models:/seoul_temp/Production")
        print("기온 예측 모델 로드 성공")
        
        # 습도 예측 모델 로드
        _humid_model = mlflow.pyfunc.load_model("models:/seoul_humid/Production")
        print("습도 예측 모델 로드 성공")
        
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        _temp_model = None
        _humid_model = None

def get_latest_features():
    """S3에서 최신 날씨 관측 피처 데이터를 가져와서 마지막 행 반환"""
    try:
        bucket_name = "mlflow"
        feature_df = download_latest_from_s3(bucket_name, "preprocess/preprocess_{}.parquet")
        
        # 가장 최신 데이터 (마지막 행) 반환
        latest_features = feature_df.iloc[-1:].copy()
        
        print(f"최신 날씨 관측 피처 데이터 로드 완료: {latest_features.shape}")
        return latest_features
        
    except Exception as e:
        print(f"최신 피처 데이터 로드 실패: {e}")
        return None

@app.on_event("startup")
def startup():
    load_models()

@app.get("/predict")
def predict():
    """최신 날씨 관측 데이터를 사용하여 24시간 후 기온과 습도 예측"""
    if _temp_model is None or _humid_model is None:
        return {"error": "모델이 로드되지 않았습니다. 먼저 모델을 훈련하고 프로덕션에 등록해주세요."}
    
    # 최신 피처 데이터 가져오기
    latest_features = get_latest_features()
    if latest_features is None:
        return {"error": "최신 날씨 관측 데이터를 가져올 수 없습니다."}
    
    try:
        # 예측 수행
        temp_pred = _temp_model.predict(latest_features)[0]
        humid_pred = _humid_model.predict(latest_features)[0]
        
        # 내일 날짜 계산
        tomorrow = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9) + dt.timedelta(days=1)).date()
        
        return {
            "forecast_date": str(tomorrow),
            "temperature": round(float(temp_pred), 1),
            "humidity": round(float(humid_pred), 0),
            "message": f"내일({tomorrow}) 예상 기온 {temp_pred:.1f}℃ / 습도 {humid_pred:.0f}%"
        }
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return {"error": f"예측 중 오류가 발생했습니다: {str(e)}"}

@app.post("/reload_model")
def reload_model():
    """모델 재로드"""
    load_models()
    status = "성공" if (_temp_model is not None and _humid_model is not None) else "실패"
    return {"status": f"모델 재로드 {status}"}

@app.get("/health")
def health_check():
    """API 상태 확인"""
    temp_model_status = "로드됨" if _temp_model is not None else "로드 안됨"
    humid_model_status = "로드됨" if _humid_model is not None else "로드 안됨"
    
    return {
        "status": "정상",
        "temp_model": temp_model_status,
        "humid_model": humid_model_status
    }
