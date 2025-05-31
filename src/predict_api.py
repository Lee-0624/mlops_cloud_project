from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import mlflow, pandas as pd, os, numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from src.s3_utils import download_latest_from_s3
from src.db_utils import init_db, save_prediction, get_latest_prediction

# MLflow 추적 URI 설정
mlflow.set_tracking_uri("http://localhost:5000")

app = FastAPI()

# 정적 파일 서비스 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 변수 초기화
_temp_model = None
_humid_model = None

@app.on_event("startup")
def startup():
    init_db()        # ← SQLite DB 초기화 추가
    load_models()    # ← 기존 모델 로딩 유지

@app.get("/")
def read_root():
    """루트 경로에서 index.html 반환"""
    return FileResponse('static/index.html')

def load_models():
    """MLflow에서 프로덕션 모델 로딩"""
    global _temp_model, _humid_model
    try:
        _temp_model = mlflow.pyfunc.load_model("models:/seoul_temp/Production")
        print("기온 예측 모델 로드 성공")
        
        _humid_model = mlflow.pyfunc.load_model("models:/seoul_humid/Production")
        print("습도 예측 모델 로드 성공")
        
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        _temp_model = None
        _humid_model = None

def get_latest_features():
    """S3에서 최신 피처 데이터 가져오기"""
    try:
        bucket_name = "mlflow"
        feature_df = download_latest_from_s3(bucket_name, "preprocess/preprocess_{}.parquet")
        latest_features = feature_df.iloc[-1:].copy()
        print(f"최신 피처 데이터 로드 완료: {latest_features.shape}")
        return latest_features
    except Exception as e:
        print(f"최신 피처 데이터 로드 실패: {e}")
        return None

@app.get("/predict")
def predict():
    """최신 피처로 예측 후 DB 저장"""
    if _temp_model is None or _humid_model is None:
        return {"error": "모델이 로드되지 않았습니다. 먼저 모델을 훈련하고 등록해주세요."}
    
    latest_features = get_latest_features()
    if latest_features is None:
        return {"error": "최신 피처 데이터를 가져올 수 없습니다."}
    
    try:
        temp_pred = _temp_model.predict(latest_features)[0]
        humid_pred = _humid_model.predict(latest_features)[0]
        tomorrow = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9) + dt.timedelta(days=1)).date()
        
        # SQLite에 예측 결과 저장
        save_prediction(str(tomorrow), temp_pred, humid_pred)
        
        return {
            "forecast_date": str(tomorrow),
            "temperature": round(float(temp_pred), 1),
            "humidity": round(float(humid_pred), 0),
            "message": f"내일({tomorrow}) 예상 기온 {temp_pred:.1f}℃ / 습도 {humid_pred:.0f}%"
        }
    except Exception as e:
        print(f"예측 중 오류: {e}")
        return {"error": f"예측 중 오류: {str(e)}"}

@app.get("/api/latest")
def latest():
    """SQLite에서 최신 예측 조회"""
    row = get_latest_prediction()
    if row:
        return {
            "forecast_date": row[0],
            "temperature": round(float(row[1]), 1),
            "humidity": round(float(row[2]), 0)
        }
    return {"error": "예측 데이터가 없습니다."}

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
