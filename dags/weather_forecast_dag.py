from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import pytz

default_args = {"retries": 1}

# KST timezone 설정
kst = pytz.timezone('Asia/Seoul')

with DAG("weather_daily",
         start_date=datetime(2025, 5, 1, tzinfo=kst),
         schedule_interval="0 2 * * *",  # 매일 오전 2시 실행
         default_args=default_args,
         catchup=False) as dag:

    # ASOS 날씨 관측 데이터 수집 (기온, 습도)
    t1 = BashOperator(task_id="ingest", 
                      bash_command="python /opt/airflow/src/data_ingest.py")
    
    # ASOS 날씨 관측 데이터 전처리 및 피처 엔지니어링
    t2 = BashOperator(task_id="preprocess", 
                      bash_command="python /opt/airflow/src/preprocess.py")
    
    # LightGBM 모델 학습 (기온, 습도 예측)
    t3 = BashOperator(task_id="train", 
                      bash_command="python /opt/airflow/src/train.py")
    
    # 모델 성능 평가 및 프로덕션 배포
    t4 = BashOperator(task_id="evaluate", 
                      bash_command="python /opt/airflow/src/evaluate.py")
    
    # 예측 API 모델 재로드
    t5 = BashOperator(task_id="reload_api",
                      bash_command="curl -X POST http://mlflow:8000/reload_model")

    t1 >> t2 >> t3 >> t4 >> t5
