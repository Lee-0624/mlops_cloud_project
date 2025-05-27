from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import subprocess

default_args = {"retries": 1}

def ingest():
    subprocess.run(["python", "src/data_ingest.py"], check=True)

def preprocess():
    # ingest에서 저장된 데이터를 읽어 전처리 후 /tmp/feature_*.parquet 저장
    subprocess.run(["python", "src/preprocess.py"], check=True)

def train():
    subprocess.run(["python", "src/train.py"], check=True)

def evaluate():
    # 모델 평가 및 프로덕션 배포 결정
    subprocess.run(["python", "src/evaluate.py"], check=True)

with DAG("weather_daily",
         start_date=datetime(2025, 5, 1),
         schedule_interval="0 2 * * *",
         default_args=default_args,
         catchup=False) as dag:

    t1 = PythonOperator(task_id="ingest", python_callable=ingest)
    t2 = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t3 = PythonOperator(task_id="train", python_callable=train)
    t4 = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t5 = BashOperator(task_id="reload_api",
                      bash_command="curl -X POST http://api:8000/reload_model")

    t1 >> t2 >> t3 >> t4 >> t5
