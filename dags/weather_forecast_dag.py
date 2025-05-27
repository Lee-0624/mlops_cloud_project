from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import subprocess, os, mlflow, json

default_args = {"retries": 1}

def ingest():
    subprocess.run(["python", "src/data_ingest.py"], check=True)

def preprocess():
    # 간단 예시 – 실습에서는 데이터 읽어 전처리 후 /tmp/feature_*.parquet 저장
    pass

def train():
    subprocess.run(["python", "src/train.py"], check=True)

def evaluate():
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("weather_24h")
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
    best = runs[0]
    if best.metrics["rmse"] < 3:     # 임계값
        client.transition_model_version_stage(
            name="seoul_temp",
            version=best.info.version,
            stage="Production",
            archive_existing_versions=True,
        )
        return True
    return False

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
