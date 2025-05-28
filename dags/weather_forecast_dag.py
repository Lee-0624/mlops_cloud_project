from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {"retries": 1}

with DAG("weather_daily",
         start_date=datetime(2025, 5, 1),
         schedule_interval="0 2 * * *",
         default_args=default_args,
         catchup=False) as dag:

    t1 = BashOperator(task_id="ingest", 
                      bash_command="python /opt/airflow/src/data_ingest.py")
    t2 = BashOperator(task_id="preprocess", 
                      bash_command="python /opt/airflow/src/preprocess.py")
    t3 = BashOperator(task_id="train", 
                      bash_command="python /opt/airflow/src/train.py")
    t4 = BashOperator(task_id="evaluate", 
                      bash_command="python /opt/airflow/src/evaluate.py")
    t5 = BashOperator(task_id="reload_api",
                      bash_command="curl -X POST http://api:8000/reload_model")

    t1 >> t2 >> t3 >> t4 >> t5
