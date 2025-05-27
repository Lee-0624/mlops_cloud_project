FROM python:3.11-slim

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# MLflow 데이터 디렉토리 생성
RUN mkdir -p /app/mlflow_data

# 환경 변수 설정
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# 포트 노출
EXPOSE 5000 8000

CMD ["python", "-m", "pip", "install", "--upgrade", "pip"]
