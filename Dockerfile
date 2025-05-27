FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
CMD ["python", "-m", "pip", "install", "--upgrade", "pip"]
