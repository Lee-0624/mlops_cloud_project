# 🌤️ 서울 날씨 예측 MLOps 시스템 

> **실시간 기상 데이터 기반 AI 예측과 자동화된 MLOps 파이프라인 구축**

<p align="center">
  <img src="static/happy_plant.png" width="100" alt="Happy Plant" />
</p>

---

## 📌 프로젝트 개요

- **개발 기간**: 2주 
- **목표**: 서울 지역 기상 데이터(기온, 습도)를 활용한 **AI 예측 모델 개발** 및 자동화된 **MLOps 시스템 구축**
- **역할 및 기여**:
  - ML 모델의 **생명주기 관리 및 자동 배포 시스템** 구축
  - **MLflow 기반 실험 관리 및 버전 관리** 담당
  - **예측 결과 DB 구축 및 API 통합 운영**

---

## 🔧 기술 스택

| 범주             | 기술 및 도구 |
|------------------|--------------|
| 프로그래밍 언어  | Python 3.11 |
| 머신러닝         | LightGBM, Pandas, NumPy |
| 실험 관리 및 MLOps | MLflow 2.11.0, Apache Airflow 2.9.0 |
| 웹 서비스       | FastAPI, Uvicorn |
| 컨테이너        | Docker, Docker Compose |
| 데이터 저장소    | MinIO (S3 호환), SQLite |
| 협업 및 버전관리 | Git, GitHub |

---

## 💻 시스템 구성 및 기능

### 🔍 주요 기능 요약

| 기능               | 설명 |
|--------------------|------|
| 데이터 수집        | 서울 ASOS 관측소 API 활용 (기온/습도 수집 자동화) |
| 전처리 및 피처 생성 | 시간 기반 변수, Lag 피처, 이동평균 처리 |
| 모델 학습         | LightGBM 기반 기온/습도 이중 예측 모델 개발 |
| 실험 및 모델 관리 | MLflow로 실험 추적, 모델 성능 평가 및 자동 배포 |
| API & 웹 대시보드 | FastAPI 기반 실시간 예측 API 및 웹 대시보드 제공 |
| 자동화 파이프라인 | Airflow DAG로 매일 새벽 자동 실행 및 예측 저장 |

### 🧩 개인 기여 세부

- **MLflow 운영 및 자동 배포 시스템 구현**  
  모델 학습 후 성능(RMSE)을 평가하여 기존 모델 대비 향상된 경우 자동으로 배포되도록 설정하였습니다. 또한, FastAPI 서버에서 `/reload_model` API를 호출하여 최신 모델을 동적으로 리로드할 수 있는 시스템을 개발하였습니다.

- **예측 결과 저장 및 조회 기능 개발**  
  SQLite를 기반으로 예측 결과를 저장하는 데이터베이스를 구축하였습니다. FastAPI를 통해 `/api/latest` 엔드포인트를 제공하여 예측 결과를 실시간으로 확인할 수 있도록 구현하였고, 이를 웹 대시보드와 연동하여 사용자에게 직관적인 예측 결과 제공이 가능하도록 구현하였습니다.

---

## 📁 프로젝트 구조 (요약)

```
mlops-weather-prediction/
├── src/                    # MLOps 핵심 로직
│   ├── train.py            # 모델 학습
│   ├── evaluate.py         # 성능 평가 및 배포
│   ├── predict_api.py      # FastAPI 예측 API
│   ├── db_utils.py         # SQLite 연동
├── dags/                   # Airflow DAG
│   └── weather_forecast_dag.py
├── static/                 # 웹 UI 리소스
├── predictions_data/       # 예측 결과 DB 저장
├── docker-compose.yaml     # 전체 스택 오케스트레이션
└── README.md
```

---

## 🚀 실행 방법

1. **클론 및 환경 설정**
```bash
git clone https://github.com/Lee-0624/mlops-weather-prediction.git
cd mlops-weather-prediction
```

2. **환경 변수 파일 설정 (.env)**
```bash
KMA_API_KEY=your_api_key
MLFLOW_TRACKING_URI=http://localhost:5000
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
```

3. **Docker로 서비스 실행**
```bash
docker-compose up -d
```

4. **접속 주소 요약**
- Airflow: http://localhost:8080
- MLflow UI: http://localhost:5000
- FastAPI Docs: http://localhost:8000/docs
- 웹 대시보드: http://localhost:8000

5. **예측 실행 테스트**
```bash
curl http://localhost:8000/predict
```

---

## 🧠 회고 및 학습 성과

- **MLOps 전반에 대한 실무 경험**: 단순한 모델 개발을 넘어서, 데이터 수집부터 배포까지 전체 파이프라인을 직접 설계하고 운영해보며 MLOps에 대한 실질적인 이해를 높일 수 있었습니다.
- **자동화의 중요성 실감**: 모델 성능 기준에 따라 자동 배포가 이루어지는 시스템을 구현함으로써 운영 효율성과 유지보수의 편의성을 체감하였습니다.
- **협업과 문서화의 가치**: 명확한 역할 분담과 체계적인 문서화가 프로젝트의 완성도와 팀워크 향상에 크게 기여함을 실감하였습니다.

---

## 📎 참고 링크

- 📂 팀 프로젝트 원본 리포지토리: [mlops-cloud-project-mlops_5](https://github.com/AIBootcamp13/mlops-cloud-project-mlops_5)
- 📧 Contact: [GitHub Profile](https://github.com/Lee-0624)
