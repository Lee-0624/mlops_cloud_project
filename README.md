# 🌤️ 서울 날씨 예측 MLOps 시스템  
**실시간 기사데이터 기반 AI 예측과 자동화된 MLOps 파이프라인 구축**

<p align="center">
  <img src="static/happy_plant.png" width="100" alt="Happy Plant" />
</p>

---

## 📌 프로젝트 개요
- **프로젝트 기간**: 2주  
- **목표**: 서울 지역 기사 데이터(기온, 습도)를 활용해서 **AI 예측 모델**을 만들고, 이를 자동화된 **MLOps 파이프라인**에서 서비스하는 시스템 구축  
- **개인 기여**:  
  - **MLflow 운영** 및 모델 버전 관리
  - **모델 자동 배포 파이프라인 설계**
  - **DB 관리** 및 API 통합

---

## 🌟 주요 기능
| 기능                  | 설명 |
|----------------------|------|
| 🛁 데이터 수집         | 서울 ASOS 관찰소 API 이용 (시간별 기온/습도 수집) |
| 🧹 데이터 전처리       | 시간 기반 변수 생성, Lag 피처, Rolling 평균 적용 |
| 🤖 머신롤링           | LightGBM 이중 모델 (기온/습도 각각 예측) |
| 🛠️ 모델 관리          | MLflow를 통해 실험 추적, 모델 성능 평가, 자동 배포 |
| 🌐 API & 웹 대시보드  | FastAPI 기반 예측 API 및 실시간 웹 UI 제공 |
| ♻ 자동화 파이프라인   | Airflow로 매일 살정 2시 자동 실행, 예측 결과 저장 |
| 📃 데이터 저장         | MinIO (데이터), SQLite (예측 결과) |

---

## 🤓 개인 기여 상세 (이준석)
- **MLflow 실험 관리**
  - 모델 학원 시 실험 기록 및 버전 관리 구현
  - 성능 감사 모델만 자동 배포되도록 파이프라인 구성
- **자동 배포 시스템 설계**
  - Airflow에서 모델 성능 평가 후 `/reload_model` 호출
  - FastAPI 서버가 자동으로 최신 모델 적용
- **DB 관리 및 통합**
  - 예측 결과를 SQLite에 저장 및 API로 조회 가능하게 구현
  - `/api/latest` 엔드포인트를 통해 시간별 확인 지원

---

## 🛠️ 기술 스테크

| 카테고리       | 사용 기술 |
|----------------|-----------|
| 언어           | Python 3.11 |
| 머신롤링       | LightGBM 4.3.0, Pandas, NumPy |
| 실험 관리      | MLflow 2.11.0 |
| 웹 서비스/API | FastAPI, Uvicorn |
| 파이프라인     | Apache Airflow 2.9.0 |
| 켄테이너       | Docker Compose |
| 스토리지/DB    | MinIO (S3 호환), SQLite |
| 기타           | 기사청 ASOS API, GitHub, Notion |

---

## 📂 프로젝트 구조 (개인 리포지토리 기준)

```
mlops-weather-prediction/
├── src/
│   ├── data_ingest.py      # 데이터 수집
│   ├── preprocess.py       # 전처리 및 피처 엔지니어링
│   ├── train.py            # 모델 학습
│   ├── evaluate.py         # 성능 평가 및 자동 배포
│   ├── predict_api.py      # FastAPI 예측 API
│   ├── s3_utils.py         # MinIO 연동
│   └── db_utils.py         # SQLite DB 관리
├── dags/
│   └── weather_forecast_dag.py
├── static/
│   ├── index.html          # 웹 대시보드
│   ├── happy_plant.png
│   └── water_please.png
├── predictions_data/
│   └── predictions.db      # 예측 이력 DB
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## 🚀 실행 방법

1. **환경 설정**
   ```bash
   git clone https://github.com/Lee-0624/mlops-weather-prediction.git
   cd mlops-weather-prediction
   ```

2. **환경 변수 설정 (.env)**
   ```bash
   KMA_API_KEY=your_api_key
   MLFLOW_TRACKING_URI=http://localhost:5000
   MINIO_ROOT_USER=minio
   MINIO_ROOT_PASSWORD=minio123
   AWS_ACCESS_KEY_ID=minio
   AWS_SECRET_ACCESS_KEY=minio123
   ```

3. **서비스 실행**
   ```bash
   docker-compose up -d
   ```

4. **접속 주소**
   - Airflow: http://localhost:8080
   - MLflow: http://localhost:5000
   - FastAPI Docs: http://localhost:8000/docs
   - 웹 대시보드: http://localhost:8000

5. **예측 테스트**
   ```bash
   curl http://localhost:8000/predict
   ```

---

## 🧠 회고 및 학습 성과
- MLOps 플랫폼의 핵심 기술인 **MLflow, Airflow, Docker**에 대한 심화 경험
- **지속적 성능 모니터링 및 자동화 배포**의 중요성 체감
- 협업 프로젝트에서 **명확한 역할 분담과 문서화의 가치** 확인
- 실제 서비스 수준의 시스템 구현 경험 → 실무 역량 향상

---

## 📌 기타
- 팀 프로젝트 원본 리포지토리: [팀 GitHub 링크](https://github.com/AIBootcamp13/mlops-cloud-project-mlops_5)  
- 본 개인 리포지토리는 **개인 기여 및 학습 내용 중심으로 정리**됨

---

## ⭐ 기텍버 방문 감사합니다!
> 이 프로젝트가 도움이 되셨다면 ⭐를 눌러주세요!  
> 문의: [GitHub @Lee-0624](https://github.com/Lee-0624)
