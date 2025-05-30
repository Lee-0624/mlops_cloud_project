import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from s3_utils import download_latest_from_s3, upload_to_s3

def run():
    """
    S3에서 저장된 날씨 관측 데이터를 읽어서 전처리 후 피처 파케이로 S3에 저장
    """
    bucket_name = "mlflow"
    
    # 최신 ingest 데이터 로드
    df = download_latest_from_s3(bucket_name, "ingest/ingest_{}.parquet")
    
    print(f"원본 날씨 관측 데이터 shape: {df.shape}")
    print(f"원본 데이터 columns: {df.columns.tolist()}")
    print(f"원본 데이터 인덱스 타입: {type(df.index)}")
    print(f"원본 데이터 인덱스 이름: {df.index.name}")
    print(f"원본 데이터 첫 5행:\n{df.head()}")
    
    # 데이터 전처리 및 피처 엔지니어링
    processed_df = preprocess_weather_data(df)
    print(f"전처리된 피처 데이터 shape: {processed_df.shape}")
    print(f"전처리된 피처 데이터 columns: {processed_df.columns.tolist()}")
    print(f"전처리된 피처 데이터: {processed_df.head()}")
    
    # S3에 피처 데이터 저장
    base_date = (dt.datetime.utcnow() + relativedelta(hours=9)).strftime("%Y%m%d")
    output_object_key = f"preprocess/preprocess_{base_date}.parquet"
    upload_to_s3(processed_df, bucket_name, output_object_key)
    print(f"전처리된 피처 데이터 저장 완료: s3://{bucket_name}/{output_object_key}")
    print(f"처리된 데이터 shape: {processed_df.shape}")
    
    return f"s3://{bucket_name}/{output_object_key}"

def preprocess_weather_data(df):
    """
    날씨 관측 데이터 전처리 및 피처 엔지니어링
    """
    # 데이터 복사
    processed_df = df.copy()
    
    # datetime 인덱스가 없는 경우 설정 (S3에서 다운로드한 데이터는 인덱스가 제거되어 있음)
    if not isinstance(processed_df.index, pd.DatetimeIndex):
        # 'datetime' 컬럼이 있는지 확인하고 인덱스로 설정
        if 'datetime' in processed_df.columns:
            processed_df['datetime'] = pd.to_datetime(processed_df['datetime'])
            processed_df = processed_df.set_index('datetime').sort_index()
        else:
            # datetime 컬럼이 없다면 에러 로그 출력
            print(f"ERROR: datetime 인덱스도 datetime 컬럼도 없습니다. 컬럼: {processed_df.columns.tolist()}")
            print(f"인덱스 타입: {type(processed_df.index)}")
            print(f"데이터 샘플:\n{processed_df.head()}")
            return processed_df.dropna()  # 빈 데이터프레임 반환
    
    # 1시간 빈도 보정
    processed_df = processed_df.asfreq("H")
    
    # 피처 엔지니어링 (new_lightgbm2.py의 build_features 함수와 동일)
    processed_df = build_features(processed_df)
    
    # 타겟 변수 생성 (24시간 후 예측)
    processed_df["ta_target"] = processed_df["ta"].shift(-24)   # 24h ahead 기온
    processed_df["hm_target"] = processed_df["hm"].shift(-24)   # 24h ahead 습도
    
    # 결측값이 있는 행 제거
    processed_df = processed_df.dropna()
    
    print(f"피처 엔지니어링 완료. 최종 데이터 shape: {processed_df.shape}")
    
    return processed_df

def build_features(df):
    """
    날씨 관측 데이터에서 피처 엔지니어링 수행
    new_lightgbm2.py의 build_features 함수와 동일한 로직
    """
    out = df[["ta", "hm"]].astype({"ta": float, "hm": float})
    
    # 시간 파생 변수
    out["hour"] = out.index.hour
    out["dow"]  = out.index.dayofweek
    out["sin_hour"] = np.sin(2*np.pi*out["hour"]/24)
    out["cos_hour"] = np.cos(2*np.pi*out["hour"]/24)
    
    # Lag 및 Rolling 피처
    for col in ["ta", "hm"]:
        out[f"{col}_lag1"]   = out[col].shift(1)
        out[f"{col}_lag24"]  = out[col].shift(24)
        out[f"{col}_roll3"]  = out[col].rolling(3).mean()
        out[f"{col}_roll24"] = out[col].rolling(24).mean()
    
    return out.dropna()

if __name__ == "__main__":
    run() 