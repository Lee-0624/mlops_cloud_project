import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import os
import glob

def run():
    """
    ingest에서 저장된 날씨 데이터를 읽어서 전처리 후 피처 파케이로 저장
    """
    # 오늘 날짜 기준으로 ingest에서 저장된 파일 찾기
    base_date = (dt.datetime.utcnow() + relativedelta(hours=9)).strftime("%Y%m%d")
    input_file = f"/tmp/weather_{base_date}.parquet"
    
    print(f"전처리 시작: {input_file}")
    
    # ingest에서 저장된 파케이 파일 읽기
    if not os.path.exists(input_file):
        print(f"입력 파일이 존재하지 않습니다: {input_file}")
        # 가장 최근 파일 찾기
        pattern = "/tmp/weather_*.parquet"
        files = glob.glob(pattern)
        if files:
            input_file = max(files, key=os.path.getctime)
            print(f"가장 최근 파일 사용: {input_file}")
        else:
            raise FileNotFoundError(f"날씨 데이터 파일을 찾을 수 없습니다: {pattern}")
    
    df = pd.read_parquet(input_file)
    print(f"원본 데이터 shape: {df.shape}")
    print(f"원본 데이터 columns: {df.columns.tolist()}")
    
    # 데이터 전처리
    processed_df = preprocess_weather_data(df)
    print(f"전처리된 피처 데이터 shape: {processed_df.shape}")
    print(f"전처리된 피처 데이터 columns: {processed_df.columns.tolist()}")
    print(f"전처리된 피처 데이터: {processed_df}")
    
    # 피처 파케이로 저장
    output_file = f"/tmp/feature_{base_date}.parquet"
    processed_df.to_parquet(output_file, index=False)
    print(f"전처리된 피처 데이터 저장 완료: {output_file}")
    print(f"처리된 데이터 shape: {processed_df.shape}")
    
    return output_file

def preprocess_weather_data(df):
    """
    날씨 데이터 전처리 함수
    """
    # 데이터 복사
    processed_df = df.copy()
    
    # 날짜/시간 관련 피처 생성
    if 'fcstDate' in processed_df.columns and 'fcstTime' in processed_df.columns:
        processed_df['datetime'] = pd.to_datetime(
            processed_df['fcstDate'].astype(str) + processed_df['fcstTime'].astype(str).str.zfill(4),
            format='%Y%m%d%H%M'
        )
        processed_df['hour'] = processed_df['datetime'].dt.hour
        processed_df['day_of_week'] = processed_df['datetime'].dt.dayofweek
        processed_df['month'] = processed_df['datetime'].dt.month
    
    # 수치형 데이터 변환
    numeric_columns = ['fcstValue']
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # 카테고리별 피처 분리 및 피벗
    if 'category' in processed_df.columns:
        # 카테고리별로 피벗하여 각 예보 요소를 별도 컬럼으로 생성
        pivot_df = processed_df.pivot_table(
            index=['baseDate', 'baseTime', 'fcstDate', 'fcstTime', 'nx', 'ny'],
            columns='category',
            values='fcstValue',
            aggfunc='first'
        ).reset_index()
        
        # 컬럼명 정리
        pivot_df.columns.name = None
        processed_df = pivot_df
    
    # 결측값 처리
    processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
    
    # 온도 관련 피처 엔지니어링 (TMP가 있는 경우)
    if 'TMP' in processed_df.columns:
        processed_df['TMP'] = pd.to_numeric(processed_df['TMP'], errors='coerce')
        # 온도 범주화
        processed_df['temp_category'] = pd.cut(
            processed_df['TMP'], 
            bins=[-float('inf'), 0, 10, 20, 30, float('inf')],
            labels=['매우추움', '추움', '보통', '따뜻함', '더움']
        )
    
    # 습도 관련 피처 (REH가 있는 경우)
    if 'REH' in processed_df.columns:
        processed_df['REH'] = pd.to_numeric(processed_df['REH'], errors='coerce')
        processed_df['humidity_high'] = (processed_df['REH'] > 70).astype(int)
    
    # 강수 관련 피처 (PCP가 있는 경우)
    if 'PCP' in processed_df.columns:
        # 강수량이 문자열인 경우 처리
        processed_df['PCP_numeric'] = processed_df['PCP'].replace('강수없음', '0')
        processed_df['PCP_numeric'] = pd.to_numeric(processed_df['PCP_numeric'], errors='coerce')
        processed_df['has_precipitation'] = (processed_df['PCP_numeric'] > 0).astype(int)
    
    # 풍속 관련 피처 (WSD가 있는 경우)
    if 'WSD' in processed_df.columns:
        processed_df['WSD'] = pd.to_numeric(processed_df['WSD'], errors='coerce')
        processed_df['wind_strong'] = (processed_df['WSD'] > 5).astype(int)
    
    # 시간대별 피처
    if 'datetime' in processed_df.columns:
        processed_df['is_morning'] = ((processed_df['hour'] >= 6) & (processed_df['hour'] < 12)).astype(int)
        processed_df['is_afternoon'] = ((processed_df['hour'] >= 12) & (processed_df['hour'] < 18)).astype(int)
        processed_df['is_evening'] = ((processed_df['hour'] >= 18) & (processed_df['hour'] < 24)).astype(int)
        processed_df['is_night'] = ((processed_df['hour'] >= 0) & (processed_df['hour'] < 6)).astype(int)
        processed_df['is_weekend'] = (processed_df['day_of_week'] >= 5).astype(int)
    
    # 타겟 변수 생성 (24시간 후 온도 예측을 위한 타겟)
    if 'TMP' in processed_df.columns and 'datetime' in processed_df.columns:
        # 24시간 후 온도를 타겟으로 설정
        processed_df = processed_df.sort_values('datetime')
        processed_df['target_temp'] = processed_df['TMP'].shift(-24)  # 24시간 후 온도
        # 타겟이 없는 마지막 24개 행 제거
        processed_df = processed_df.dropna(subset=['target_temp'])
    elif 'TMP' in processed_df.columns:
        # datetime이 없는 경우 단순히 다음 행의 온도를 타겟으로 사용
        processed_df['target_temp'] = processed_df['TMP'].shift(-1)
        processed_df = processed_df.dropna(subset=['target_temp'])
    else:
        # TMP 컬럼이 없는 경우 더미 타겟 생성 (실제 사용시에는 적절한 타겟 설정 필요)
        processed_df['target_temp'] = np.random.normal(15, 5, len(processed_df))
        print("경고: TMP 컬럼이 없어 더미 타겟을 생성했습니다.")
    
    # 불필요한 컬럼 제거 (원본 문자열 컬럼들)
    columns_to_drop = ['PCP'] if 'PCP' in processed_df.columns else []
    if columns_to_drop:
        processed_df = processed_df.drop(columns=columns_to_drop)
    
    # LightGBM이 처리할 수 없는 object 타입 컬럼들 제거
    object_columns = processed_df.select_dtypes(include=['object']).columns.tolist()
    datetime_columns = ['datetime'] if 'datetime' in processed_df.columns else []
    
    # 제거할 컬럼들 (문자열 컬럼들과 datetime 컬럼)
    columns_to_remove = object_columns + datetime_columns
    columns_to_remove = [col for col in columns_to_remove if col in processed_df.columns]
    
    if columns_to_remove:
        processed_df = processed_df.drop(columns=columns_to_remove)
        print(f"LightGBM 호환성을 위해 제거된 컬럼들: {columns_to_remove}")
    
    # 수치형 데이터만 남기기
    processed_df = processed_df.select_dtypes(include=[np.number])
    
    print(f"전처리 완료.")
    
    return processed_df

if __name__ == "__main__":
    run() 