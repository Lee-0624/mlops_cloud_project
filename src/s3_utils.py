import boto3
import os
import pandas as pd
from io import BytesIO
import datetime as dt
from dateutil.relativedelta import relativedelta

def get_s3_client():
    """MinIO S3 클라이언트 생성"""
    return boto3.client(
        's3',
        endpoint_url=os.environ.get('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', 'minio'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'minio123'),
        region_name='us-east-1'
    )

def download_from_s3(bucket_name, object_key):
    """S3에서 parquet 파일을 다운로드하여 DataFrame으로 반환"""
    s3_client = get_s3_client()
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        buffer = BytesIO(response['Body'].read())
        df = pd.read_parquet(buffer)
        print(f"S3에서 다운로드 완료: s3://{bucket_name}/{object_key}")
        return df
    except Exception as e:
        print(f"S3 다운로드 실패: {e}")
        return None

def upload_to_s3(df, bucket_name, object_key):
    """DataFrame을 parquet으로 변환하여 S3에 업로드"""
    s3_client = get_s3_client()
    
    # 메모리 버퍼에 parquet 파일 생성
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    # S3에 업로드
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=buffer.getvalue()
    )
    print(f"S3에 업로드 완료: s3://{bucket_name}/{object_key}")

def list_s3_objects(bucket_name, prefix):
    """S3 버킷에서 특정 접두사로 시작하는 객체 목록 반환"""
    s3_client = get_s3_client()
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []
    except Exception as e:
        print(f"S3 객체 목록 조회 실패: {e}")
        return []

def download_latest_from_s3(bucket_name, prefix_template, date_str=None, fallback_to_latest=False):
    """
    지정된 날짜 기준으로 파일을 찾고, 옵션에 따라 가장 최근 파일을 다운로드
    
    Args:
        bucket_name: S3 버킷명
        prefix_template: 파일 경로 템플릿 (예: "ingest/ingest_{}.parquet")
        date_str: 특정 날짜 (없으면 오늘 날짜 사용)
        fallback_to_latest: 파일이 없을 때 최근 파일을 찾을지 여부 (기본값: False)
    
    Returns:
        DataFrame 또는 None
        
    Raises:
        FileNotFoundError: 파일을 찾을 수 없고 fallback_to_latest=False인 경우
    """
    # 날짜가 지정되지 않으면 오늘 날짜 사용
    if date_str is None:
        date_str = (dt.datetime.utcnow() + relativedelta(hours=9)).strftime("%Y%m%d")
    
    # 지정된 날짜 기준 파일 경로
    object_key = prefix_template.format(date_str)
    
    print(f"파일 다운로드 시도: s3://{bucket_name}/{object_key}")
    
    # 지정된 날짜 파일 다운로드 시도
    df = download_from_s3(bucket_name, object_key)
    
    if df is None:
        if not fallback_to_latest:
            # fallback 옵션이 False면 바로 예외 발생
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: s3://{bucket_name}/{object_key}")
        
        print(f"지정된 날짜 파일이 없습니다. 가장 최근 파일을 찾습니다.")
        
        # prefix에서 날짜 부분을 제거하여 검색용 prefix 생성
        # 예: "ingest/ingest_{}.parquet" -> "ingest/ingest_"
        search_prefix = prefix_template.split('{}')[0]
        
        # 가장 최근 파일 찾기
        files = list_s3_objects(bucket_name, search_prefix)
        if files:
            # 파일명에서 날짜 추출하여 정렬 (가장 최근 파일)
            files.sort(reverse=True)
            object_key = files[0]
            print(f"가장 최근 파일 사용: s3://{bucket_name}/{object_key}")
            df = download_from_s3(bucket_name, object_key)
        
        if df is None:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: s3://{bucket_name}/{search_prefix}*")
    
    return df 