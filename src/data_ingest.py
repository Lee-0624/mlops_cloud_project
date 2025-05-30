import requests, os, pandas as pd, datetime as dt, sys, json
from dateutil.relativedelta import relativedelta
from s3_utils import upload_to_s3

def run():
    """ASOS 날씨 관측 시간자료(ta: 기온, hm: 습도) 수집 후 S3에 저장"""
    # 지난 30일 데이터 수집 범위 설정
    today_kst = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)
    start_day = (today_kst - relativedelta(days=30)).date()
    end_day = (today_kst - relativedelta(days=1)).date()  # D-1 23:00 까지
    
    # ASOS 날씨 관측 데이터 수집
    df = fetch_weather_data_hourly(start_day, end_day)
    
    # MinIO S3에 저장
    bucket_name = "mlflow"
    base_date = today_kst.strftime("%Y%m%d")
    object_key = f"ingest/ingest_{base_date}.parquet"
    upload_to_s3(df, bucket_name, object_key)
    
    print(f"ASOS 날씨 관측 데이터 수집 완료: {len(df)}개 행")
    print(f"데이터 저장 완료: s3://{bucket_name}/{object_key}")

def fetch_weather_data_hourly(start, end):
    """ASOS 날씨 관측 시간자료(ta: 기온, hm: 습도) 수집 -> DataFrame(index=datetime)"""
    SERVICE_KEY = os.getenv("KMA_API_KEY")          # 기상청 인증키
    STATION_ID  = "108"  # 지점 코드 (108: 서울, 133: 대전, 159: 부산, 184: 대구, 206: 광주, 233: 제주)
    BASE_URL    = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"  # https://www.data.go.kr/data/15057210/openapi.do

    params = {
        "ServiceKey": SERVICE_KEY,
        "dataCd": "ASOS", "dateCd": "HR",
        "startDt": start.strftime("%Y%m%d"), "startHh": "00",
        "endDt":   end.strftime("%Y%m%d"),   "endHh":   "23",
        "stnIds":  STATION_ID,
        "dataType": "JSON", "numOfRows": "999", "pageNo": "1"
    }
    
    print(f"ASOS 날씨 관측 API 호출: {start} ~ {end}")
    print(f"BASE_URL: {BASE_URL}")
    print(f"API 호출 파라미터: {mask_service_key(params)}")
    r = requests.get(BASE_URL, params=params, timeout=15)
    
    try:
        r_json = r.json()
        print("API Response JSON:")
        print(json.dumps(r_json, indent=2, ensure_ascii=False))
        r.raise_for_status()
        items = r_json["response"]["body"]["items"]["item"]
        df = pd.DataFrame(items)
        df["datetime"] = pd.to_datetime(df["tm"])
        return df.set_index("datetime").sort_index().drop(columns=["tm"])
    except Exception as e:
        print(r.text)
        print(f"Error: {e}")
        raise

def mask_service_key(params):
    masked_params = params.copy()
    masked_params["ServiceKey"] = masked_params["ServiceKey"][:8] + "..." + masked_params["ServiceKey"][-8:]
    return masked_params

if __name__ == "__main__":
    run()
