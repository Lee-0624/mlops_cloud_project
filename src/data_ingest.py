import requests, os, pandas as pd, datetime as dt, sys
from dateutil.relativedelta import relativedelta

KMA_API_KEY = os.environ["KMA_API_KEY"]

# BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
# BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
BASE_URL =  "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"


def run():
    base_date = (dt.datetime.utcnow() + relativedelta(hours=9)).strftime("%Y%m%d")
    api_key_name = "authKey" if BASE_URL.startswith("https://apihub.kma.go.kr") else "serviceKey"
    params = {
        api_key_name: KMA_API_KEY,
        "pageNo": "1", "numOfRows": "1000",
        "dataType": "JSON",
        "base_date": base_date, "base_time": "0200",
        "nx": "60", "ny": "127"
    }
    print(f"BASE_URL: {BASE_URL}, params: {params}")
    r = requests.get(BASE_URL, params=params, timeout=10)
    print(f"r: {r}")
    print(f"r.text: {r.text}")
    r.raise_for_status()
    items = r.json()["response"]["body"]["items"]["item"]
    df = pd.DataFrame(items)
    out = f"/tmp/weather_{base_date}.parquet"
    df.to_parquet(out, index=False)
    print(out)

if __name__ == "__main__":
    run()
