import requests, os, pandas as pd, datetime as dt, sys
from dateutil.relativedelta import relativedelta

KMA_KEY = os.environ["KMA_API_KEY"]
BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

def run():
    base_date = (dt.datetime.utcnow() + relativedelta(hours=9)).strftime("%Y%m%d")
    params = {
        "serviceKey": KMA_KEY,
        "pageNo": "1", "numOfRows": "1000",
        "dataType": "JSON",
        "base_date": base_date, "base_time": "0200",
        "nx": "60", "ny": "127"
    }
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    items = r.json()["response"]["body"]["items"]["item"]
    df = pd.DataFrame(items)
    out = f"/tmp/weather_{base_date}.parquet"
    df.to_parquet(out, index=False)
    print(out)

if __name__ == "__main__":
    run()
