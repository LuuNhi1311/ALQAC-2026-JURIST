import time

import requests

URL = "https://alqac-api.ngrok.pro/retrieve"
API_KEY = "alqac_Subc4BDmE1TQdZD8Sst8lKKeJkUKqqOy"


payload = {
    "query": "hợp đồng chuyển nhượng đất",
    "case_id": "case_1615",
}


def retrieve(payload, max_retries=5):
    for attempt in range(1, max_retries + 1):
        resp = requests.post(
            URL,
            headers={"X-API-Key": API_KEY},
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429:
            print(f"[{attempt}] rate limited, chờ 6s...")
            time.sleep(6)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        return resp.json()
    raise RuntimeError("Vẫn bị rate limit sau nhiều lần thử")


data = retrieve(payload)
print(data)
