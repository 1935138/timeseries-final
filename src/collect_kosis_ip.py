"""
KOSIS 전산업생산지수(원지수, DT_1JH20201) 월별 수집.

출처: 통계청 산업활동동향, 2020=100
산출: data/raw/kosis_ip_total_2010_2025.csv
  columns: prd (YYYYMM), c1, c1_name, value

수집 정의:
  - tblId=DT_1JH20201 (전산업생산지수 원지수)
  - itmId=T1 (원지수)
  - objL1=ALL (전산업·광공업·서비스업·건설업·공공행정 5개 분류 모두)
  - 본 분석에서는 C1=0 (전산업)만 외생변수로 사용
  - 발표 지연(약 익월 말) → lag 처리는 wide-table 통합 단계에서

C1 분류 코드 매핑(통계청 산업활동동향 표준):
  0  = 전산업
  1B = 광공업
  1C = 서비스업
  1D = 건설업
  1E = 공공행정
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "raw" / "kosis_ip_total_2010_2025.csv"

API_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
ORG_ID = "101"
TBL_ID = "DT_1JH20201"

C1_NAMES = {
    "0": "전산업",
    "1B": "광공업",
    "1C": "서비스업",
    "1D": "건설업",
    "1E": "공공행정",
}

START_PRD = "201001"
END_PRD = "202512"


def fetch(api_key: str, start: str, end: str, max_retries: int = 3) -> list[dict]:
    """KOSIS에서 [start, end] 구간 전체 수집. params dict가 outputFields 등을
    안정적으로 인식하지 않는 사례가 있어 URL 문자열에 직접 인자 부착."""
    url = (
        f"{API_URL}?method=getList"
        f"&apiKey={api_key}"
        f"&itmId=T1+&objL1=ALL"
        f"&format=json&jsonVD=Y&prdSe=M"
        f"&startPrdDe={start}&endPrdDe={end}"
        f"&orgId={ORG_ID}&tblId={TBL_ID}"
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                last_err = f"unexpected payload: {str(data)[:200]}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except (requests.RequestException, ValueError) as e:
            last_err = str(e)
        wait = 2 ** attempt
        print(f"   재시도 {attempt+1}/{max_retries} ({last_err[:80]}) — {wait}s 대기")
        time.sleep(wait)
    raise RuntimeError(f"{max_retries}회 재시도 후 실패: {last_err}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("KOSIS_API_KEY")
    if not api_key:
        sys.exit("ERROR: KOSIS_API_KEY가 .env에 없습니다.")

    print(f"수집: {TBL_ID} (전산업생산지수 원지수)  {START_PRD} ~ {END_PRD}")
    items = fetch(api_key, START_PRD, END_PRD)
    print(f"수신: {len(items):,}건 (= 분류 5 × 월 ?)")

    df = pd.DataFrame(
        [
            {
                "prd": d["PRD_DE"],
                "c1": d["C1"],
                "c1_name": C1_NAMES.get(d["C1"], d["C1"]),
                "value": float(d["DT"]),
            }
            for d in items
        ]
    )
    df = df.sort_values(["c1", "prd"]).reset_index(drop=True)

    print(f"\n분류별 행 수:")
    print(df.groupby(["c1", "c1_name"]).size().to_string())
    print(f"\n기간(전산업): {df.loc[df['c1']=='0', 'prd'].min()} ~ {df.loc[df['c1']=='0', 'prd'].max()}")

    total = df[df["c1"] == "0"].copy()
    print(f"\n전산업 sanity check:")
    print(f"  2020-01 값(=100 근처여야): {total.loc[total['prd']=='202001', 'value'].iloc[0]}")
    print(f"  2010-01: {total.loc[total['prd']=='201001', 'value'].iloc[0]}")
    print(f"  최근 12개월:")
    print(total.tail(12).to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
