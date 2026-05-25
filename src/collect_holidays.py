"""
공공데이터포털 특일 정보 API(한국천문연구원)로 2010-2025년 공휴일을 수집.

산출물: data/raw/holidays_2010_2025.csv
  columns: date (YYYY-MM-DD), name, is_holiday (Y/N), date_kind

수집 정의:
  - getRestDeInfo: 국경일·공휴일 (대체공휴일·임시공휴일 포함)
  - 본 분석에서는 is_holiday=Y만 사용 (24절기 등 평일은 제외)
  - 연 단위 호출 × 16년 = 16회
"""
from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "raw" / "holidays_2010_2025.csv"

# 가이드 명세상 HTTP 전용 (HTTPS 호출 시 403). docs/OpenAPI활용가이드..._v1.4.docx 참조.
API_URL = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
YEARS = list(range(2010, 2026))


def call_year(service_key: str, year: int, max_retries: int = 5) -> list[dict]:
    """한 해의 공휴일 목록 조회. JSON 우선, 실패 시 XML 폴백.

    공공데이터포털 키는 Encoding(이미 URL-encoded)과 Decoding 두 종류 발급됨.
    requests의 params= 는 자동 재인코딩하므로 Encoding 키를 그대로 넘기면
    이중 인코딩(%25%2B 등)으로 401. → serviceKey는 URL 문자열에 직접 붙이고
    나머지 파라미터만 params로 넘긴다.

    공공데이터포털은 산발적 403(rate limit 또는 게이트웨이 일시 차단)을
    반환할 수 있으므로 지수 백오프 재시도.
    """
    url = f"{API_URL}?serviceKey={service_key}"
    params = {
        "solYear": str(year),
        "numOfRows": "100",
        "_type": "json",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                break
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        wait = 2 ** attempt  # 1, 2, 4, 8, 16초
        print(f"   재시도 {attempt+1}/{max_retries} ({last_err[:60]}) — {wait}s 대기")
        time.sleep(wait)
    else:
        raise RuntimeError(f"{max_retries}회 재시도 후 실패: {last_err}")

    # JSON 시도
    try:
        data = resp.json()
        items = data["response"]["body"]["items"]
        if items == "" or items is None:
            return []
        items = items.get("item", [])
        if isinstance(items, dict):
            items = [items]
        return items
    except (ValueError, KeyError):
        # XML 폴백
        return parse_xml(resp.text)


def parse_xml(text: str) -> list[dict]:
    """XML 응답을 dict 목록으로."""
    root = ET.fromstring(text)
    items = []
    for item in root.iter("item"):
        d = {child.tag: child.text for child in item}
        items.append(d)
    return items


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("DATA_GO_KR_HOLIDAYS_KEY")
    if not api_key:
        sys.exit("ERROR: DATA_GO_KR_HOLIDAYS_KEY가 .env에 없습니다.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_items = []
    for year in YEARS:
        items = call_year(api_key, year)
        print(f"  {year}: {len(items):>2}개 항목")
        all_items.extend(items)
        time.sleep(1.0)

    df = pd.DataFrame(all_items)
    if df.empty:
        sys.exit("ERROR: 빈 응답")

    # 컬럼 표준화
    df["date"] = pd.to_datetime(df["locdate"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"dateName": "name", "isHoliday": "is_holiday", "dateKind": "date_kind"})
    df = df[["date", "name", "is_holiday", "date_kind"]].sort_values("date").reset_index(drop=True)

    # 검증
    print(f"\n총 항목: {len(df):,}")
    holidays = df[df["is_holiday"] == "Y"]
    print(f"is_holiday=Y: {len(holidays):,}")
    print(f"\n연도별 공휴일 수 (is_holiday=Y):")
    print(holidays["date"].str[:4].value_counts().sort_index().to_string())

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
