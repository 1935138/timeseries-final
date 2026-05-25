"""
KOSIS OpenAPI로 17개 시·도 연간 주민등록인구(2010~2025)를 수집.

산출물: data/raw/kosis_population_sido_2010_2025.csv
  columns: year, sido_code, sido_name, population

수집 정의 (docs/research_log/2026-05-25_대표관측소_인구가중평균.md):
  - 통계: DT_1YL20651E 주민등록인구(시도/시/군/구) — 행정안전부
  - 시점: 연도 말(12월말) — KOSIS 연별 통계의 기준 시점
  - 단위: 17개 시·도 (세종은 가중치 산출 단계에서 충남에 통합)

연도 매핑:
  - KOSIS "y년" 값(y-12-31 기준) → 그대로 y년 가중치로 사용.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_CSV = RAW_DIR / "kosis_population_sido_2010_2025.csv"

KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
ORG_ID = "101"           # 행정안전부
TBL_ID = "DT_1YL20651E"  # 주민등록인구(시도/시/군/구)
ITM_ID = "T20"           # T20=계, T21=남, T22=여 — 총인구만 필요
PRD_SE = "Y"             # 연별

# 전국(00) + 17개 시·도 코드 (KOSIS objL1 명시 호출용)
SIDO_OBJL1 = (
    "00+11+26+27+28+29+30+31+36+41+51+43+44+52+46+47+48+50+"
)
# 가중치 적용 연도 = KOSIS 통계 시점 (연말 기준값을 그대로 그 해 가중치로 사용)
YEARS = list(range(2010, 2026))


def call_kosis(api_key: str, start: str, end: str) -> list[dict]:
    params = {
        "method": "getList",
        "apiKey": api_key,
        "format": "json",
        "jsonVD": "Y",
        "orgId": ORG_ID,
        "tblId": TBL_ID,
        "objL1": SIDO_OBJL1,
        "objL2": "",
        "objL3": "",
        "objL4": "",
        "itmId": ITM_ID,
        "prdSe": PRD_SE,
        "startPrdDe": start,
        "endPrdDe": end,
    }
    resp = requests.get(KOSIS_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("err"):
        raise RuntimeError(f"KOSIS API error: {data}")
    return data


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("KOSIS_API_KEY")
    if not api_key:
        sys.exit("ERROR: KOSIS_API_KEY가 .env에 없습니다.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start, end = str(YEARS[0]), str(YEARS[-1])
    print(f"KOSIS 호출: {start} ~ {end} (연별, {len(YEARS)}개 시점)")
    rows = call_kosis(api_key, start, end)
    print(f"  응답 행 수: {len(rows):,}")

    df = pd.DataFrame(rows)

    # 시·도 단위(2자리 코드)만 추출 — itmId=T20(계)만 호출했으므로 항목 필터 불필요
    df = df[df["C1"].str.len() == 2].copy()

    df["year"] = df["PRD_DE"].astype(int)
    df["population"] = pd.to_numeric(df["DT"], errors="coerce").astype("Int64")
    df = df.rename(columns={"C1": "sido_code", "C1_NM": "sido_name"})

    # 전국(00) 제외 — 검증용으로만 사용
    nationwide = df[df["sido_code"] == "00"][["year", "population"]]
    df = df[df["sido_code"] != "00"].copy()

    df = df[["year", "sido_code", "sido_name", "population"]]
    df = df.sort_values(["year", "sido_code"]).reset_index(drop=True)

    # 검증
    print(f"\n시·도 데이터: {len(df):,}행 (이론 최대 17 × 16 = 272)")
    by_year = df.groupby("year")["sido_code"].nunique()
    print("\n연도별 시·도 수 (세종 2012년 출범 → 그 이전 16개 정상):")
    print(by_year.to_string())

    # 합산 검증: 시·도 합 ≈ 전국(00) 값
    print("\n검증: 시·도 합 vs 전국(00)")
    sum_by_year = df.groupby("year")["population"].sum()
    check = nationwide.set_index("year")["population"].rename("nationwide")
    diff = pd.concat([sum_by_year.rename("sum_sido"), check], axis=1)
    diff["delta"] = diff["sum_sido"] - diff["nationwide"]
    print(diff.to_string())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT_CSV} ({len(df):,}행)")


if __name__ == "__main__":
    main()
