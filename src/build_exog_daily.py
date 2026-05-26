"""
외생변수 wide-table 통합.

입력 (모두 5,844행, tm 키, data/processed/):
  - weather_nationwide_2010_2025.csv     기상 인구가중 전국평균
  - cdd_hdd_2010_2025.csv                CDD/HDD 6변수
  - feels_like_daily_2010_2025.csv       체감온도 일집계
  - extreme_flags_2010_2025.csv          폭염·한파 임계 더미 4개
  - calendar_2010_2025.csv               달력 변수
  - pop_total_daily_2010_2025.csv        인구 합계 (일별 ffill)
  - ip_total_daily_2010_2025.csv         전산업생산지수 (월→일, lag 2개월)

산출: data/processed/exog_daily_2010_2025.csv  (5,844행 × tm + 27개 외생변수)

설계:
  - tm 키 기준 inner join (모두 동일 5,844행 가정, assert로 검증).
  - feels_like_daily 중 `ta_hourly_max/min`는 weather_nationwide의 ta_max/min
    과 거의 동일하지만 시간자료 기반이라 정의가 미세하게 달라 함께 보존.
  - 종속변수 peak_mw는 별도 파일(`peak_mw_2010_2025.csv`)로 분리 유지.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_CSV = PROCESSED / "exog_daily_2010_2025.csv"

EXPECTED_ROWS = 5844

INPUTS = [
    "weather_nationwide_2010_2025.csv",
    "cdd_hdd_2010_2025.csv",
    "feels_like_daily_2010_2025.csv",
    "extreme_flags_2010_2025.csv",
    "calendar_2010_2025.csv",
    "pop_total_daily_2010_2025.csv",
    "ip_total_daily_2010_2025.csv",
]


def main() -> None:
    frames: list[pd.DataFrame] = []
    for name in INPUTS:
        path = PROCESSED / name
        if not path.exists():
            sys.exit(f"ERROR: {path} 없음.")
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "tm" not in df.columns:
            sys.exit(f"ERROR: {name}에 tm 컬럼 없음.")
        if len(df) != EXPECTED_ROWS:
            sys.exit(f"ERROR: {name} 행 수 {len(df)} ≠ {EXPECTED_ROWS}")
        frames.append(df)
        print(f"  로드 {name}: {len(df):,}행, {len(df.columns) - 1}개 변수")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="tm", how="inner", validate="one_to_one")

    assert len(merged) == EXPECTED_ROWS, f"merge 후 {len(merged)} ≠ {EXPECTED_ROWS}"

    missing = merged.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\nWARNING — 결측 컬럼:")
        print(missing.to_string())
    else:
        print("\n전 컬럼 결측 0.")

    n_vars = len(merged.columns) - 1
    print(f"\n통합 결과: {len(merged):,}행 × tm + {n_vars}개 변수")
    print(f"기간: {merged['tm'].iloc[0]} ~ {merged['tm'].iloc[-1]}")
    print("\n컬럼 목록:")
    for col in merged.columns:
        print(f"  - {col}")

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
