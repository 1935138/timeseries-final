"""
2010-01-01 ~ 2025-12-31 일별 달력 변수 생성.

입력: data/raw/holidays_2010_2025.csv (특일 API 수집 결과)
산출: data/processed/calendar_2010_2025.csv (5,844행)
  columns: tm, dow, month, is_weekend, is_holiday

변수 정의:
  - dow: 요일 (0=월 … 6=일)
  - month: 1~12
  - is_weekend: 토(5)·일(6) → 1, 그 외 0
  - is_holiday: 특일 API의 is_holiday=Y인 날 → 1 (대체·임시공휴일 포함)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HOLIDAYS_CSV = ROOT / "data" / "raw" / "holidays_2010_2025.csv"
OUT_CSV = ROOT / "data" / "processed" / "calendar_2010_2025.csv"

DATE_START = "2010-01-01"
DATE_END = "2025-12-31"


def main() -> None:
    if not HOLIDAYS_CSV.exists():
        sys.exit(f"ERROR: {HOLIDAYS_CSV} 없음. src/collect_holidays.py 먼저 실행.")

    hol = pd.read_csv(HOLIDAYS_CSV)
    holiday_dates = set(hol.loc[hol["is_holiday"] == "Y", "date"])
    print(f"공휴일 입력: 전체 {len(hol):,}건 중 is_holiday=Y {len(holiday_dates):,}건")

    idx = pd.date_range(DATE_START, DATE_END, freq="D")
    df = pd.DataFrame({"tm": idx.strftime("%Y-%m-%d")})
    df["dow"] = idx.dayofweek          # 0=월 … 6=일
    df["month"] = idx.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")
    df["is_holiday"] = df["tm"].isin(holiday_dates).astype("int8")

    # 검증
    print(f"\n산출: {len(df):,}행 (기대 5,844)")
    print(f"기간: {df['tm'].min()} ~ {df['tm'].max()}")
    print(f"\n요일 분포:")
    print(df["dow"].value_counts().sort_index().to_string())
    print(f"\nis_weekend: {df['is_weekend'].sum()}일 ({df['is_weekend'].mean()*100:.1f}%)")
    print(f"is_holiday: {df['is_holiday'].sum()}일")
    print(f"\n연도별 is_holiday 수:")
    print(df.groupby(df["tm"].str[:4])["is_holiday"].sum().to_string())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
