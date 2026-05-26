"""
전국 주민등록인구 합계 산출 (장기 추세 외생변수).

입력: data/raw/kosis_population_sido_2010_2025.csv (KOSIS DT_1YL20651E)
산출:
  - data/processed/pop_total_2010_2025.csv  (연별 16~17행)
  - data/processed/pop_total_daily_2010_2025.csv  (일별 5,844행)

설계:
  - 시·도(세종 포함 17개, 2010-2011은 16개) 단순 합산 → 전국.
  - 세종 통합 처리 불필요(합계는 통합 여부 무관).
  - 일별 변환: 해당 연도 값을 그 해 모든 날짜에 ffill (12-31 기준).
  - 인구는 발표 lag가 짧고 사후 개정도 거의 없어 lag 강제 없이 사용.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "raw" / "kosis_population_sido_2010_2025.csv"
OUT_YEARLY = ROOT / "data" / "processed" / "pop_total_2010_2025.csv"
OUT_DAILY = ROOT / "data" / "processed" / "pop_total_daily_2010_2025.csv"

DATE_START = "2010-01-01"
DATE_END = "2025-12-31"


def main() -> None:
    if not IN_CSV.exists():
        sys.exit(f"ERROR: {IN_CSV} 없음. src/collect_kosis_population.py 먼저 실행.")

    df = pd.read_csv(IN_CSV)
    yearly = (
        df.groupby("year", as_index=False)["population"]
        .sum()
        .rename(columns={"population": "pop_total"})
        .sort_values("year")
    )

    print(f"연별 산출: {len(yearly)}행")
    print(yearly.to_string(index=False))
    print(f"\n2010 → 2025 증감: {yearly['pop_total'].iloc[-1] - yearly['pop_total'].iloc[0]:+,}명")
    print(f"정점 연도: {yearly.loc[yearly['pop_total'].idxmax(), 'year']}")

    OUT_YEARLY.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(OUT_YEARLY, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_YEARLY}")

    idx = pd.date_range(DATE_START, DATE_END, freq="D")
    daily = pd.DataFrame({"tm": idx.strftime("%Y-%m-%d"), "year": idx.year})
    daily = daily.merge(yearly, on="year", how="left").drop(columns=["year"])

    assert daily["pop_total"].isna().sum() == 0, "일별 ffill 후 결측 발생"
    assert len(daily) == 5844, f"기대 5,844행, 실제 {len(daily)}"

    daily.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_DAILY} ({len(daily):,}행)")


if __name__ == "__main__":
    main()
