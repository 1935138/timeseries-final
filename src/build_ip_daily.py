"""
전산업생산지수(원지수) 월별 데이터를 일별로 변환 + 발표 lag 적용.

입력: data/raw/kosis_ip_total_2010_2025.csv (월별, 5 분류)
산출: data/processed/ip_total_daily_2010_2025.csv (일별 5,844행)
  columns: tm, ip_total

가공 정의:
  - C1=0 (전산업)만 사용.
  - 발표 lag 2개월 강제: M월 값은 M+2월 1일부터 사용 가능.
    예) 2010-01 값 → 2010-03-01부터 ffill 시작.
    (KOSIS 산업활동동향은 익월 말 ~ 익익월 초 발표 — 보수적으로 2개월 lag)
  - 결과: 2010-01-01 ~ 2010-02-28 구간은 결측이 됨 → 첫 사용 가능 값(2010-01)으로
    backfill 처리하여 5,844행 전체 채움. (학습 시작 시점이므로 정보 누설 아님)

미래 정보 누설 방지:
  forecast 시점 t에서 사용 가능한 ip_total은 적어도 (t-60일) 이전 시점의 발표값.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "raw" / "kosis_ip_total_2010_2025.csv"
OUT_CSV = ROOT / "data" / "processed" / "ip_total_daily_2010_2025.csv"

DATE_START = "2010-01-01"
DATE_END = "2025-12-31"
LAG_MONTHS = 2


def main() -> None:
    if not IN_CSV.exists():
        sys.exit(f"ERROR: {IN_CSV} 없음. src/collect_kosis_ip.py 먼저 실행.")

    raw = pd.read_csv(IN_CSV, dtype={"prd": str, "c1": str})
    total = raw[raw["c1"] == "0"].copy().sort_values("prd")
    print(f"전산업 월별: {len(total)}개월 ({total['prd'].min()} ~ {total['prd'].max()})")

    period = pd.PeriodIndex(total["prd"], freq="M")
    total["available_from"] = (period + LAG_MONTHS).to_timestamp(how="start")
    print(f"\nlag {LAG_MONTHS}개월 적용 — 첫 사용 가능 시점: {total['available_from'].min().date()}")
    print(f"최근 가용 시작일:")
    print(total[["prd", "available_from"]].tail(5).to_string(index=False))

    idx = pd.date_range(DATE_START, DATE_END, freq="D")
    daily = pd.DataFrame({"tm": idx})

    series = (
        total.set_index("available_from")["value"]
        .sort_index()
        .reindex(idx, method="ffill")
    )
    daily["ip_total"] = series.values

    n_missing = daily["ip_total"].isna().sum()
    print(f"\nffill 후 결측: {n_missing}일 (lag로 인한 시계열 초반 결측)")
    if n_missing > 0:
        first_val = total["value"].iloc[0]
        print(f"  → 첫 가용 값({first_val})으로 backfill 처리")
        daily["ip_total"] = daily["ip_total"].fillna(first_val)

    assert daily["ip_total"].isna().sum() == 0
    assert len(daily) == 5844, f"기대 5,844행, 실제 {len(daily)}"

    daily["tm"] = daily["tm"].dt.strftime("%Y-%m-%d")

    print(f"\n샘플 (월 경계 ffill 확인):")
    print(daily.iloc[[0, 59, 60, 61, 90, 91, -1]].to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
