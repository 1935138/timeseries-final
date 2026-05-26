"""
기상청 공식 특보 발효기준에 정합한 폭염·한파 임계 더미 산출.

입력: data/processed/feels_like_daily_2010_2025.csv (5,844행)
  사용 컬럼: feels_like_max, ta_hourly_min

산출: data/processed/extreme_flags_2010_2025.csv (5,844행, tm + 4 더미)
  - heat_feels_th_2day_adv : 폭염주의보 ① 정합  (feels_like_max ≥ 33°C 2일 연속)
  - heat_feels_th_2day_wrn : 폭염경보 ① 정합   (feels_like_max ≥ 35°C 2일 연속)
  - cold_th_2day_adv       : 한파주의보 ② 정합 (ta_min ≤ -12°C 2일 연속, 10~4월)
  - cold_th_2day_wrn       : 한파경보 ② 정합   (ta_min ≤ -15°C 2일 연속, 10~4월)

근거: docs/research_log/2026-05-25_2139_체감온도_공식_확정.md, 기상청
  특보 발표기준(https://www.kma.go.kr/kma/biz/forecast03.jsp).

제외:
  - 한파 ① 급강하 조건: 평년편차 기준 필요 → 본 모델 제외.
  - 폭염·한파 ②/③ "정성적 광범위 피해" 조항: 수식 불가.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "processed" / "feels_like_daily_2010_2025.csv"
OUT_CSV = ROOT / "data" / "processed" / "extreme_flags_2010_2025.csv"

COLD_MONTHS = {10, 11, 12, 1, 2, 3, 4}


def two_day_streak(series: pd.Series) -> pd.Series:
    """오늘 + 어제 모두 True 일 때만 True. NaN 안전."""
    today = series.fillna(False).astype(bool)
    yesterday = today.shift(1, fill_value=False)
    return (today & yesterday).astype("int8")


def main() -> None:
    if not IN_CSV.exists():
        sys.exit(f"ERROR: {IN_CSV} 없음. src/build_feels_like.py 먼저 실행.")

    src = pd.read_csv(IN_CSV)
    assert len(src) == 5844, f"기대 5,844행, 실제 {len(src)}"

    df = pd.DataFrame({"tm": src["tm"]})
    month = pd.to_datetime(src["tm"]).dt.month

    # 폭염 — 연중 (공식 기준에 계절 한정 없음)
    heat_adv_today = src["feels_like_max"] >= 33.0
    heat_wrn_today = src["feels_like_max"] >= 35.0
    df["heat_feels_th_2day_adv"] = two_day_streak(heat_adv_today)
    df["heat_feels_th_2day_wrn"] = two_day_streak(heat_wrn_today)

    # 한파 — 10~4월 한정
    is_cold_season = month.isin(COLD_MONTHS)
    cold_adv_today = (src["ta_hourly_min"] <= -12.0) & is_cold_season
    cold_wrn_today = (src["ta_hourly_min"] <= -15.0) & is_cold_season
    df["cold_th_2day_adv"] = two_day_streak(cold_adv_today)
    df["cold_th_2day_wrn"] = two_day_streak(cold_wrn_today)

    # 검증
    print(f"산출: {len(df):,}행")
    print(f"\n변수별 발효일수:")
    for c in ["heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
              "cold_th_2day_adv", "cold_th_2day_wrn"]:
        print(f"  {c}: {df[c].sum():,}일 ({df[c].mean()*100:.2f}%)")

    df["year"] = df["tm"].str[:4].astype(int)
    yearly = df.groupby("year")[
        ["heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
         "cold_th_2day_adv", "cold_th_2day_wrn"]
    ].sum()
    print(f"\n연도별 발효일수:")
    print(yearly.to_string())

    # 샘플 — 알려진 폭염년 2018, 한파년 2018-01
    print(f"\n2018-07-15 ~ 2018-08-05 (폭염 정점):")
    sample = df[(df["tm"] >= "2018-07-15") & (df["tm"] <= "2018-08-05")][
        ["tm", "heat_feels_th_2day_adv", "heat_feels_th_2day_wrn"]
    ]
    print(sample.to_string(index=False))

    print(f"\n2018-01-22 ~ 2018-02-05 (한파):")
    sample2 = df[(df["tm"] >= "2018-01-22") & (df["tm"] <= "2018-02-05")][
        ["tm", "cold_th_2day_adv", "cold_th_2day_wrn"]
    ]
    print(sample2.to_string(index=False))

    df = df.drop(columns=["year"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
