"""
냉방도일(CDD) · 난방도일(HDD) 및 이동평균 변수 생성.

입력: data/processed/weather_nationwide_2010_2025.csv  (인구가중 전국평균 ta_avg)
산출: data/processed/cdd_hdd_2010_2025.csv  (5,844행, 6 변수 + tm)

산출 변수:
  cdd_18      = max(ta_avg - 18, 0)
  hdd_18      = max(18 - ta_avg, 0)
  cdd_18_ma3  = cdd_18 의 3일 이동평균 (window=[t-2, t])
  hdd_18_ma3  = 동일
  cdd_18_ma7  = 7일 이동평균
  hdd_18_ma7  = 동일

근거: 한국 전력 도메인 표준 base=18°C (한전 전력연구원·에너지경제연구원).
      ma3·ma7는 건물 열관성·누적 부하 효과 반영
      (Bessec 2008, Apadula 2012, Mirasgedis 2006, KEEI 2021).
      상세는 docs/research_log/2026-05-25_2124_냉방도일_난방도일_정의.md

미래 정보 누설:
  rolling 은 [t-w+1, t] 윈도라 t 시점 가용 정보만 사용 → 안전.
  시계열 초반 ma3 2일 / ma7 6일 NaN 은 backfill (학습 시작 전 구간이므로
  정보 누설 아님).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "processed" / "weather_nationwide_2010_2025.csv"
OUT_CSV = ROOT / "data" / "processed" / "cdd_hdd_2010_2025.csv"

BASE = 18.0


def main() -> None:
    if not IN_CSV.exists():
        sys.exit(f"ERROR: {IN_CSV} 없음. src/build_weather_nationwide.py 먼저 실행.")

    src = pd.read_csv(IN_CSV)
    assert src["ta_avg"].isna().sum() == 0, "ta_avg 결측 존재 — 입력 정합 확인 필요"

    df = pd.DataFrame({"tm": src["tm"]})
    df["cdd_18"] = (src["ta_avg"] - BASE).clip(lower=0)
    df["hdd_18"] = (BASE - src["ta_avg"]).clip(lower=0)

    for w in (3, 7):
        df[f"cdd_18_ma{w}"] = df["cdd_18"].rolling(w, min_periods=w).mean()
        df[f"hdd_18_ma{w}"] = df["hdd_18"].rolling(w, min_periods=w).mean()

    # 시계열 초반 결측 backfill (ma3: 2일, ma7: 6일)
    n_missing_before = df.isna().sum()
    df = df.bfill()

    assert df.isna().sum().sum() == 0
    assert len(df) == 5844, f"기대 5,844행, 실제 {len(df)}"

    print(f"산출: {len(df):,}행")
    print(f"\nbackfill 전 결측:")
    print(n_missing_before[n_missing_before > 0].to_string())

    print(f"\n변수별 요약통계:")
    print(df.drop(columns=["tm"]).describe().round(3).to_string())

    # 계절 분포 sanity check
    df["month"] = pd.to_datetime(df["tm"]).dt.month
    monthly = df.groupby("month")[["cdd_18", "hdd_18"]].mean().round(2)
    print(f"\n월별 평균 (sanity — CDD는 6-9월, HDD는 12-2월에 집중):")
    print(monthly.to_string())

    # 연도 합 trend
    df["year"] = df["tm"].str[:4].astype(int)
    yearly = df.groupby("year")[["cdd_18", "hdd_18"]].sum().round(1)
    print(f"\n연도별 합계 (기후변화 trend 확인):")
    print(yearly.to_string())

    df = df.drop(columns=["month", "year"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
