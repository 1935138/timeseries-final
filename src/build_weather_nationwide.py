"""
16개 시·도 대표 관측소의 imputed ASOS 일자료 × 연도별 인구 가중치를 결합하여
전국 가중 평균 일자료를 산출한다.

입력:
  - data/processed/asos_imputed/asos_stn<지점>_2010_2025_imputed.csv (16개)
  - data/processed/sido_weights_2010_2025.csv (256행)

산출:
  - data/processed/weather_nationwide_2010_2025.csv (5,844행, wide)
    컬럼: tm, ta_avg, ta_max, ta_min, hm_avg, ws_avg, rn_day, ss_day,
          ss_day_n_stations (ss_day 가중에 실제 사용된 관측소 수)

방법 (docs/research_log/2026-05-25_전국가중평균_산출.md):
  - 기온/습도/풍속: imputed에서 결측 0 → 16개 풀 가중 평균
  - 강수(rn_day): 결측을 0mm로 간주 → 16개 풀 가중 평균
  - 일조(ss_day): 결측 관측소 제외 후 가중치 재정규화 (그 날 사용된 관측소만)
  - 일사(si_day): Phase 1 제외 (천안·울산·창원 대량 결측)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMPUTED_DIR = ROOT / "data" / "processed" / "asos_imputed"
WEIGHTS_CSV = ROOT / "data" / "processed" / "sido_weights_2010_2025.csv"
OUT_CSV = ROOT / "data" / "processed" / "weather_nationwide_2010_2025.csv"

# 시·도 코드 ↔ ASOS 대표 관측소 (결정: 2026-05-25_대표관측소_인구가중평균.md)
SIDO_TO_STN: dict[str, str] = {
    "11": "108",  # 서울
    "26": "159",  # 부산
    "27": "143",  # 대구
    "28": "112",  # 인천
    "29": "156",  # 광주
    "30": "133",  # 대전
    "31": "152",  # 울산
    "41": "119",  # 경기 (수원)
    "43": "131",  # 충북 (청주)
    "44": "232",  # 충남+세종 (천안)
    "46": "165",  # 전남 (목포)
    "47": "136",  # 경북 (안동)
    "48": "155",  # 경남 (창원)
    "50": "184",  # 제주
    "51": "101",  # 강원 (춘천)
    "52": "146",  # 전북 (전주)
}

# 16개 시·도 풀 가중 평균 (결측 없음 가정)
FULL_COLS = ["ta_avg", "ta_max", "ta_min", "hm_avg", "ws_avg"]
# 강수: 결측을 0으로 간주 후 풀 가중 평균
RAIN_COLS = ["rn_day"]
# 부분 가중 평균 (결측 관측소 제외 + 재정규화)
PARTIAL_COLS = ["ss_day"]


def load_long_table() -> pd.DataFrame:
    """16개 imputed CSV를 long 형식(tm × sido × variable)으로 통합."""
    frames = []
    for sido, stn in SIDO_TO_STN.items():
        p = IMPUTED_DIR / f"asos_stn{stn}_2010_2025_imputed.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, parse_dates=["tm"])
        df["sido_code"] = sido
        keep = ["tm", "sido_code"] + FULL_COLS + RAIN_COLS + PARTIAL_COLS
        frames.append(df[keep])
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["tm", "sido_code"]).reset_index(drop=True)
    return long


def main() -> None:
    if not WEIGHTS_CSV.exists():
        sys.exit(f"ERROR: {WEIGHTS_CSV} 없음")

    long = load_long_table()
    weights = pd.read_csv(WEIGHTS_CSV, dtype={"sido_code": str})
    weights = weights[["year", "sido_code", "weight"]]

    long["year"] = long["tm"].dt.year
    df = long.merge(weights, on=["year", "sido_code"], how="left")
    if df["weight"].isna().any():
        miss = df[df["weight"].isna()][["year","sido_code"]].drop_duplicates()
        sys.exit(f"ERROR: 가중치 매칭 실패\n{miss}")

    # 강수량: 결측 → 0mm
    df["rn_day"] = df["rn_day"].fillna(0.0)

    out_frames = []

    # 1. FULL_COLS + RAIN_COLS: 단순 가중 평균 (결측 없음)
    for col in FULL_COLS + RAIN_COLS:
        if df[col].isna().any():
            n = df[col].isna().sum()
            print(f"WARN: {col}에 잔여 결측 {n}건 — 가중 평균 결과가 NaN 될 수 있음")
        agg = (
            df.assign(wval=df[col] * df["weight"])
              .groupby("tm")["wval"].sum()
              .rename(col)
        )
        out_frames.append(agg)

    # 2. PARTIAL_COLS (ss_day): 결측 관측소 제외 + 가중치 재정규화
    for col in PARTIAL_COLS:
        valid = df.dropna(subset=[col]).copy()
        # 그 날 사용 가능한 관측소들의 가중치 합으로 나눠 재정규화
        valid["w_norm"] = (
            valid["weight"] / valid.groupby("tm")["weight"].transform("sum")
        )
        agg = (
            valid.assign(wval=valid[col] * valid["w_norm"])
                 .groupby("tm")["wval"].sum()
                 .rename(col)
        )
        n_stations = (
            valid.groupby("tm")["sido_code"].nunique()
                 .rename(f"{col}_n_stations")
        )
        out_frames.append(agg)
        out_frames.append(n_stations)

    out = pd.concat(out_frames, axis=1).reset_index()
    out["tm"] = out["tm"].dt.strftime("%Y-%m-%d")
    # 컬럼 순서 정렬
    col_order = ["tm"] + FULL_COLS + RAIN_COLS + PARTIAL_COLS + [f"{c}_n_stations" for c in PARTIAL_COLS]
    out = out[col_order]

    # 검증
    print(f"산출: {len(out):,}행 (기대 5,844)")
    print(f"기간: {out['tm'].min()} ~ {out['tm'].max()}")
    print(f"컬럼: {list(out.columns)}")
    print()
    print("결측 현황:")
    print(out.isna().sum().to_string())
    print()
    print(f"ss_day 사용 관측소 수 분포:")
    print(out["ss_day_n_stations"].value_counts().sort_index().to_string())
    print()
    print("기온 통계:")
    print(out[["ta_avg","ta_max","ta_min"]].describe().round(2).to_string())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
