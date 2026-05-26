"""
시간별 ASOS imputed 데이터 × 연도별 인구 가중치 → 시간별 전국 평균 (TA/HM/WS)
→ 시간별 체감온도(PT) → 일 max/min 집계.

산출:
  data/processed/feels_like_daily_2010_2025.csv (5,844행)
    columns: tm, feels_like_max, feels_like_min, ta_hourly_max, ta_hourly_min

체감온도 공식 (docs/research_log/2026-05-25_2139_체감온도_공식_확정.md):
  5/1 ~ 9/30  : 여름 공식 (Stull Tw + 기상청 PT_summer)
  10/1 ~ 4/30 : 겨울 공식 (NWS Wind Chill, T≤10°C AND V≥1.3m/s 시)
  연중 단일 컬럼 feels_like = PT_summer 또는 PT_winter

집계 정의:
  feels_like_max: 그날 24시간 PT 중 최댓값
  feels_like_min: 그날 24시간 PT 중 최솟값
  ta_hourly_max/min: 동일 방식의 실측 기온 비교용 (sanity check 보조)

처리 흐름:
  1. 16개 imputed CSV 로드 → tm × sido long table
  2. 연도별 인구 가중치 merge → 시간별 가중 평균 (TA/HM/WS)
  3. 월 기준 여름/겨울 공식 분기로 시간별 PT 산출
  4. 일 단위 max/min 집계
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMPUTED_DIR = ROOT / "data" / "processed" / "asos_hourly_imputed"
WEIGHTS_CSV = ROOT / "data" / "processed" / "sido_weights_2010_2025.csv"
OUT_DAILY = ROOT / "data" / "processed" / "feels_like_daily_2010_2025.csv"

SIDO_TO_STN: dict[str, str] = {
    "11": "108", "26": "159", "27": "143", "28": "112",
    "29": "156", "30": "133", "31": "152", "41": "119",
    "43": "131", "44": "232", "46": "165", "47": "136",
    "48": "155", "50": "184", "51": "101", "52": "146",
}


def feels_like_summer(T: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """기상청 폭염 체감온도 (2020 개정판) — 여름 5~9월."""
    Tw = (
        T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.67633)
        + 0.00391838 * np.power(RH, 1.5) * np.arctan(0.023101 * RH)
        - 4.686035
    )
    PT = (
        -0.2442
        + 0.55399 * Tw
        + 0.45535 * T
        - 0.0022 * Tw * Tw
        + 0.00278 * Tw * T
        + 3.0
    )
    return PT


def feels_like_winter(T: np.ndarray, V_ms: np.ndarray) -> np.ndarray:
    """NWS/환경캐나다 풍속냉각 공식 (2001) — 겨울 10~4월.
    적용 조건: T ≤ 10°C AND V ≥ 1.3 m/s. 그 외 PT = T."""
    V_kmh = V_ms * 3.6
    # power는 V_kmh > 0 가정. V_kmh = 0이면 적용 조건 미충족이므로 안전.
    safe_kmh = np.where(V_kmh > 0, V_kmh, 1e-9)
    PT_wc = (
        13.12
        + 0.6215 * T
        - 11.37 * np.power(safe_kmh, 0.16)
        + 0.3965 * T * np.power(safe_kmh, 0.16)
    )
    apply = (T <= 10.0) & (V_ms >= 1.3)
    return np.where(apply, PT_wc, T)


def feels_like(T: np.ndarray, RH: np.ndarray, V_ms: np.ndarray, month: np.ndarray) -> np.ndarray:
    is_summer = (month >= 5) & (month <= 9)
    pt_s = feels_like_summer(T, RH)
    pt_w = feels_like_winter(T, V_ms)
    return np.where(is_summer, pt_s, pt_w)


def load_long_hourly() -> pd.DataFrame:
    """16개 imputed 시간자료 → tm × sido_code long."""
    frames = []
    for sido, stn in SIDO_TO_STN.items():
        p = IMPUTED_DIR / f"asos_hourly_stn{stn}_2010_2025_imputed.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, parse_dates=["tm"], usecols=["tm", "ta", "hm", "ws"])
        df["sido_code"] = sido
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    return long


def main() -> None:
    if not WEIGHTS_CSV.exists():
        sys.exit(f"ERROR: {WEIGHTS_CSV} 없음")

    print("[1/4] 16지점 시간자료 로드", flush=True)
    long = load_long_hourly()
    print(f"   long rows: {len(long):,}")

    # 결측 점검 — imputed 단계에서 0이어야 함
    for c in ["ta", "hm", "ws"]:
        n = long[c].isna().sum()
        if n > 0:
            print(f"   WARN: {c}에 잔여 결측 {n}건 — 가중 평균 NaN 가능")

    print("[2/4] 연도 가중치 merge + 시간별 가중 평균", flush=True)
    weights = pd.read_csv(WEIGHTS_CSV, dtype={"sido_code": str})[
        ["year", "sido_code", "weight"]
    ]
    long["year"] = long["tm"].dt.year
    df = long.merge(weights, on=["year", "sido_code"], how="left")
    if df["weight"].isna().any():
        miss = df[df["weight"].isna()][["year", "sido_code"]].drop_duplicates()
        sys.exit(f"ERROR: 가중치 매칭 실패\n{miss}")

    # 시간별 가중 평균 — groupby tm
    for c in ["ta", "hm", "ws"]:
        df[f"w_{c}"] = df[c] * df["weight"]
    hourly = df.groupby("tm")[["w_ta", "w_hm", "w_ws"]].sum()
    hourly.columns = ["ta", "hm", "ws"]
    print(f"   hourly rows: {len(hourly):,} (기대 = 5,844 × 24 = 140,256)")

    print("[3/4] 시간별 체감온도 산출", flush=True)
    hourly = hourly.reset_index()
    months = hourly["tm"].dt.month.values
    hourly["feels_like"] = feels_like(
        hourly["ta"].values,
        hourly["hm"].values,
        hourly["ws"].values,
        months,
    )

    # sanity 검증값
    print(f"   체감온도 통계 (전체):")
    print(f"     min={hourly['feels_like'].min():.2f}, max={hourly['feels_like'].max():.2f}")
    print(f"     mean={hourly['feels_like'].mean():.2f}")
    summer = hourly[hourly["tm"].dt.month.isin([6, 7, 8])]
    winter = hourly[hourly["tm"].dt.month.isin([12, 1, 2])]
    print(f"   6-8월 max  : {summer['feels_like'].max():.2f}°C  (실측 ta_max: {summer['ta'].max():.2f}°C)")
    print(f"   12-2월 min : {winter['feels_like'].min():.2f}°C  (실측 ta_min: {winter['ta'].min():.2f}°C)")

    print("[4/4] 일 max/min 집계", flush=True)
    hourly["date"] = hourly["tm"].dt.strftime("%Y-%m-%d")
    daily = hourly.groupby("date").agg(
        feels_like_max=("feels_like", "max"),
        feels_like_min=("feels_like", "min"),
        ta_hourly_max=("ta", "max"),
        ta_hourly_min=("ta", "min"),
    ).reset_index().rename(columns={"date": "tm"})

    assert len(daily) == 5844, f"기대 5,844행, 실제 {len(daily)}"

    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_DAILY} ({len(daily):,}행)")

    # 표본 검증
    print(f"\n샘플 — 폭염일 (2018-08-01):")
    print(daily[daily["tm"] == "2018-08-01"].to_string(index=False))
    print(f"샘플 — 한파일 (2018-01-26 추정):")
    print(daily[daily["tm"] == "2018-01-26"].to_string(index=False))


if __name__ == "__main__":
    main()
