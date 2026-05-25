"""
기상청 API허브로 16개 시·도 대표 관측소의 ASOS 일자료(2010-2025)를 수집.

산출물: data/raw/asos/asos_stn<지점번호>_2010_2025.csv (관측소당 1개 파일, 16개)
  columns: tm(YYYY-MM-DD), stn, ta_avg, ta_max, ta_min, hm_avg, ws_avg, rn_day, ss_day, si_day

엔드포인트: /url/kma_sfcdd3.php  (지상관측 > 종관기상관측(ASOS) > 일자료)

관측소 매핑 (docs/research_log/2026-05-25_대표관측소_인구가중평균.md):
  서울108, 부산159, 대구143, 인천112, 광주156, 대전133, 울산152,
  수원119(경기), 춘천101(강원), 청주131(충북), 천안232(충남+세종),
  전주146(전북), 목포165(전남), 안동136(경북), 창원155(경남), 제주184
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "asos"

API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"

# 16개 시·도 대표 관측소
STATIONS = {
    "108": "서울",
    "159": "부산",
    "143": "대구",
    "112": "인천",
    "156": "광주",
    "133": "대전",
    "152": "울산",
    "119": "수원_경기",
    "101": "춘천_강원",
    "131": "청주_충북",
    "232": "천안_충남세종",
    "146": "전주_전북",
    "165": "목포_전남",
    "136": "안동_경북",
    "155": "창원_경남",
    "184": "제주",
}

# 결측 대체용 백업 관측소 (방법론: docs/research_log/2026-05-25_기상결측_대체보간_방법론.md)
BACKUP_STATIONS = {
    "114": "원주",
    "105": "강릉",
    "137": "상주",
    "138": "포항",
    "168": "여수",
    "189": "서귀포",
}

TM1 = "20100101"
TM2 = "20251231"

# kma_sfcdd3.php의 56개 표준 컬럼 (help=1 응답 확인 기준)
ASOS_COLUMNS = [
    "TM", "STN",
    "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM",
    "WD_INS", "WS_INS", "WS_INS_TM",
    "TA_AVG", "TA_MAX", "TA_MAX_TM", "TA_MIN", "TA_MIN_TM",
    "TD_AVG", "TS_AVG", "TG_MIN",
    "HM_AVG", "HM_MIN", "HM_MIN_TM",
    "PV_AVG", "EV_S", "EV_L", "FG_DUR",
    "PA_AVG", "PS_AVG", "PS_MAX", "PS_MAX_TM", "PS_MIN", "PS_MIN_TM",
    "CA_TOT",
    "SS_DAY", "SS_DUR", "SS_CMB",
    "SI_DAY", "SI_60M_MAX", "SI_60M_MAX_TM",
    "RN_DAY", "RN_D99", "RN_DUR",
    "RN_60M_MAX", "RN_60M_MAX_TM", "RN_10M_MAX", "RN_10M_MAX_TM",
    "RN_POW_MAX", "RN_POW_MAX_TM",
    "SD_NEW", "SD_NEW_TM", "SD_MAX", "SD_MAX_TM",
    "TE_05", "TE_10", "TE_15", "TE_30", "TE_50",
]

# 우리 분석에서 보존할 컬럼만 추출
KEEP_COLUMNS = [
    "TM", "STN",
    "TA_AVG", "TA_MAX", "TA_MIN",   # 기온 (핵심)
    "HM_AVG",                        # 상대습도
    "WS_AVG",                        # 풍속
    "RN_DAY",                        # 강수량
    "SS_DAY",                        # 일조시간
    "SI_DAY",                        # 일사량
]


def call_asos(api_key: str, stn: str, tm1: str, tm2: str) -> str:
    params = {
        "tm1": tm1,
        "tm2": tm2,
        "stn": stn,
        "help": "0",
        "authKey": api_key,
    }
    resp = requests.get(API_URL, params=params, timeout=180)
    resp.raise_for_status()
    return resp.text


def parse_asos_text(text: str) -> pd.DataFrame:
    """공백 구분 데이터 라인만 추출 → 56개 컬럼 적용 → 관심 컬럼만 반환."""
    data_lines = [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not data_lines:
        return pd.DataFrame()

    rows = [ln.split() for ln in data_lines]
    # 헤더 길이와 데이터 폭이 다를 수 있어 안전 처리
    width = min(len(ASOS_COLUMNS), max(len(r) for r in rows))
    rows = [r[:width] for r in rows]
    cols = ASOS_COLUMNS[:width]
    df = pd.DataFrame(rows, columns=cols)

    # ASOS 결측 sentinel (API FAQ 공식): -9.0, -99.9, -99.0, -999 등.
    # 컬럼별 분리 처리:
    #   - 온도 컬럼은 -9.0이 실측값으로 가능 (한겨울 한파일).
    #     → -99.x / -999.x 만 sentinel (수치적으로 < -50 인 경우).
    #   - 비음수 컬럼(습도/풍속/강수/일조/일사)은 음수 자체가 불가능.
    #     → -9.0 포함 모든 sentinel을 NaN.
    temp_cols = {"TA_AVG", "TA_MAX", "TA_MIN"}
    nonneg_cols = {"HM_AVG", "WS_AVG", "RN_DAY", "SS_DAY", "SI_DAY"}

    for c in [c for c in KEEP_COLUMNS if c not in ("TM", "STN")]:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if c in temp_cols:
            v = v.mask(v <= -50)              # -99.x, -999 등만 결측
        elif c in nonneg_cols:
            v = v.mask(v < 0)                 # 음수 모두 결측
        df[c] = v

    df["TM"] = pd.to_datetime(df["TM"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df["STN"] = df["STN"].astype(str)

    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("KMA_API_KEY")
    if not api_key:
        sys.exit("ERROR: KMA_API_KEY가 .env에 없습니다.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 대표 + 백업 모두 동일 파이프라인으로 수집
    all_stations = {**STATIONS, **BACKUP_STATIONS}

    summary = []
    for stn, label in all_stations.items():
        out_csv = OUT_DIR / f"asos_stn{stn}_2010_2025.csv"
        if out_csv.exists():
            existing = pd.read_csv(out_csv)
            print(f"[{stn} {label}] 이미 존재, 건너뜀 ({len(existing):,}행)")
            summary.append((stn, label, len(existing), "skipped"))
            continue

        print(f"[{stn} {label}] 호출 중 {TM1}~{TM2} ...", flush=True)
        try:
            text = call_asos(api_key, stn, TM1, TM2)
        except Exception as e:
            print(f"   ERROR: {e}")
            summary.append((stn, label, 0, f"error: {e}"))
            continue

        df = parse_asos_text(text)
        if df.empty:
            print(f"   ! 빈 응답. 처음 200자: {text[:200]}")
            summary.append((stn, label, 0, "empty"))
            continue

        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        n = len(df)
        n_missing_tavg = df["ta_avg"].isna().sum()
        print(f"   저장: {out_csv.name} ({n:,}행, ta_avg 결측 {n_missing_tavg})")
        summary.append((stn, label, n, f"ok (na_tavg={n_missing_tavg})"))
        time.sleep(0.5)

    print("\n=== 수집 요약 ===")
    for stn, label, n, status in summary:
        print(f"  {stn:>4s} {label:<14s} {n:>6,}행  {status}")


if __name__ == "__main__":
    main()
