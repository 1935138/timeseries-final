"""
기상청 API허브로 16개 시·도 대표 + 6개 백업 관측소의 ASOS 시간자료(2010-2025)
를 수집. 체감온도(시간별 PT) 산출용 입력.

산출물:
  data/raw/asos_hourly/asos_hourly_stn<지점>_2010_2025.csv (관측소당 1개, 22개)
  컬럼: tm(YYYY-MM-DD HH:00), stn, ta, hm, ws

엔드포인트: /url/kma_sfctm3.php (지상관측 시간자료, 구간조회)
호출 단위: 월 단위 (1회당 약 720~744행)
총 호출: 22지점 × 192개월 = 4,224회, sleep 1s 포함 약 1.2시간 예상

결측 sentinel (일자료와 동일):
  - TA: -50 이하만 sentinel (실측 -9.0°C 가능)
  - HM, WS: 음수 모두 sentinel

지점 매핑 / 결정 근거: docs/research_log/2026-05-25_대표관측소_인구가중평균.md,
  docs/research_log/2026-05-25_2139_체감온도_공식_확정.md
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "asos_hourly"
API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"

# 16개 시·도 대표 + 6개 백업
STATIONS = {
    "108": "서울", "159": "부산", "143": "대구", "112": "인천",
    "156": "광주", "133": "대전", "152": "울산",
    "119": "수원_경기", "101": "춘천_강원", "131": "청주_충북",
    "232": "천안_충남세종", "146": "전주_전북", "165": "목포_전남",
    "136": "안동_경북", "155": "창원_경남", "184": "제주",
}
BACKUP_STATIONS = {
    "114": "원주", "105": "강릉", "137": "상주",
    "138": "포항", "168": "여수", "189": "서귀포",
}

# kma_sfctm3.php 46 컬럼 — 핵심 5개만 보존
SFCTM3_COLUMNS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM",
    "PA", "PS", "PT", "PR",
    "TA", "TD", "HM", "PV",
    "RN", "RN_DAY", "RN_JUN", "RN_INT",
    "SD_HR3", "SD_DAY", "SD_TOT",
    "WC", "WP", "WW",
    "CA_TOT", "CA_MID", "CH_MIN",
    "CT", "CT_TOP", "CT_MID", "CT_LOW",
    "VS", "SS", "SI",
    "ST_GD", "TS", "TE_005", "TE_01", "TE_02", "TE_03",
    "ST_SEA", "WH", "BF", "IR", "IX",
]
KEEP = ["TM", "STN", "TA", "HM", "WS"]

START_YEAR, END_YEAR = 2010, 2025
SLEEP_SEC = 1.0
MAX_RETRIES = 3


def month_ranges() -> list[tuple[str, str]]:
    """월 단위 (tm1, tm2) 구간 생성. tm2는 해당 월 마지막날 23시."""
    ranges = []
    for y in range(START_YEAR, END_YEAR + 1):
        for m in range(1, 13):
            tm1 = f"{y:04d}{m:02d}010000"
            # 다음 달 1일 0시 직전의 마지막 시각
            if m == 12:
                next_first = pd.Timestamp(year=y + 1, month=1, day=1)
            else:
                next_first = pd.Timestamp(year=y, month=m + 1, day=1)
            last_hour = next_first - pd.Timedelta(hours=1)
            tm2 = last_hour.strftime("%Y%m%d%H00")
            ranges.append((tm1, tm2))
    return ranges


def call_month(api_key: str, stn: str, tm1: str, tm2: str) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                API_URL,
                params={
                    "tm1": tm1, "tm2": tm2, "stn": stn,
                    "help": "0", "authKey": api_key,
                },
                timeout=180,
            )
            if resp.status_code == 200:
                return resp.text
            last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
        except requests.RequestException as e:
            last_err = str(e)
        wait = 2 ** attempt
        print(f"      재시도 {attempt+1}/{MAX_RETRIES} ({last_err[:60]}) — {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"호출 실패: {last_err}")


def parse_text(text: str) -> pd.DataFrame:
    lines = [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not lines:
        return pd.DataFrame(columns=[c.lower() for c in KEEP])

    rows = [ln.split() for ln in lines]
    width = min(len(SFCTM3_COLUMNS), max(len(r) for r in rows))
    rows = [r[:width] for r in rows]
    df = pd.DataFrame(rows, columns=SFCTM3_COLUMNS[:width])

    # sentinel 처리
    ta = pd.to_numeric(df["TA"], errors="coerce").mask(lambda s: s <= -50)
    hm = pd.to_numeric(df["HM"], errors="coerce").mask(lambda s: s < 0)
    ws = pd.to_numeric(df["WS"], errors="coerce").mask(lambda s: s < 0)

    tm = pd.to_datetime(df["TM"], format="%Y%m%d%H%M", errors="coerce")
    out = pd.DataFrame({
        "tm": tm.dt.strftime("%Y-%m-%d %H:00"),
        "stn": df["STN"].astype(str),
        "ta": ta.values,
        "hm": hm.values,
        "ws": ws.values,
    })
    return out


def collect_station(api_key: str, stn: str, label: str) -> None:
    out_csv = OUT_DIR / f"asos_hourly_stn{stn}_2010_2025.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        print(f"[{stn} {label}] 이미 존재 ({len(existing):,}행) — 건너뜀")
        return

    ranges = month_ranges()
    print(f"[{stn} {label}] 시작 — {len(ranges)}개월 수집", flush=True)
    parts = []
    t0 = time.time()
    for i, (tm1, tm2) in enumerate(ranges, 1):
        text = call_month(api_key, stn, tm1, tm2)
        df = parse_text(text)
        parts.append(df)
        if i % 24 == 0 or i == len(ranges):
            elapsed = time.time() - t0
            rate = i / elapsed * 60
            print(
                f"   [{stn}] {i:3d}/{len(ranges)}  누적 {sum(len(p) for p in parts):>7,}행  "
                f"({elapsed/60:.1f}분, {rate:.1f}호출/분)",
                flush=True,
            )
        time.sleep(SLEEP_SEC)

    df_all = pd.concat(parts, ignore_index=True).sort_values("tm").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 요약
    n_total = len(df_all)
    n_missing_ta = df_all["ta"].isna().sum()
    n_missing_hm = df_all["hm"].isna().sum()
    n_missing_ws = df_all["ws"].isna().sum()
    print(
        f"[{stn} {label}] 완료 {n_total:,}행, "
        f"결측 TA {n_missing_ta} / HM {n_missing_hm} / WS {n_missing_ws} → {out_csv.name}",
        flush=True,
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("KMA_API_KEY")
    if not api_key:
        sys.exit("ERROR: KMA_API_KEY가 .env에 없습니다.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stations = {**STATIONS, **BACKUP_STATIONS}
    print(f"=== ASOS 시간자료 수집 — {len(all_stations)}지점 × {(END_YEAR - START_YEAR + 1)*12}개월 ===")
    print(f"시작: {datetime.now().isoformat(timespec='seconds')}")

    overall_t0 = time.time()
    for stn, label in all_stations.items():
        collect_station(api_key, stn, label)
    print(
        f"\n=== 전체 완료 ({(time.time()-overall_t0)/60:.1f}분, "
        f"{datetime.now().isoformat(timespec='seconds')}) ==="
    )


if __name__ == "__main__":
    main()
