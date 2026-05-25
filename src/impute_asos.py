"""
16개 시·도 대표 관측소의 ASOS 결측치를 백업 관측소(bias 보정)로 대체하고
잔여 결측은 시점 선형 보간으로 채운다.

방법론: docs/research_log/2026-05-25_기상결측_대체보간_방법론.md
입력: data/raw/asos/asos_stn<지점>_2010_2025.csv (대표 16개 + 백업 6개)
산출: data/processed/asos_imputed/asos_stn<지점>_2010_2025_imputed.csv (16개)
  컬럼: tm, stn, ta_avg, ta_max, ta_min, hm_avg, ws_avg, rn_day, ss_day, si_day,
        ta_avg_src, ta_max_src, ta_min_src, hm_avg_src, ws_avg_src

처리 우선순위:
  1. raw (원본)
  2. 1차 백업 (bias 보정)
  3. 2차 백업 (bias 보정)
  4. 시점 선형 보간 (limit=3일)

미래정보 누설 방지: bias(평균 차이)는 학습 구간 2010-2024 데이터로만 계산.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "asos"
OUT_DIR = ROOT / "data" / "processed" / "asos_imputed"

# 백업 매핑 (대표 → [1차 백업, 2차 백업])
BACKUP_MAP: dict[str, list[str]] = {
    "108": ["119", "112"],      # 서울 ← 수원, 인천
    "119": ["108", "112"],      # 수원 ← 서울, 인천
    "112": ["108", "119"],      # 인천 ← 서울, 수원
    "101": ["114", "105"],      # 춘천 ← 원주, 강릉
    "131": ["232", "133"],      # 청주 ← 천안, 대전
    "232": ["133", "131"],      # 천안 ← 대전, 청주
    "133": ["232", "131"],      # 대전 ← 천안, 청주
    "146": ["156", "165"],      # 전주 ← 광주, 목포
    "156": ["165", "146"],      # 광주 ← 목포, 전주
    "165": ["156", "168"],      # 목포 ← 광주, 여수
    "136": ["137", "138"],      # 안동 ← 상주, 포항
    "155": ["159", "152"],      # 창원 ← 부산, 울산
    "159": ["152", "155"],      # 부산 ← 울산, 창원
    "152": ["159", "138"],      # 울산 ← 부산, 포항
    "143": ["137", "136"],      # 대구 ← 상주, 안동
    "184": ["189"],             # 제주 ← 서귀포
}

# bias 보정 대상 컬럼 (강수·일조·일사는 대체 안 함 — 방법론 §4)
IMPUTE_COLS = ["ta_avg", "ta_max", "ta_min", "hm_avg", "ws_avg"]

# 학습 구간 (bias 계산용) — 미래정보 누설 방지
TRAIN_END = "2024-12-31"

# 시점 선형 보간 한계 (연속 결측 일수)
INTERP_LIMIT = 3


def load_station(stn: str) -> pd.DataFrame:
    """관측소 raw CSV를 읽어 인덱스를 datetime으로."""
    p = RAW_DIR / f"asos_stn{stn}_2010_2025.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=["tm"])
    df = df.set_index("tm").sort_index()
    return df


def compute_bias(main: pd.Series, backup: pd.Series) -> float:
    """학습 구간에서 양쪽 모두 값이 있는 날의 평균 차이 (main - backup)."""
    m = main.loc[:TRAIN_END]
    b = backup.loc[:TRAIN_END]
    common = m.notna() & b.notna()
    if common.sum() == 0:
        return 0.0
    return float((m[common] - b[common]).mean())


def impute_one_station(main_stn: str, backups: list[str]) -> tuple[pd.DataFrame, dict]:
    """대표 1개에 대해 백업 → 보간 순으로 대체. 출처 컬럼 동시 생성."""
    main_df = load_station(main_stn)
    backup_dfs = {b: load_station(b) for b in backups}

    out = main_df.copy()
    stats = {"stn": main_stn, "columns": {}}

    for col in IMPUTE_COLS:
        src = pd.Series("raw", index=out.index)
        src[out[col].isna()] = pd.NA

        # 백업 우선순위로 시도
        for b_stn in backups:
            b_df = backup_dfs[b_stn]
            if col not in b_df.columns:
                continue
            bias = compute_bias(main_df[col], b_df[col])
            # 결측이면서 백업이 값을 가진 경우 채우기
            need = out[col].isna() & b_df[col].notna()
            if need.any():
                out.loc[need, col] = b_df.loc[need, col] + bias
                src[need] = f"backup_{b_stn}"

        # 잔여 결측은 시점 선형 보간 (limit=3)
        before_interp_na = out[col].isna()
        out[col] = out[col].interpolate(method="time", limit=INTERP_LIMIT)
        interp_done = before_interp_na & out[col].notna()
        src[interp_done] = "interp"

        out[f"{col}_src"] = src

        # 통계 집계
        n_total = len(out)
        n_raw = (src == "raw").sum()
        n_backup = sum((src == f"backup_{b}").sum() for b in backups)
        n_interp = (src == "interp").sum()
        n_na = out[col].isna().sum()
        stats["columns"][col] = {
            "raw": n_raw, "backup": n_backup, "interp": n_interp, "na": n_na,
            "total": n_total,
        }

    # 강수·일조·일사는 raw 그대로 유지 (출처 컬럼 없음)
    return out, stats


def main() -> None:
    if not RAW_DIR.exists():
        sys.exit(f"ERROR: {RAW_DIR} 없음")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for stn, backups in BACKUP_MAP.items():
        out, stats = impute_one_station(stn, backups)
        out = out.reset_index()  # tm을 다시 컬럼으로
        out["tm"] = out["tm"].dt.strftime("%Y-%m-%d")
        out_csv = OUT_DIR / f"asos_stn{stn}_2010_2025_imputed.csv"
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        all_stats.append(stats)

    # 요약 출력 — 컬럼별 결측 잔존 + 출처 비율
    print(f"{'stn':>4s} " + "  ".join(
        f"{c:>20s}" for c in IMPUTE_COLS
    ))
    print(f"{'':>4s} " + "  ".join(
        f"{'raw/backup/interp/na':>20s}" for _ in IMPUTE_COLS
    ))
    for s in all_stats:
        stn = s["stn"]
        cells = []
        for c in IMPUTE_COLS:
            d = s["columns"][c]
            cells.append(f"{d['raw']:>4d}/{d['backup']:>3d}/{d['interp']:>3d}/{d['na']:>3d}")
        print(f"{stn:>4s} " + "  ".join(f"{x:>20s}" for x in cells))

    # 합계
    print()
    print("=== 합계 (16 관측소) ===")
    for c in IMPUTE_COLS:
        tot = {k: sum(s["columns"][c][k] for s in all_stats) for k in ["raw", "backup", "interp", "na"]}
        print(f"  {c:>8s}: raw={tot['raw']:,}  backup={tot['backup']:,}  interp={tot['interp']:,}  na={tot['na']:,}")


if __name__ == "__main__":
    main()
