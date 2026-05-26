"""
22지점 ASOS 시간자료 결측을 (1) 인접 지점 bias 보정 → (2) 시간 선형 보간으로
대체. 대표 16지점만 산출하고 백업 6지점은 입력으로만 사용.

방법론: docs/research_log/2026-05-25_1623_기상결측_대체보간_방법론.md (일자료)와
동일한 원칙을 시간자료에 적용:
  - bias = mean(main_train) - mean(backup_train), train period: ≤ 2024-12-31
  - bias는 연중 단일값 (시간별로 쪼개면 표본·노이즈 문제)
  - 시간 선형 보간 한계: 연속 결측 12시간 (반나절)

입력: data/raw/asos_hourly/asos_hourly_stn<지점>_2010_2025.csv (22개)
산출: data/processed/asos_hourly_imputed/asos_hourly_stn<지점>_2010_2025_imputed.csv
  (대표 16개)
  컬럼: tm, stn, ta, hm, ws, ta_src, hm_src, ws_src
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "asos_hourly"
OUT_DIR = ROOT / "data" / "processed" / "asos_hourly_imputed"

# 일자료와 동일 백업 매핑
BACKUP_MAP: dict[str, list[str]] = {
    "108": ["119", "112"],
    "119": ["108", "112"],
    "112": ["108", "119"],
    "101": ["114", "105"],
    "131": ["232", "133"],
    "232": ["133", "131"],
    "133": ["232", "131"],
    "146": ["156", "165"],
    "156": ["165", "146"],
    "165": ["156", "168"],
    "136": ["137", "138"],
    "155": ["159", "152"],
    "159": ["152", "155"],
    "152": ["159", "138"],
    "143": ["137", "136"],
    "184": ["189"],
}

IMPUTE_COLS = ["ta", "hm", "ws"]
TRAIN_END = "2024-12-31 23:00"

# 시간 선형 보간 한계 (연속 결측 시간 — 반나절)
INTERP_LIMIT_HOURS = 12


def load_station(stn: str) -> pd.DataFrame:
    """관측소 시간자료 로드. 중복 시각(KMA가 보정·재발표한 케이스, 지점당
    수~수십건)은 수치 평균으로 통합."""
    p = RAW_DIR / f"asos_hourly_stn{stn}_2010_2025.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=["tm"])
    # 중복 시각 평균 통합 (수치 컬럼만)
    num_cols = [c for c in df.columns if c not in ("tm", "stn")]
    df = df.groupby("tm", as_index=True)[num_cols].mean().sort_index()
    # 시간축 완전성 보장 — 누락 시각은 NaN으로 채워 보간 대상에 포함
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = "tm"
    df["stn"] = stn
    return df


def compute_bias(main: pd.Series, backup: pd.Series) -> float:
    m = main.loc[:TRAIN_END]
    b = backup.loc[:TRAIN_END]
    common = m.notna() & b.notna()
    if common.sum() == 0:
        return 0.0
    return float((m[common] - b[common]).mean())


def impute_station(main_stn: str, backups: list[str]) -> tuple[pd.DataFrame, dict]:
    main_df = load_station(main_stn)
    backup_dfs = {b: load_station(b) for b in backups}

    out = main_df.copy()
    stats = {"stn": main_stn, "columns": {}}

    for col in IMPUTE_COLS:
        src = pd.Series("raw", index=out.index, dtype="object")
        src[out[col].isna()] = pd.NA

        for b_stn in backups:
            b_df = backup_dfs[b_stn]
            if col not in b_df.columns:
                continue
            bias = compute_bias(main_df[col], b_df[col])
            # 같은 시간 인덱스에서 결측 + 백업 가용
            b_aligned = b_df[col].reindex(out.index)
            need = out[col].isna() & b_aligned.notna()
            if need.any():
                out.loc[need, col] = b_aligned.loc[need] + bias
                src[need] = f"backup_{b_stn}"

        # 시간 선형 보간
        before_interp = out[col].isna()
        out[col] = out[col].interpolate(method="time", limit=INTERP_LIMIT_HOURS)
        interp_done = before_interp & out[col].notna()
        src[interp_done] = "interp"

        out[f"{col}_src"] = src

        n_total = len(out)
        n_raw = (src == "raw").sum()
        n_backup = sum((src == f"backup_{b}").sum() for b in backups)
        n_interp = (src == "interp").sum()
        n_na = out[col].isna().sum()
        stats["columns"][col] = {
            "raw": n_raw, "backup": n_backup, "interp": n_interp,
            "na": n_na, "total": n_total,
        }

    return out, stats


def main() -> None:
    if not RAW_DIR.exists():
        sys.exit(f"ERROR: {RAW_DIR} 없음")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for stn, backups in BACKUP_MAP.items():
        print(f"[{stn}] 보간 시작 (backups={backups})", flush=True)
        out, stats = impute_station(stn, backups)
        out = out.reset_index()
        out["tm"] = out["tm"].dt.strftime("%Y-%m-%d %H:00")
        keep = ["tm", "stn"] + [
            c for col in IMPUTE_COLS for c in (col, f"{col}_src")
        ]
        out = out[keep]
        out_csv = OUT_DIR / f"asos_hourly_stn{stn}_2010_2025_imputed.csv"
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        all_stats.append(stats)
        cells = []
        for c in IMPUTE_COLS:
            d = stats["columns"][c]
            cells.append(f"{c}: raw {d['raw']:,} / bk {d['backup']} / ip {d['interp']} / na {d['na']}")
        print(f"   완료 → {out_csv.name}  |  " + "  |  ".join(cells), flush=True)

    print(f"\n=== 16 대표 관측소 보간 합계 ===")
    for c in IMPUTE_COLS:
        tot = {k: sum(s["columns"][c][k] for s in all_stats) for k in ["raw", "backup", "interp", "na"]}
        grand = sum(tot.values())
        print(
            f"  {c}: raw={tot['raw']:,} ({tot['raw']/grand*100:.3f}%)  "
            f"backup={tot['backup']:,}  interp={tot['interp']:,}  na={tot['na']:,}"
        )


if __name__ == "__main__":
    main()
