"""
종속변수(일별 최대전력) 정제.

입력: data/한국전력거래소_일별 최대전력기준 전력수급 정보_2010-2025.csv
산출: data/processed/peak_mw_2010_2025.csv  (tm, peak_mw — 5,844행)

설계:
  - 원본은 (년, 월, 일) + 다중 컬럼. 본 모델은 `peak_mw`(최대전력)만 사용.
  - 공급능력·예비율 등은 역인과(수요의 결과) → 외생변수에서 제외
    (`docs/research_log/2026-05-25_1937_경제인구_추세변수_추가결정.md`).
  - 키 컬럼은 `tm`(YYYY-MM-DD)으로 통일 — 모든 processed 파일과 정합.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "한국전력거래소_일별 최대전력기준 전력수급 정보_2010-2025.csv"
OUT_CSV = ROOT / "data" / "processed" / "peak_mw_2010_2025.csv"

DATE_START = "2010-01-01"
DATE_END = "2025-12-31"
EXPECTED_ROWS = 5844


def main() -> None:
    if not IN_CSV.exists():
        sys.exit(f"ERROR: {IN_CSV} 없음.")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

    df["tm"] = pd.to_datetime(
        df[["년", "월", "일"]].rename(columns={"년": "year", "월": "month", "일": "day"})
    ).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"최대전력(MW)": "peak_mw"})[["tm", "peak_mw"]]
    df = df.sort_values("tm").reset_index(drop=True)

    full = pd.DataFrame({"tm": pd.date_range(DATE_START, DATE_END, freq="D").strftime("%Y-%m-%d")})
    out = full.merge(df, on="tm", how="left")

    missing = out["peak_mw"].isna().sum()
    assert missing == 0, f"peak_mw 결측 {missing}건 (정합 기간 누락 가능성)"
    assert len(out) == EXPECTED_ROWS, f"기대 {EXPECTED_ROWS:,}행, 실제 {len(out):,}"

    print(f"정제 완료: {len(out):,}행")
    print(f"기간: {out['tm'].iloc[0]} ~ {out['tm'].iloc[-1]}")
    print(f"peak_mw 통계: min={out['peak_mw'].min():,} / mean={out['peak_mw'].mean():,.0f} / max={out['peak_mw'].max():,}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
