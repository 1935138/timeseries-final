"""
data/raw/kosis_population_sido_2010_2025.csv 의 17개 시·도 인구를
세종 → 충남 통합하여 16개 단위로 정리하고 연도별 가중치를 산출한다.

산출물: data/processed/sido_weights_2010_2025.csv
  columns: year, sido_code, sido_name, population, weight

가중치 정의 (docs/research_log/2026-05-25_대표관측소_인구가중평균.md):
  - 16개 단위 (세종 통합): 서울/부산/대구/인천/광주/대전/울산/경기/강원/충북/
    충남(+세종)/전북/전남/경북/경남/제주
  - weight_i(y) = pop_i(y) / Σ_j pop_j(y)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "raw" / "kosis_population_sido_2010_2025.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "sido_weights_2010_2025.csv"

SEJONG_CODE = "36"
CHUNGNAM_CODE = "44"


def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"ERROR: 입력 파일 없음 — {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, dtype={"sido_code": str})
    print(f"입력: {INPUT_CSV.name} ({len(df):,}행)")

    # 1) 세종 → 충남에 인구 합산. 세종 행은 코드만 충남으로 바꿔서 groupby로 합산.
    df.loc[df["sido_code"] == SEJONG_CODE, "sido_code"] = CHUNGNAM_CODE
    df.loc[df["sido_code"] == CHUNGNAM_CODE, "sido_name"] = "충청남도(세종통합)"

    df = (
        df.groupby(["year", "sido_code", "sido_name"], as_index=False)["population"]
        .sum()
    )
    print(f"세종→충남 통합 후: {len(df):,}행")

    # 2) 연도별 가중치 산출
    df["weight"] = df.groupby("year")["population"].transform(lambda s: s / s.sum())

    # 3) 검증
    n_sido_by_year = df.groupby("year")["sido_code"].nunique()
    weight_sum_by_year = df.groupby("year")["weight"].sum()
    print("\n연도별 시·도 수 (모두 16이어야 정상):")
    print(n_sido_by_year.to_string())

    print("\n연도별 가중치 합 (모두 1.0):")
    print(weight_sum_by_year.round(10).to_string())

    assert (n_sido_by_year == 16).all(), "통합 후 시·도 수가 16이 아닙니다"
    assert ((weight_sum_by_year - 1).abs() < 1e-9).all(), "가중치 합이 1이 아닙니다"

    # 4) 수도권 비중 모니터링 (참고)
    capital = df[df["sido_code"].isin(["11", "28", "41"])]  # 서울·인천·경기
    cap_share = capital.groupby("year")["weight"].sum()
    print("\n수도권(서울·인천·경기) 비중 추세:")
    print(cap_share.round(4).to_string())

    # 5) 저장
    df = df[["year", "sido_code", "sido_name", "population", "weight"]]
    df = df.sort_values(["year", "sido_code"]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT_CSV} ({len(df):,}행)")


if __name__ == "__main__":
    main()
