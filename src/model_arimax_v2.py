"""
ARIMAX v2 — 다중공선성 정리 그리드.

v1 결과 (2026-05-26_1830 로그) 후속:
  - `is_weekend` 계수 ≈ 0 → 제외 (SARIMA s=7이 흡수)
  - `ip_total` 계수 음수 — `pop_total`과 다중공선성 → 둘 중 하나만 검토
  - `cdd_18`·`cdd_18_ma7` 동시 투입 — 같은 정보 중복 → 하나만 검토

설계:
  - 차수 고정: (1,1,2)(1,1,1,7) (v1 최적)
  - 외생변수 그리드 = {ip_total, pop_total} × {cdd_18, cdd_18+ma7} × {hdd_18, hdd_18+ma7} = 8개
  - 비교: AIC + 2025 365일 one-shot MAE/MAPE
  - v1 phase1 16개를 같은 표에 baseline으로 포함

산출: data/processed/arimax/v2_grid.csv (서브셋별 AIC, BIC, MAE, RMSE, MAPE)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_arimax import (
    EXOG_SUBSETS,
    fit_one,
    forecast_oneshot,
    load_data,
    metrics,
)

ORDER = (1, 1, 2)
SORDER = (1, 1, 1, 7)
OUT_CSV = ROOT / "data" / "processed" / "arimax" / "v2_grid.csv"

# 공통 외생변수 (제거된 is_weekend 외)
COMMON = [
    "hm_avg", "ws_avg", "rn_day", "ss_day",
    "heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
    "cold_th_2day_adv", "cold_th_2day_wrn",
    "is_holiday",
]

TREND_OPTS = {"ip": ["ip_total"], "pop": ["pop_total"]}
CDD_OPTS = {"cdd_only": ["cdd_18"], "cdd_both": ["cdd_18", "cdd_18_ma7"]}
HDD_OPTS = {"hdd_only": ["hdd_18"], "hdd_both": ["hdd_18", "hdd_18_ma7"]}


def build_subsets() -> dict[str, list[str]]:
    """8개 서브셋 + phase1 baseline."""
    subsets: dict[str, list[str]] = {}
    for t_name, t_cols in TREND_OPTS.items():
        for c_name, c_cols in CDD_OPTS.items():
            for h_name, h_cols in HDD_OPTS.items():
                name = f"{t_name}_{c_name}_{h_name}"
                subsets[name] = COMMON + t_cols + c_cols + h_cols
    subsets["v1_phase1"] = EXOG_SUBSETS["phase1"]
    return subsets


def main() -> None:
    print("=== v2 다중공선성 정리 그리드 ===")
    data = load_data()
    subsets = build_subsets()

    rows = []
    for name, exog_names in subsets.items():
        print(f"  fit subset={name} (n_exog={len(exog_names)})")
        res, info = fit_one(data, ORDER, SORDER, exog_names)
        pred, ci = forecast_oneshot(res, data, exog_names)
        m = metrics(data.y_test, pred, name)
        in_ci = ((data.y_test >= ci.iloc[:, 0].values) & (data.y_test <= ci.iloc[:, 1].values)).mean()
        rows.append({
            "subset": name,
            "n_exog": len(exog_names),
            "AIC": info["AIC"],
            "BIC": info["BIC"],
            "LB_p_lag10": info["LB_p_lag10"],
            "MAE_2025": m["MAE"],
            "RMSE_2025": m["RMSE"],
            "MAPE_2025_%": m["MAPE_%"],
            "bias_2025": m["bias_mean"],
            "PI95_cov_%": in_ci * 100,
        })

    df = pd.DataFrame(rows).sort_values("MAE_2025").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== 결과 (2025 MAE 오름차순) ===")
    print(df.round(2).to_string(index=False))

    best = df.iloc[0]
    print(f"\n최적 (MAE 최저): {best['subset']}")
    print(f"  MAE={best['MAE_2025']:,.0f}, MAPE={best['MAPE_2025_%']:.2f}%, AIC={best['AIC']:,.0f}")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
