"""
ARIMAX v3 — 잔차·극단 계절 개선 3단계 (log → OR더미 → 비선형 기온).

v2 (2026-05-26_1734_ARIMAX_v2_다중공선성_정리.md) 후속.
v2 한계 진단:
  - 잔차 skew -2.30, kurtosis 28.8 (정규성 깨짐)
  - Ljung-Box p<0.001 (자기상관 잔존)
  - 8월·1월 극단 계절 MAE 4,500+ (4-5월 1,500의 3배)
  - 희소 더미 4개 중 3개 p>0.4 (사실상 noise)

v3 개선 3단계:
  - v3a: log 변환 — 잔차 정규성·분산 안정화
  - v3b: v3a + 희소 더미 OR 통합 — heat_th_2day_any, cold_th_2day_any (14→12개)
  - v3c: v3b + 비선형 기온항 — cdd_18²·hdd_18² (12→14개)

설계 고정:
  - 차수: (1,1,2)(1,1,1,7) (v1·v2 공통 최적)
  - 학습/평가 split: v2와 동일 (2010-2024 / 2025)
  - 외생변수 표준화 통계: 학습 구간에서만

산출: data/processed/arimax/v3_results.csv (v2 + 3단계 비교)
      data/processed/arimax/v3c_forecast.csv (최종 모델 예측)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_arimax import SplitData, load_data, metrics

ORDER = (1, 1, 2)
SORDER = (1, 1, 1, 7)
OUT_DIR = ROOT / "data" / "processed" / "arimax"

# v2 확정 외생변수 (14개)
V2_EXOG = [
    "cdd_18", "cdd_18_ma7", "hdd_18", "hdd_18_ma7",
    "hm_avg", "ws_avg", "rn_day", "ss_day",
    "heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
    "cold_th_2day_adv", "cold_th_2day_wrn",
    "is_holiday", "pop_total",
]


def derive_extra_features(data: SplitData) -> SplitData:
    """v3b·v3c용 파생 변수 부착 (학습 평균 사용은 std_exog가 알아서 함)."""
    X_train = data.X_train.copy()
    X_test = data.X_test.copy()

    # v3b: OR 통합
    X_train["heat_th_2day_any"] = (
        (X_train["heat_feels_th_2day_adv"] | X_train["heat_feels_th_2day_wrn"]).astype(int)
    )
    X_train["cold_th_2day_any"] = (
        (X_train["cold_th_2day_adv"] | X_train["cold_th_2day_wrn"]).astype(int)
    )
    X_test["heat_th_2day_any"] = (
        (X_test["heat_feels_th_2day_adv"] | X_test["heat_feels_th_2day_wrn"]).astype(int)
    )
    X_test["cold_th_2day_any"] = (
        (X_test["cold_th_2day_adv"] | X_test["cold_th_2day_wrn"]).astype(int)
    )

    # v3c: 비선형 기온항 (제곱)
    X_train["cdd_18_sq"] = X_train["cdd_18"] ** 2
    X_train["hdd_18_sq"] = X_train["hdd_18"] ** 2
    X_test["cdd_18_sq"] = X_test["cdd_18"] ** 2
    X_test["hdd_18_sq"] = X_test["hdd_18"] ** 2

    mu = X_train.mean(numeric_only=True)
    sd = X_train.std(numeric_only=True)

    return SplitData(data.y_train, data.y_test, X_train, X_test, mu, sd)


def std_exog(data: SplitData, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu, sd = data.exog_mean[names], data.exog_std[names].replace(0, 1.0)
    return (
        (data.X_train[names] - mu) / sd,
        (data.X_test[names] - mu) / sd,
    )


def fit_eval(
    data: SplitData,
    exog_names: list[str],
    label: str,
    log_target: bool,
) -> dict:
    """단일 변형 학습·평가. log_target이면 log 변환 후 exp 역변환."""
    exog_train, exog_test = std_exog(data, exog_names) if exog_names else (None, None)

    if log_target:
        y_train = np.log(data.y_train)
    else:
        y_train = data.y_train

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)

    fc = res.get_forecast(steps=len(data.y_test), exog=exog_test)
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    pred.index = data.y_test.index

    if log_target:
        # log-가우시안 역변환 (편향 보정: exp(μ + σ²/2))
        var = res.params.get("sigma2", res.scale)
        pred = np.exp(pred + var / 2)
        ci.iloc[:, 0] = np.exp(ci.iloc[:, 0] + var / 2)
        ci.iloc[:, 1] = np.exp(ci.iloc[:, 1] + var / 2)

    in_ci = ((data.y_test.values >= ci.iloc[:, 0].values) &
             (data.y_test.values <= ci.iloc[:, 1].values)).mean()

    m = metrics(data.y_test, pred, label)

    # 잔차 진단 — 칼만 필터 워밍업 outlier 제거 위해 첫 60일 trim
    # (SARIMA s=7 초기 상태가 안정화되기까지 ~8주 필요)
    resid = res.resid.iloc[60:]
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)

    return {
        "model": label,
        "log_target": log_target,
        "n_exog": len(exog_names),
        "AIC": res.aic,
        "BIC": res.bic,
        "MAE_2025": m["MAE"],
        "RMSE_2025": m["RMSE"],
        "MAPE_2025_%": m["MAPE_%"],
        "bias_2025": m["bias_mean"],
        "PI95_cov_%": in_ci * 100,
        "resid_skew": float(stats.skew(resid)),
        "resid_kurt": float(stats.kurtosis(resid)),
        "LB_p_lag10": lb.loc[10, "lb_pvalue"],
        "LB_p_lag20": lb.loc[20, "lb_pvalue"],
        "_fit_obj": res,
        "_pred": pred,
        "_ci": ci,
    }


def main() -> None:
    print("=== v3 단계별 개선 ===")
    data = load_data()
    data = derive_extra_features(data)
    print(f"train: {data.y_train.shape}, test: {data.y_test.shape}")

    # 단계별 외생변수 셋
    v3a_exog = V2_EXOG  # 동일
    v3b_exog = [c for c in V2_EXOG if c not in {
        "heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
        "cold_th_2day_adv", "cold_th_2day_wrn",
    }] + ["heat_th_2day_any", "cold_th_2day_any"]
    v3c_exog = v3b_exog + ["cdd_18_sq", "hdd_18_sq"]

    results = []

    print("\n[v2 재학습 — baseline (log 없음, 14 exog)]")
    r = fit_eval(data, V2_EXOG, "v2 (baseline)", log_target=False)
    results.append(r)

    print("[v3a — log 변환만, 14 exog]")
    r = fit_eval(data, v3a_exog, "v3a log", log_target=True)
    results.append(r)

    print("[v3b — log + OR더미 통합, 12 exog]")
    r = fit_eval(data, v3b_exog, "v3b log+OR더미", log_target=True)
    results.append(r)

    print("[v3c — log + OR더미 + 비선형 기온, 14 exog]")
    r = fit_eval(data, v3c_exog, "v3c log+OR+nonlinear", log_target=True)
    results.append(r)

    # 비교 표
    cmp_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])
    cmp_df.to_csv(OUT_DIR / "v3_results.csv", index=False, encoding="utf-8-sig")

    print("\n=== 단계별 비교 ===")
    show_cols = ["model", "n_exog", "AIC", "MAE_2025", "MAPE_2025_%", "PI95_cov_%",
                 "resid_skew", "resid_kurt", "LB_p_lag10"]
    print(cmp_df[show_cols].round(3).to_string(index=False))

    # 최종 모델 예측 저장 (v3c)
    final = results[-1]
    out = pd.DataFrame({
        "actual": data.y_test,
        "pred": final["_pred"].values,
        "ci_low": final["_ci"].iloc[:, 0].values,
        "ci_high": final["_ci"].iloc[:, 1].values,
    }, index=data.y_test.index)
    out.to_csv(OUT_DIR / "v3c_forecast.csv", encoding="utf-8-sig")

    # 계수 (v3c)
    print("\n=== v3c 외생변수 계수 ===")
    res = final["_fit_obj"]
    exog_idx = [n for n in res.params.index if n in v3c_exog]
    coef = pd.DataFrame({
        "var": exog_idx,
        "coef": res.params.loc[exog_idx].values,
        "p": res.pvalues.loc[exog_idx].values,
    }).sort_values("coef", key=abs, ascending=False).reset_index(drop=True)
    coef["sig"] = (coef.p < 0.05).map({True: "✓", False: ""})
    print(coef.round(4).to_string(index=False))

    print(f"\n저장: {OUT_DIR}/v3_results.csv, v3c_forecast.csv")


if __name__ == "__main__":
    main()
