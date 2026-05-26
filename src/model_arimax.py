"""
ARIMAX/SARIMAX 학습·평가 파이프라인.

설계 (결정 경위: docs/research_log/2026-05-26_1711_EDA_종속외생_진단.md):
  - 학습 2010-01-01 ~ 2024-12-31 (5,479일)
  - 평가 2025-01-01 ~ 2025-12-31 (365일, 홀드아웃)
  - 차수 3개 후보 × 외생변수 3서브셋 = 9개 조합 학습
  - AIC/BIC + Ljung-Box로 최적 1개 선택
  - 2025: (a) 365일 one-shot, (b) rolling 1-step 둘 다 평가

산출:
  - data/processed/arimax/grid_results.csv   9개 조합 AIC/BIC/Ljung-Box
  - data/processed/arimax/forecast_oneshot.csv  최적 + 베이스라인 예측
  - data/processed/arimax/forecast_rolling.csv  rolling 1-step
  - data/processed/arimax/metrics.csv         MAE/RMSE/MAPE/coverage
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "arimax"

SPLIT = "2024-12-31"

ORDERS: list[tuple[tuple, tuple]] = [
    ((1, 1, 1), (1, 1, 1, 7)),
    ((2, 1, 1), (1, 1, 1, 7)),
    ((1, 1, 2), (1, 1, 1, 7)),
]

EXOG_SUBSETS: dict[str, list[str]] = {
    "none": [],
    "minimal": [
        "cdd_18", "hdd_18", "is_weekend", "is_holiday", "ip_total", "pop_total",
    ],
    "phase1": [
        "cdd_18", "hdd_18", "cdd_18_ma7", "hdd_18_ma7",
        "hm_avg", "ws_avg", "rn_day", "ss_day",
        "heat_feels_th_2day_adv", "heat_feels_th_2day_wrn",
        "cold_th_2day_adv", "cold_th_2day_wrn",
        "is_weekend", "is_holiday",
        "ip_total", "pop_total",
    ],
}


@dataclass
class SplitData:
    y_train: pd.Series
    y_test: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    exog_mean: pd.Series
    exog_std: pd.Series

    def std_exog(self, names: list[str]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        if not names:
            return None, None
        mu, sd = self.exog_mean[names], self.exog_std[names].replace(0, 1.0)
        return (
            (self.X_train[names] - mu) / sd,
            (self.X_test[names] - mu) / sd,
        )


def load_data() -> SplitData:
    y = pd.read_csv(PROCESSED / "peak_mw_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm").sort_index()
    X = pd.read_csv(PROCESSED / "exog_daily_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm").sort_index()
    y.index.freq = "D"
    X.index.freq = "D"

    y_train = y.loc[:SPLIT, "peak_mw"]
    y_test = y.loc["2025":, "peak_mw"]
    X_train = X.loc[:SPLIT]
    X_test = X.loc["2025":]

    mu = X_train.mean(numeric_only=True)
    sd = X_train.std(numeric_only=True)

    return SplitData(y_train, y_test, X_train, X_test, mu, sd)


def fit_one(data: SplitData, order: tuple, sorder: tuple,
            exog_names: list[str]) -> tuple[object, dict]:
    exog_train, _ = data.std_exog(exog_names)
    model = SARIMAX(
        data.y_train,
        exog=exog_train,
        order=order,
        seasonal_order=sorder,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)
    lb = acorr_ljungbox(res.resid, lags=[10, 20], return_df=True)
    info = {
        "order": str(order),
        "seasonal_order": str(sorder),
        "exog_subset": "_".join(exog_names[:1] + ["..."]) if exog_names else "none",
        "n_exog": len(exog_names),
        "AIC": res.aic,
        "BIC": res.bic,
        "loglik": res.llf,
        "LB_p_lag10": lb.loc[10, "lb_pvalue"],
        "LB_p_lag20": lb.loc[20, "lb_pvalue"],
    }
    return res, info


def run_grid(data: SplitData) -> tuple[pd.DataFrame, dict]:
    rows = []
    fitted: dict[tuple, object] = {}
    for order, sorder in ORDERS:
        for subset_name, exog_names in EXOG_SUBSETS.items():
            print(f"  fit order={order} sorder={sorder} exog={subset_name}({len(exog_names)})")
            res, info = fit_one(data, order, sorder, exog_names)
            info["exog_subset"] = subset_name
            rows.append(info)
            fitted[(order, sorder, subset_name)] = res
    return pd.DataFrame(rows), fitted


def metrics(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict:
    err = y_true - y_pred
    return {
        "model": label,
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAPE_%": (err.abs() / y_true).mean() * 100,
        "bias_mean": err.mean(),
    }


def forecast_oneshot(res, data: SplitData, exog_names: list[str]) -> pd.Series:
    _, exog_test = data.std_exog(exog_names)
    fc = res.get_forecast(steps=len(data.y_test), exog=exog_test)
    pred = fc.predicted_mean
    pred.index = data.y_test.index
    return pred, fc.conf_int(alpha=0.05)


def baseline_naive(data: SplitData) -> pd.Series:
    """전일값"""
    last_train = data.y_train.iloc[-1]
    return pd.Series(
        np.r_[last_train, data.y_test.iloc[:-1].values],
        index=data.y_test.index, name="naive",
    )


def baseline_snaive(data: SplitData) -> pd.Series:
    """7일 전 같은 요일"""
    tail = data.y_train.iloc[-7:].values
    extended = np.concatenate([tail, data.y_test.values])
    snaive = extended[:-7][-len(data.y_test):]
    return pd.Series(snaive, index=data.y_test.index, name="snaive")


def rolling_one_step(data: SplitData, order: tuple, sorder: tuple,
                     exog_names: list[str]) -> pd.Series:
    """매일 실측을 추가하며 1-step ahead 예측. 재학습 비용을 줄이려고
    SARIMAX.append(refit=False)로 상태만 업데이트."""
    exog_train, exog_test = data.std_exog(exog_names)

    model = SARIMAX(
        data.y_train, exog=exog_train,
        order=order, seasonal_order=sorder,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=200)

    preds = []
    for i in range(len(data.y_test)):
        ex_step = exog_test.iloc[[i]] if exog_test is not None else None
        fc = res.get_forecast(steps=1, exog=ex_step)
        preds.append(float(fc.predicted_mean.iloc[0]))
        # append actual & exog (no refit)
        res = res.append(
            data.y_test.iloc[[i]],
            exog=ex_step,
            refit=False,
        )
    return pd.Series(preds, index=data.y_test.index, name="rolling")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 데이터 로드 ===")
    data = load_data()
    print(f"  train: {data.y_train.shape}, test: {data.y_test.shape}")

    print("\n=== 9개 조합 그리드 학습 ===")
    grid, fitted = run_grid(data)
    grid_path = OUT_DIR / "grid_results.csv"
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(grid.round(3).to_string(index=False))

    # 최적: AIC 최저
    best_idx = grid["AIC"].idxmin()
    best_row = grid.loc[best_idx]
    best_order = eval(best_row["order"])
    best_sorder = eval(best_row["seasonal_order"])
    best_subset = best_row["exog_subset"]
    best_exog = EXOG_SUBSETS[best_subset]
    print(f"\n최적 (AIC 최저): order={best_order} sorder={best_sorder} exog={best_subset}")

    print("\n=== 365일 one-shot 예측 ===")
    res = fitted[(best_order, best_sorder, best_subset)]
    pred_oneshot, ci = forecast_oneshot(res, data, best_exog)
    naive = baseline_naive(data)
    snaive = baseline_snaive(data)

    # SARIMA(no exog) 베이스라인: grid 결과 활용
    sarima_res = fitted[(best_order, best_sorder, "none")]
    pred_sarima, _ = forecast_oneshot(sarima_res, data, [])

    forecast_df = pd.DataFrame({
        "actual": data.y_test,
        "arimax_oneshot": pred_oneshot,
        "ci_low": ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values,
        "sarima_oneshot": pred_sarima,
        "naive": naive,
        "snaive": snaive,
    })

    print("\n=== rolling 1-step 예측 (ARIMAX) ===")
    rolling = rolling_one_step(data, best_order, best_sorder, best_exog)
    forecast_df["arimax_rolling"] = rolling

    forecast_df.to_csv(OUT_DIR / "forecast.csv", encoding="utf-8-sig")

    print("\n=== 평가 지표 ===")
    rows = [
        metrics(data.y_test, pred_oneshot, "ARIMAX one-shot"),
        metrics(data.y_test, rolling, "ARIMAX rolling 1-step"),
        metrics(data.y_test, pred_sarima, "SARIMA(no exog) one-shot"),
        metrics(data.y_test, naive, "Naive (전일)"),
        metrics(data.y_test, snaive, "Seasonal Naive (전주 동요일)"),
    ]
    # 95% PI 커버리지
    in_ci = ((data.y_test >= ci.iloc[:, 0].values) & (data.y_test <= ci.iloc[:, 1].values)).mean()
    rows[0]["PI95_cov_%"] = in_ci * 100

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics_df.round(2).to_string(index=False))

    print(f"\n저장: {OUT_DIR}/")


if __name__ == "__main__":
    main()
