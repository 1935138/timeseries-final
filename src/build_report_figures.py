"""
팀원 공유용 핵심 시각화 6개 PNG 산출.

산출: reports/figures/
  fig01_peak_mw_timeline.png       종속변수 16년 전체 시계열
  fig02_exog_corr_heatmap.png      외생변수 + peak_mw 상관행렬
  fig03_stl_decomposition.png      STL 분해 (period=7)
  fig04_model_mae_compare.png      6개 모델 2025 MAE 수평 비교 (필수)
  fig05_v3c_forecast.png           v3c 365일 예측 vs 실측 + 95% PI (필수)
  fig06_monthly_mae_v2_v3c.png     v2 vs v3c 월별 MAE (필수)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트
NOTO_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if NOTO_PATH.exists():
    font_manager.fontManager.addfont(str(NOTO_PATH))
KOR_FONT = "Noto Sans CJK JP"
mpl.rcParams["font.family"] = KOR_FONT
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"
sns.set_theme(style="whitegrid", rc={"font.family": KOR_FONT})


def load_data():
    y = pd.read_csv(PROCESSED / "peak_mw_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    X = pd.read_csv(PROCESSED / "exog_daily_2010_2025.csv",
                    encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    return y, X


def fig01_peak_mw_timeline(y: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.index, y["peak_mw"], lw=0.4, color="steelblue")
    ax.set_title("일별 최대전력(peak_mw) — 2010-2025 (5,844일)")
    ax.set_xlabel("date"); ax.set_ylabel("MW")
    # 학습/평가 경계
    ax.axvline(pd.Timestamp("2025-01-01"), color="crimson", lw=1, ls="--", alpha=0.7)
    ax.text(pd.Timestamp("2025-01-01"), ax.get_ylim()[1]*0.95,
            "  2025 holdout", color="crimson", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_peak_mw_timeline.png")
    plt.close(fig)


def fig02_exog_corr_heatmap(y: pd.DataFrame, X: pd.DataFrame):
    df = X.join(y, how="inner")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=False, square=True, linewidths=0.3,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("외생변수 + peak_mw 상관행렬 (Pearson)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_exog_corr_heatmap.png")
    plt.close(fig)


def fig03_stl_decomposition(y: pd.DataFrame):
    from statsmodels.tsa.seasonal import STL
    s = y["peak_mw"].asfreq("D")
    stl = STL(s, period=7, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(s.index, stl.observed, lw=0.5); axes[0].set_ylabel("관측")
    axes[1].plot(s.index, stl.trend, lw=0.7, color="darkorange"); axes[1].set_ylabel("추세")
    axes[2].plot(s.index, stl.seasonal, lw=0.4, color="seagreen"); axes[2].set_ylabel("계절(7일)")
    axes[3].plot(s.index, stl.resid, lw=0.4, color="gray"); axes[3].set_ylabel("잔차")
    axes[3].set_xlabel("date")
    fig.suptitle("STL 분해 (주기=7일) — 주간 계절 진폭 약 22,290 MW (평균의 33%)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_stl_decomposition.png")
    plt.close(fig)


def fig04_model_mae_compare():
    """기존 metrics.csv + v3 결과를 통합한 6개 모델 비교."""
    # v1·SARIMA·Naive·Snaive
    m1 = pd.read_csv(PROCESSED / "arimax" / "metrics.csv", encoding="utf-8-sig")
    # v2 best
    v2 = pd.read_csv(PROCESSED / "arimax" / "v2_grid.csv", encoding="utf-8-sig")
    v2_best = v2.loc[v2["MAE_2025"].idxmin()]
    # v3c
    v3 = pd.read_csv(PROCESSED / "arimax" / "v3_results.csv", encoding="utf-8-sig")
    v3c = v3.loc[v3["model"].str.contains("v3c")].iloc[0]

    rows = [
        {"label": "Naive (전일)",
         "MAE": float(m1.loc[m1["model"].str.contains("Naive \\("), "MAE"].iloc[0])},
        {"label": "Seasonal Naive (전주)",
         "MAE": float(m1.loc[m1["model"].str.contains("Seasonal"), "MAE"].iloc[0])},
        {"label": "SARIMA (외생변수 없음)",
         "MAE": float(m1.loc[m1["model"].str.contains("SARIMA"), "MAE"].iloc[0])},
        {"label": "v1 ARIMAX (16 exog)",
         "MAE": float(m1.loc[m1["model"] == "ARIMAX one-shot", "MAE"].iloc[0])},
        {"label": "v2 ARIMAX (14 exog)",
         "MAE": float(v2_best["MAE_2025"])},
        {"label": "v3c ARIMAX log+비선형 ★",
         "MAE": float(v3c["MAE_2025"])},
    ]
    df = pd.DataFrame(rows).sort_values("MAE", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["seagreen" if "★" in l else
              ("crimson" if "SARIMA " in l or "Naive" in l else "steelblue")
              for l in df["label"]]
    bars = ax.barh(df["label"], df["MAE"], color=colors)
    for bar, val in zip(bars, df["MAE"]):
        ax.text(val + 100, bar.get_y() + bar.get_height() / 2, f"{val:,.0f}",
                va="center", fontsize=10)
    ax.set_xlabel("MAE (MW) — 2025 365일 one-shot 예측")
    ax.set_title("모델별 2025 홀드아웃 MAE 비교 (작을수록 우수)")
    ax.set_xlim(0, df["MAE"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_model_mae_compare.png")
    plt.close(fig)


def fig05_v3c_forecast():
    fc = pd.read_csv(PROCESSED / "arimax" / "v3c_forecast.csv",
                     encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(fc.index, fc["actual"], lw=1.0, color="black", label="실측")
    ax.plot(fc.index, fc["pred"], lw=0.9, color="seagreen", label="v3c 예측 (one-shot)")
    ax.fill_between(fc.index, fc["ci_low"], fc["ci_high"],
                    color="seagreen", alpha=0.15, label="95% PI")
    ax.set_title("v3c 2025년 365일 예측 vs 실측 (MAPE 3.63%, 95% PI 커버 95.3%)")
    ax.set_ylabel("peak_mw (MW)"); ax.set_xlabel("date")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_v3c_forecast.png")
    plt.close(fig)


def fig06_monthly_mae_v2_v3c():
    v2 = pd.read_csv(PROCESSED / "arimax" / "forecast.csv",
                     encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    v3c = pd.read_csv(PROCESSED / "arimax" / "v3c_forecast.csv",
                      encoding="utf-8-sig", parse_dates=["tm"]).set_index("tm")
    v2["err"] = v2["actual"] - v2["arimax_oneshot"]
    v3c["err"] = v3c["actual"] - v3c["pred"]
    v2["m"] = v2.index.month
    v3c["m"] = v3c.index.month
    monthly = pd.DataFrame({
        "v2": v2.groupby("m")["err"].apply(lambda x: x.abs().mean()),
        "v3c": v3c.groupby("m")["err"].apply(lambda x: x.abs().mean()),
    })

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(12); w = 0.4
    ax.bar(x - w/2, monthly["v2"], w, label="v2 (이전)", color="steelblue")
    ax.bar(x + w/2, monthly["v3c"], w, label="v3c (현재 최종)", color="seagreen")
    ax.set_xticks(x); ax.set_xticklabels(range(1, 13))
    ax.set_xlabel("월"); ax.set_ylabel("MAE (MW)")
    ax.set_title("월별 MAE — v2 vs v3c (2025 one-shot 예측)")
    ax.legend()
    # 주석: 9월 후퇴 (xticks는 0-indexed, 9월 = index 8)
    sep_v3c = monthly.loc[9, "v3c"]
    ax.annotate("9월 +74% 후퇴\n(추석 연휴)", xy=(8 + w/2, sep_v3c),
                xytext=(9.5, sep_v3c + 800), fontsize=9, color="crimson",
                ha="left",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_monthly_mae_v2_v3c.png")
    plt.close(fig)


def main():
    print("loading data...")
    y, X = load_data()
    print(f"  y={y.shape}, X={X.shape}")

    print("fig01 — peak_mw timeline")
    fig01_peak_mw_timeline(y)
    print("fig02 — exog correlation heatmap")
    fig02_exog_corr_heatmap(y, X)
    print("fig03 — STL decomposition")
    fig03_stl_decomposition(y)
    print("fig04 — model MAE comparison")
    fig04_model_mae_compare()
    print("fig05 — v3c forecast vs actual")
    fig05_v3c_forecast()
    print("fig06 — monthly MAE v2 vs v3c")
    fig06_monthly_mae_v2_v3c()

    print(f"\n저장 완료: {FIG_DIR}/")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
