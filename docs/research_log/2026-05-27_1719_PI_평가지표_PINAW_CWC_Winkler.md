# Research Log

## 2026-05-27 — 예측구간(PI) 평가지표 확장: PINAW · CWC · Winkler 추가

v3(2026-05-26_1815) 후속. 기존 코드는 95% PI 검증을 **PICP 단독**(`PI95_cov_%`)으로만 보고 있었음.
PICP는 구간을 무한히 넓히면 100%가 되는 함정이 있어, **구간 폭(sharpness)** 정보 없이는
"정직한 95%"인지 판단 불가. 논문 표준에 맞춰 PINAW / CWC / Winkler를 함께 산출.

### 산출물

| 파일 | 변경 |
|---|---|
| `src/model_arimax.py` | `pi_metrics()` 헬퍼 신규 + `main()`에서 활용 |
| `src/model_arimax_v2.py` | subset 루프에 PINAW/CWC/Winkler 컬럼 추가 |
| `src/model_arimax_v3.py` | `fit_eval()` 리턴 dict + `show_cols`에 추가 |
| `data/processed/arimax/v3_results.csv` | 컬럼 4개 신규 (PINAW, CWC, Winkler) |

### 1. 지표 정의

```python
def pi_metrics(y_true, ci_low, ci_high, alpha=0.05, eta=50.0):
    # PICP = P(y ∈ [lo, hi])
    # PINAW = mean(hi - lo) / (max(y) - min(y))
    # CWC = PINAW * (1 + γ * exp(-η*(PICP - μ)))    (Khosravi et al. 2011)
    #   γ = 1 if PICP < μ else 0,  μ = 1 - α = 0.95
    # Winkler = mean[(hi-lo) + (2/α)·max(0, lo-y) + (2/α)·max(0, y-hi)]
```

- **PICP** (Prediction Interval Coverage Probability): 명목 신뢰수준 95%에 가까울수록 calibration ↑
- **PINAW** (Prediction Interval Normalized Average Width): 0에 가까울수록 sharp.
  데이터 range로 정규화해 시계열 스케일에 무관.
- **CWC** (Coverage Width Criterion): PICP ≥ 95%면 PINAW와 동일,
  PICP < 95%면 `exp(-50·(PICP-0.95))`로 강한 페널티. **단일 숫자 랭킹**용.
- **Winkler / Interval Score**: proper scoring rule. 구간 밖 관측은 거리에 비례해 (2/α=40) 가중.
  학계 표준에 가장 가까운 단일 지표.

### 2. v3 단계별 비교 (새 지표 포함)

| 모델 | n_exog | MAE | MAPE | **PICP** | **PINAW** | **CWC** | **Winkler** |
|---|---|---|---|---|---|---|---|
| v2 baseline | 14 | 2,969 | 4.04% | **97.5%** | 0.597 | 0.597 | 27,377 |
| v3a log | 14 | 2,736 | 3.70% | 95.1% | 0.303 | 0.303 | 18,560 |
| v3b log+OR | 12 | 2,739 | 3.71% | 95.1% | 0.304 | 0.304 | 18,567 |
| **v3c log+OR+비선형** | 14 | **2,668** | **3.63%** | **95.3%** | **0.298** | **0.298** | **18,305** |

### 3. 핵심 발견 — v2 baseline은 "과잉 커버리지"였음

PICP만 봤을 땐 v2(97.5%)가 v3c(95.3%)보다 "더 안전해" 보였으나, 새 지표로 드러난 진실:

- v2의 PINAW = **0.597** — 구간폭이 데이터 range의 60%
- v3a 이후 PINAW = **0.30 수준** — 절반 이하로 sharp

즉 v2의 높은 PICP는 **구간을 넓혀서 커버한 결과**였고, 진짜 calibration이 좋아진 건
log 변환을 적용한 v3a 이후. v3c는 PICP 95.3%로 명목값에 거의 일치 + PINAW까지 최소.

**Winkler score**로도 v3c가 18,305으로 최저 — proper scoring rule 기준으로도 우승.

### 4. 의사결정

- 보고/논문 표는 PICP·PINAW·Winkler 3종 병기 (CWC는 PICP≥95%일 땐 PINAW와 동일해 중복)
- 모델 간 단일 숫자 비교가 필요할 땐 **Winkler** 우선 (이론적으로 가장 정당)
- v3c가 점예측(MAE/MAPE)뿐 아니라 **구간 예측 품질에서도 dominant** — 최종 모델 확정 유지

### 5. v2 그리드 재실행 결과 (9개 서브셋)

`model_arimax_v2.py`를 새 지표와 함께 재실행 → `v2_grid.csv`에 PINAW/CWC/Winkler 컬럼 채움.
MAE 오름차순 + PI 지표 병기:

| subset | n_exog | MAE | PICP | PINAW | CWC | Winkler |
|---|---|---|---|---|---|---|
| pop_cdd_both_hdd_both | 14 | 2,969 | 97.53% | 0.597 | 0.597 | 27,377 |
| ip_cdd_both_hdd_both | 14 | 3,039 | 96.99% | 1.159 | 1.159 | 52,830 |
| **v1_phase1** | 16 | 3,062 | **95.34%** | **0.437** | **0.437** | **23,711** |
| pop_cdd_both_hdd_only | 13 | 3,576 | 95.07% | 0.537 | 0.537 | 27,540 |
| ip_cdd_both_hdd_only | 13 | 3,831 | 96.16% | 0.558 | 0.558 | 28,434 |
| **pop_cdd_only_hdd_both** | 13 | 3,941 | **90.68%** | 0.452 | **4.364** | 26,223 |
| ip_cdd_only_hdd_both | 13 | 4,789 | 99.18% | 2.532 | 2.532 | 108,996 |
| pop_cdd_only_hdd_only | 12 | 5,998 | 98.08% | 2.032 | 2.032 | 89,163 |
| ip_cdd_only_hdd_only | 12 | 8,207 | 99.45% | 3.227 | 3.227 | 138,854 |

#### 5.1 발견

- **`v1_phase1`은 PI 품질 1위** (Winkler 23,711, PINAW 0.437) — MAE는 3등이지만 구간 예측은 v2 그리드 내 최강.
  v2가 다중공선성 정리로 외생변수를 줄였지만 PI sharpness 측면에선 손해.
- **CWC 페널티 작동 확인**: `pop_cdd_only_hdd_both`(PICP 90.68%)에서 CWC = 0.452 × exp(50·0.043) ≈ **4.36**.
  PINAW 대비 9.7배 폭증 — Khosravi 페널티가 의도대로 미달 모델을 강하게 깎아냄.
- **하지만 v3c(Winkler 18,305)는 v2 그리드 전체를 dominant** — `v1_phase1`보다도 23% 낮음.

### 6. rolling 1-step 추가 (v3c)

노트북에서 v2 vs v3c rolling 비교가 `KeyError: 'rolling'`로 실패 → v3 스크립트가 rolling을 생성하지 않던 잠복 버그 발견.
`rolling_one_step_log()` 신규 함수로 해결.

**구현 핵심**:
- SARIMAX `append(refit=False)`로 상태만 갱신, 재학습 비용 없음
- 매 step의 1-step 예측 평균을 `exp(μ + σ²/2)` 역변환 → 원 스케일 저장
- 실측은 `log(y_test[i])`로 append (학습 스케일 유지)

**결과** (one-shot 대비):

| 모델 | one-shot MAE | rolling MAE | 개선폭 |
|---|---|---|---|
| v2 | 2,969 | 1,523 | -49% |
| **v3c** | **2,668** | **1,357** | **-49%** |

- v3c rolling MAPE = **1.87%** — v2 rolling(2.09%) 대비 11% 추가 개선
- one-shot dominance(MAE -10.1%)가 rolling에서도 유지됨 → 운영 시나리오에서도 v3c 우승

### 7. 노트북 섹션 7 추가

`notebooks/modeling.ipynb`에 섹션 7 (PI 평가지표 4종 비교) 추가:

| 셀 | 내용 |
|---|---|
| markdown 헤더 | 4종 지표 정의 + 본 로그 링크 |
| v3 단계별 PI 표 | v2 → v3a → v3b → v3c |
| v2 그리드 PI 표 | 9개 subset, MAE 오름차순 |
| 4분할 시각화 | (좌상) PICP–PINAW 산점 / (우상) Winkler 랭킹 / (좌하) v3 4지표 정규화 / (우하) CWC 페널티 |

전체 `nbconvert --execute` 통과 확인.

### 8. 구현 메모

- `pi_metrics()`는 `model_arimax.py`에 정의 → v2/v3에서 import해 재사용
- v3는 log target 모델이므로 `exp(ci + σ²/2)` 역변환 **후** ci를 넘김 (원 스케일 비교 보장)
- rolling도 동일한 lognormal 편향 보정 적용 (`exp(μ + σ²/2)`)
- η=50, α=0.05 하이퍼파라미터는 Khosravi 원논문 권장값으로 박아둠

### 참고

- Khosravi, A., Nahavandi, S., Creighton, D., & Atiya, A. F. (2011).
  *Lower upper bound estimation method for construction of neural network-based prediction intervals.*
  IEEE Transactions on Neural Networks, 22(3), 337-346. — CWC 정의
- Gneiting, T., & Raftery, A. E. (2007). *Strictly proper scoring rules, prediction, and estimation.*
  JASA. — Winkler/Interval Score의 properness
