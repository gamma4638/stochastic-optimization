# Stochastic Optimization - Policy Summary

이 문서는 코드베이스에 구현된 각 프로젝트별 정책(Policy)들을 정리한 것입니다.

---

## 1. AdaptiveMarketPlanning

**파일**: `AdaptiveMarketPlanningPolicy.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **Harmonic Rule** | 조화 스텝 크기 정책. `step_size = theta / (theta + t - 1)` 형태로 시간에 따라 감소하는 스텝 사이즈 사용 |
| **Kesten's Rule** | Kesten 규칙 기반 정책. `step_size = theta / (theta + counter - 1)` 형태로 상태 카운터 기반 스텝 사이즈 사용 |
| **Constant Rule** | 상수 스텝 크기 정책. `step_size = theta`로 고정된 스텝 사이즈 사용 |

### 특징
- 적응형 학습률(step size) 기반 의사결정
- 시장 계획 최적화를 위한 스토캐스틱 근사(Stochastic Approximation) 방법 적용

---

## 2. AssetSelling

**파일**: `AssetSellingPolicy.py`, `AssetSellingPolicy_Q3.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **Sell Low Policy** | 가격이 하한선(lower_limit) 미만일 때 매도 |
| **High Low Policy** | 가격이 하한선 미만이거나 상한선 초과일 때 매도 |
| **Track Policy** | 이동평균 기반 정책. 평활화된 가격(smoothed price)과 현재 가격 비교하여 매도/보유 결정. 가중치: 0.7(현재) + 0.2(t-1) + 0.1(t-2) |
| **Time Series Policy** | 시계열 정책. 최근 3시점 가중평균 예상가격에서 임계치 theta 이상 벗어나면 매도 |

### 특징
- Grid Search를 통한 최적 theta 파라미터 탐색
- Heat Map 시각화 지원
- 할인율(discount factor, gamma) 적용 가능

---

## 3. BloodManagement

**파일**: `BloodManagementPolicy.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **LP-based Policy** | 선형계획법(Linear Programming) 기반 혈액 관리 정책 |

### 특징
- CVXOPT 라이브러리를 사용한 LP 최적화
- 혈액 수요-공급 매칭 최적화
- VFA(Value Function Approximation) 업데이트 기능
  - Constant stepsize (`C`)
  - Adaptive stepsize (`A`) - AdaGrad 스타일
- 투영 알고리즘: `Avg`, `Copy`, 또는 iterative projection

---

## 4. ClinicalTrials

**파일**: `ClinicalTrialsPolicy.py`, `ClinicalTrialsPolicySolutionQ6.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **Model A Policy** | 결정론적 최단 경로 문제(Deterministic Shortest Path) 기반 Lookahead 정책. 고정된 horizon H 동안 Bellman 방정식 풀이 |
| **Model B Policy** | 확률론적 Lookahead 정책. 베타 분포 샘플링을 통한 기대값 계산 |
| **Model C Policy** | 하이브리드 정책. Backward ADP와 함수 근사(linear/quadratic fitting) 결합 |
| **Model C Extension Policy** | Model C의 확장. 등록 인원 수 변동과 반응률(l_response) 학습 포함 |

### 공통 정지 규칙
- `p_belief > theta_stop_high`: 약물 성공 선언
- `p_belief < theta_stop_low`: 약물 실패 선언

### 특징
- 베이지안 업데이트 (Beta-Binomial 모델)
- 환자 등록 비용과 프로그램 비용 최적화
- 샘플링 기반 파라미터 피팅

---

## 5. EnergyStorage_I

**파일**: `EnergyStoragePolicy.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **Buy Low Sell High Policy** | 가격이 하한선 이하면 구매, 상한선 이상이면 판매 |
| **Bellman Policy** | Bellman 방정식 기반 최적 정책. 이산화된 가격 변화에 대한 기대값 계산 |

### 특징
- Grid Search를 통한 최적 (theta_buy, theta_sell) 탐색
- Heat Map 시각화 지원
- 마지막 시점에서 자동 판매

---

## 6. MedicalDecisionDiabetes

**파일**: `MedicalDecisionDiabetesPolicy.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **UCB (Upper Confidence Bound)** | 탐색-활용 균형을 위한 UCB 정책. `mu + theta * sqrt(log(t+1) / n)` 형태 |
| **IE (Interval Estimation)** | 구간 추정 정책. `mu + theta / sqrt(beta)` 형태 |
| **Pure Exploitation** | 순수 활용 정책 (theta = 0). 현재까지 가장 좋은 약물 선택 |
| **Pure Exploration** | 순수 탐색 정책. 무작위로 약물 선택 |

### 특징
- 다중 팔 밴딧(Multi-Armed Bandit) 문제 프레임워크
- 당뇨병 치료약 선택 최적화
- 경험적 평균(mu_empirical)과 정밀도(beta) 추적

---

## 7. StochasticShortestPath_Dynamic

**파일**: `Policy.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **Lookahead Policy** | Percentile 기반 Lookahead 정책. Backward DP로 최적 경로 계산 |

### 특징
- 동적 그래프에서의 최단 경로 탐색
- Percentile 기반 비용 추정: `cost = mean * (1 - spread + 2 * spread * theta)`
- 시간 역순 Bellman 업데이트

---

## 8. StochasticShortestPath_Static

**파일**: `PolicyAdaptive.py`

### 구현된 정책들

| 정책명 | 설명 |
|--------|------|
| **Greedy Decision Policy** | 현재 링크 비용 + 미래 가치함수(V_t) 최소화 기반 의사결정 |

### 특징
- 정적 그래프에서의 적응형 최단 경로
- 실시간 링크 비용 정보 활용
- Dead-end 예외 처리

---

## 9. TwoNewsvendor

**파일**: `TwoNewsvendorPolicy.py`

### Field Agent 정책들

| 정책명 | 설명 |
|--------|------|
| **Regular** | 고정 bias 기반 수요 요청. `request = estimate - source_bias - central_bias + theta` |
| **Learning UCB** | UCB 학습 기반 bias 선택 |
| **Learning IE** | Interval Estimation 학습 기반 bias 선택 |

### Central Agent 정책들

| 정책명 | 설명 |
|--------|------|
| **Regular** | 고정 bias 기반 할당. `allocation = field_request - field_bias + theta` |
| **Punishing** | Field의 bias가 양수일 경우 벌칙 적용. `allocation = request - 2 * field_bias_hat` |
| **Learning UCB** | UCB 학습 기반 bias 선택 |
| **Learning IE** | Interval Estimation 학습 기반 bias 선택 |
| **Learning IE Two Estimates** | 두 추정치(Field, Source) 가중 결합 |

### 특징
- 두 에이전트(Field, Central) 간 게임 이론적 상호작용
- 다양한 학습 정책 조합 가능
- Heat Map을 통한 정책 조합 성능 비교

---

## 정책 분류 요약

### 의사결정 접근법별 분류

| 접근법 | 프로젝트 |
|--------|----------|
| **Parametric Cost Function Approximation (CFA)** | AssetSelling (High-Low, Track), EnergyStorage (Buy-Low-Sell-High) |
| **Lookahead Policy** | ClinicalTrials (Model A/B/C), StochasticShortestPath_Dynamic |
| **Value Function Approximation (VFA)** | BloodManagement, StochasticShortestPath_Static |
| **Policy Search (Stochastic Approximation)** | AdaptiveMarketPlanning (Harmonic, Kesten, Constant) |
| **Learning/Bandit Policies** | MedicalDecisionDiabetes (UCB, IE), TwoNewsvendor (Learning UCB/IE) |

### 학습 유형별 분류

| 유형 | 정책 |
|------|------|
| **Exploration-Exploitation Tradeoff** | UCB, Interval Estimation, Pure Exploration/Exploitation |
| **Stepsize Rules** | Harmonic, Kesten, Constant, AdaGrad-style |
| **Threshold-based** | Sell-Low, High-Low, Buy-Low-Sell-High |
| **Model-based Lookahead** | Bellman, Model A/B/C, Percentile-based |

---

## 파일 구조

```
stochastic-optimization/
├── AdaptiveMarketPlanning/
│   └── AdaptiveMarketPlanningPolicy.py
├── AssetSelling/
│   ├── AssetSellingPolicy.py
│   └── AssetSellingPolicy_Q3.py
├── BloodManagement/
│   └── BloodManagementPolicy.py
├── ClinicalTrials/
│   ├── ClinicalTrialsPolicy.py
│   └── ClinicalTrialsPolicySolutionQ6.py
├── EnergyStorage_I/
│   └── EnergyStoragePolicy.py
├── MedicalDecisionDiabetes/
│   └── MedicalDecisionDiabetesPolicy.py
├── StochasticShortestPath_Dynamic/
│   └── Policy.py
├── StochasticShortestPath_Static/
│   └── PolicyAdaptive.py
└── TwoNewsvendor/
    └── TwoNewsvendorPolicy.py
```
