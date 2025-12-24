# 문제 4.16과 4.17 구현

이 디렉토리는 의료 의사결정(당뇨병 치료) 최적화 문제를 위한 Python 모듈을 포함합니다.
엑셀 파일 의존성을 제거하고 파라미터를 코드 내에서 직접 정의하도록 수정했습니다.

## 파일 구조

### 핵심 모듈
- `MedicalDecisionDiabetesModel.py`: 의료 의사결정 모델 클래스
- `MedicalDecisionDiabetesPolicy.py`: 의사결정 정책 클래스 (UCB, IE, PureExploitation, PureExploration)
- `MedicalDecisionDiabetesDriverScript.py`: 기존 드라이버 스크립트 (엑셀 연동 제거됨)

### 문제 해결 스크립트
- `problem_4_16.py`: 문제 4.16 해결 (IE 정책 평가, L=1,000)
- `problem_4_17a.py`: 문제 4.17(a) 해결 (Prior와 Truth 분포 불일치, L=10,000)
- `problem_4_17b.py`: 문제 4.17(b) 해결 (Prior 기반 Truth 샘플링, L=10,000)

## 주요 변경사항

### 1. 엑셀 파일 의존성 제거
- 기존: `pd.read_excel()`로 파라미터 로드
- 변경: 파라미터를 딕셔너리로 직접 코드에 정의

### 2. 모델 클래스 수정
- `MedicalDecisionDiabetesModel` 클래스가 DataFrame과 딕셔너리 모두 지원
- 하위 호환성 유지

### 3. 파라미터 구조
```python
S0 = {
    'M': {
        'mu_0': 0.32, 'sigma_0': 0.12,
        'mu_truth': 0.25, 'sigma_truth': 0,
        'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
        'prior_mult_a': -0.5, 'prior_mult_b': 0.5
    },
    # ... 다른 약물들
}

additional_params = {
    'sigma_w': 0.5,
    'N': 20,
    'L': 1000,
    'theta_start': 0,
    'theta_end': 2.1,
    'increment': 0.2,
    'truth_type': 'known',  # 'known', 'fixed_uniform', 'prior_uniform'
    'policy': 'IE'
}
```

## 문제 설명

### 문제 4.16: IE 정책 평가 (고정된 Truth)

**설정:**
- Budget: N = 20 실험
- Simulations: L = 1,000
- Truth type: 'known' (Table 2의 고정 값 사용)
- θ^IE 범위: 0, 0.2, 0.4, ..., 2.0

**부문제:**
- (a) θ^IE = 1에서 평균과 표준편차 계산
- (b) 모든 θ 값에 대해 F^IE(θ) 계산 및 플롯

**실행 방법:**
```bash
cd /Users/junchan/Documents/GitHub/stochastic-optimization/MedicalDecisionDiabetes
python problem_4_16.py
```

**예상 출력:**
- 콘솔: 각 θ 값에 대한 평균과 표준편차
- 플롯: `problem_4_16_plot.png` (θ vs 성능 그래프)

**예상 실행 시간:** 약 5-10분

---

### 문제 4.17(a): Prior와 Truth 분포 불일치

**설정:**
- Budget: N = 20 실험
- Simulations: L = 10,000
- Prior: 모든 약물에 대해 μ_x^0 = 0.3, σ_x^0 = 0.10
- Truth type: 'fixed_uniform'
  - μ̂_x = 0.3 + ε
  - ε ~ Uniform[-0.15, +0.15]
- θ^IE 범위: 0, 0.2, 0.4, ..., 2.0

**목적:**
Prior는 정규분포를 가정하지만, Truth는 균등분포로 샘플링되는 분포 불일치 상황에서의 정책 성능 평가

**실행 방법:**
```bash
cd /Users/junchan/Documents/GitHub/stochastic-optimization/MedicalDecisionDiabetes
python problem_4_17a.py
```

**예상 출력:**
- 콘솔: 각 θ 값에 대한 평균과 표준편차 (진행 상황 포함)
- 플롯: `problem_4_17a_plot.png`

**예상 실행 시간:** 약 30-60분 (10,000 반복)

---

### 문제 4.17(b): Prior 기반 Truth 샘플링

**설정:**
- Budget: N = 20 실험
- Simulations: L = 10,000
- Prior: Table 2의 A1C reduction 값 사용
  - M: 0.32, Sens: 0.28, Secr: 0.30, AGI: 0.26, PA: 0.21
- Truth type: 'prior_uniform'
  - μ_x = μ̄_x^0 + ε
  - ε ~ Uniform[-0.5 * μ̄_x^0, +0.5 * μ̄_x^0]
- θ^IE 범위: 0, 0.2, 0.4, ..., 2.0

**목적:**
각 약물의 prior가 다르고, truth가 prior를 중심으로 ±50% 범위에서 샘플링되는 상황에서의 정책 성능 평가

**실행 방법:**
```bash
cd /Users/junchan/Documents/GitHub/stochastic-optimization/MedicalDecisionDiabetes
python problem_4_17b.py
```

**예상 출력:**
- 콘솔: 각 θ 값에 대한 평균과 표준편차, Prior 정보
- 플롯: `problem_4_17b_plot.png`

**예상 실행 시간:** 약 30-60분 (10,000 반복)

---

## 약물 정보

### 5가지 약물 (x_names)
- **M**: Metformin (메트포르민)
- **Sens**: Sensitizers (민감도 개선제)
- **Secr**: Secretagogues (분비 촉진제)
- **AGI**: Alpha-glucosidase inhibitors (알파-글루코시다아제 억제제)
- **PA**: Peptide analogs (펩타이드 유사체)

### Table 1 파라미터 (Prior)
| 약물 | μ_0 (A1C reduction) | σ_0 |
|------|---------------------|-----|
| M    | 0.32                | 0.12|
| Sens | 0.28                | 0.19|
| Secr | 0.30                | 0.17|
| AGI  | 0.26                | 0.15|
| PA   | 0.21                | 0.21|

### Table 2 파라미터 (Truth)
| 약물 | Truth (특정 환자) |
|------|------------------|
| M    | 0.25             |
| Sens | 0.30             |
| Secr | 0.28             |
| AGI  | 0.34             |
| PA   | 0.24             |

## 정책 설명

### IE (Interval Estimation) 정책
IE 정책은 각 약물의 신뢰구간 상한을 기준으로 의사결정합니다:

```
decision = argmax_x (μ̄_x + θ / √β_x)
```

여기서:
- μ̄_x: 약물 x의 현재 평균 추정치
- β_x: 약물 x의 정밀도 (precision)
- θ: exploration 파라미터

**θ의 역할:**
- θ = 0: Pure exploitation (현재 최선으로 보이는 약물만 선택)
- θ > 0: Exploration 포함 (불확실성이 높은 약물도 탐색)
- θ가 클수록 더 많은 exploration 수행

## 결과 해석

### 그래프 분석 포인트
1. **최적 θ 값**: 어떤 θ에서 최고 성능이 나타나는가?
2. **Exploration-Exploitation 균형**: θ가 너무 낮거나 높을 때의 성능 저하
3. **표준편차**: 성능의 변동성 (불확실성)
4. **분포 불일치의 영향**: 4.17(a)와 4.17(b)의 차이

### 예상되는 인사이트
- Prior 정보의 품질이 최적 θ 값에 영향
- Prior와 Truth의 분포 불일치가 클수록 exploration이 더 중요
- Budget(N)이 제한적일 때 적절한 θ 선택의 중요성

## 의존성

필요한 Python 패키지:
```bash
pip install numpy matplotlib
```

주의: pandas는 더 이상 필요하지 않습니다 (엑셀 연동 제거됨)

## 문제 해결

### 실행 시 오류가 발생하는 경우
1. Python 경로 확인: `which python` 또는 `which python3`
2. 필요 패키지 설치: `pip install numpy matplotlib`
3. 작업 디렉토리 확인: `pwd`

### 메모리 부족 오류
- L = 10,000이 너무 크면 L 값을 줄여서 시도 (예: L = 5,000)
- 각 스크립트에서 `additional_params['L']` 값 수정

### 실행 시간이 너무 긴 경우
- θ 범위를 줄이거나 increment를 늘림
- 예: `theta_end=1.2`, `increment=0.4`

## 라이센스 및 참고

이 코드는 확률적 최적화(Stochastic Optimization) 교육용으로 작성되었습니다.

**참고 문헌:**
- Powell, W. B. (2019). *A unified framework for stochastic optimization*. European Journal of Operational Research.
- Chapter 4: Learning Policies

## 연락처

문제가 있거나 질문이 있으면 이슈를 생성해주세요.


