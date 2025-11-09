"""
Problem 4.16 - Interval Estimation Policy Evaluation
문제 4.16: IE 정책 평가

Perform L = 1000 simulations of the interval estimation policy over a budget of N = 20 experiments
using θ^IE = 1. Let F^IE be the performance of a drug for a particular sample path.
Make the assumption that the true performance of a drug μ_x is given in Table 2,
and use the assumptions for the standard deviation of each belief from Table 1.
Also use the standard deviation σ^W = 5 for the experimental variation (문제에서는 σ^W=5라고 했으나 실제로는 0.5인 것으로 추정)

a) Compute the mean and standard deviation of the value of the policy F^IE(θ^IE) with θ^IE = 1
b) Evaluate the IE policy for θ^IE = (0, 0.2, 0.4, ..., 2.0) and plot F^IE(θ). What do you learn from this plot?
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import time

from MedicalDecisionDiabetesModel import MedicalDecisionDiabetesModel as MDDM
from MedicalDecisionDiabetesPolicy import MDDMPolicy

def run_problem_4_16():
    """
    문제 4.16 실행 함수
    """
    # 초기 파라미터
    seed = 19783167
    
    # 약물 이름
    x_names = ['M', 'Sens', 'Secr', 'AGI', 'PA']
    policy_names = ['IE']
    
    # Parameters from Table 1 and Table 2
    S0 = {
        'M': {
            'mu_0': 0.32, 'sigma_0': 0.12, 
            'mu_truth': 0.25, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'Sens': {
            'mu_0': 0.28, 'sigma_0': 0.09,
            'mu_truth': 0.30, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'Secr': {
            'mu_0': 0.30, 'sigma_0': 0.17,
            'mu_truth': 0.28, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'AGI': {
            'mu_0': 0.26, 'sigma_0': 0.15,
            'mu_truth': 0.34, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'PA': {
            'mu_0': 0.21, 'sigma_0': 0.11,
            'mu_truth': 0.24, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        }
    }
    
    # Additional parameters for Problem 4.16
    additional_params = {
        'sigma_w': 5.0,  # σ^W = 5 (Problem 4.13 기준)
        'N': 20,         # Budget: 20 experiments
        'L': 1000,       # Number of simulations
        'theta_start': 0,
        'theta_end': 2.1,
        'increment': 0.2,
        'truth_type': 'known',  # Use fixed truth values from Table 2
        'policy': 'IE'
    }
    
    # 실험 설정
    t_stop = int(additional_params['N'])
    L = int(additional_params['L'])
    theta_range = np.arange(additional_params['theta_start'],
                           additional_params['theta_end'],
                           additional_params['increment'])
    
    # 모델 초기화
    Model = MDDM(x_names, x_names, S0, additional_params)
    print("=" * 70)
    print("문제 4.16: IE 정책 평가")
    print("=" * 70)
    print("\n초기 설정:")
    Model.printTruth()
    Model.printState()
    
    # 정책 초기화
    P = MDDMPolicy(Model, policy_names, seed)
    
    # 결과 저장용 딕셔너리
    theta_obj_mean = []
    theta_obj_std = []
    
    # θ 값에 대한 루프
    print("\n시뮬레이션 시작...")
    for theta in theta_range:
        Model.prng = np.random.RandomState(seed)
        P.prng = np.random.RandomState(seed)
        
        F_hat = []  # 각 샘플 경로의 성능 저장
        
        start_time = time.time()
        
        # L번 시뮬레이션 (샘플 경로)
        for l in range(1, L+1):
            # 모델의 새 복사본 생성
            model_copy = copy(Model)
            
            # truth 샘플링 (known이므로 고정 값 사용)
            model_copy.exog_info_sample_mu()
            
            # 최적 치료법 결정
            best_treatment = max(model_copy.mu, key=model_copy.mu.get)
            
            # N번의 실험 수행
            for n in range(t_stop):
                # IE 정책에 따라 의사결정
                decision = P.IE(model_copy, theta)
                
                # 시간 진행 및 상태 업데이트
                exog_info = model_copy.step(decision)
            
            # 이 샘플 경로의 성능 저장
            F_hat.append(model_copy.obj)
        
        # θ에 대한 평균과 표준편차 계산
        F_hat_mean = np.array(F_hat).mean()
        F_hat_var = np.sum(np.square(np.array(F_hat) - F_hat_mean)) / (L - 1)
        F_hat_std = np.sqrt(F_hat_var / L)
        
        theta_obj_mean.append(F_hat_mean)
        theta_obj_std.append(F_hat_std)
        
        elapsed_time = time.time() - start_time
        print(f"θ = {theta:.1f}: F̄^IE = {F_hat_mean:.4f}, Std = {F_hat_std:.4f} (소요시간: {elapsed_time:.2f}초)")
    
    print("\n" + "=" * 70)
    print("Part (a): θ^IE = 1.0 결과")
    print("=" * 70)
    # θ = 1.0에 해당하는 인덱스 찾기
    theta_1_idx = np.where(np.isclose(theta_range, 1.0))[0][0]
    print(f"평균 (Mean): {theta_obj_mean[theta_1_idx]:.4f}")
    print(f"표준편차 (Std): {theta_obj_std[theta_1_idx]:.4f}")
    
    # Part (b): 플롯 생성
    print("\n" + "=" * 70)
    print("Part (b): θ에 따른 F^IE(θ) 플롯 생성")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 평균과 신뢰구간 플롯
    theta_obj_mean_arr = np.array(theta_obj_mean)
    theta_obj_std_arr = np.array(theta_obj_std)
    
    ax.plot(theta_range, theta_obj_mean_arr, 'bo-', label='Mean Performance', linewidth=2, markersize=8)
    ax.fill_between(theta_range, 
                     theta_obj_mean_arr - theta_obj_std_arr, 
                     theta_obj_mean_arr + theta_obj_std_arr,
                     alpha=0.3, label='Std Dev Band')
    
    ax.set_xlabel('theta (Exploration Parameter)', fontsize=12)
    ax.set_ylabel('Average Performance', fontsize=12)
    ax.set_title('Problem 4.16(b): IE Policy Performance\nTruth=Known, N=20, L=1000', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Mark best theta
    best_theta_idx = np.argmax(theta_obj_mean_arr)
    best_theta = theta_range[best_theta_idx]
    best_performance = theta_obj_mean_arr[best_theta_idx]
    ax.plot(best_theta, best_performance, 'r*', markersize=20, 
            label=f'Best: theta={best_theta:.1f}, F={best_performance:.4f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/Users/junchan/Documents/GitHub/stochastic-optimization/MedicalDecisionDiabetes/problem_4_16_plot.png', dpi=300)
    print(f"\n플롯이 저장되었습니다: problem_4_16_plot.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("결과 분석:")
    print("=" * 70)
    print(f"최적 θ 값: {best_theta:.1f}")
    print(f"최적 성능: {best_performance:.4f}")
    print("\n해석:")
    print("- θ가 0에 가까울수록 exploitation(현재 최선으로 보이는 약물 선택)")
    print("- θ가 클수록 exploration(불확실성이 높은 약물 탐색)")
    print("- 적절한 θ 값은 exploitation과 exploration의 균형을 맞춥니다.")
    print("- 그래프에서 최적 θ 값은 충분한 탐색과 활용의 균형점을 나타냅니다.")
    
    return theta_range, theta_obj_mean, theta_obj_std

if __name__ == "__main__":
    theta_range, theta_obj_mean, theta_obj_std = run_problem_4_16()

