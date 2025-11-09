"""
Problem 4.17(a) - IE Policy with Prior Distribution Mismatch
문제 4.17(a): 사전 분포 불일치 상황에서의 IE 정책 평가

Evaluate the IE policy given a budget N = 20 over the values θ^IE = (0, 0.2, 0.4, ..., 2.0) 
for the following truth setup:

a) First assume that the prior is μ_x^0 = 0.3 for all drugs x and where the initial standard deviation
   σ_x^0 = 0.10. This means we are assuming that the truth μ_x ≈ N(μ̄_x^0, (σ_x^0)^2). However, we are going
   to sample our truth using:
   
   μ̂_x = 0.3 + ε
   
   where ε is uniformly distributed in the interval [-0.15, +0.15]. This is an example of having a prior
   distribution of belief (in this case, that is normally distributed around 0.3) but sampling the truth
   from a different distribution (that is uniformly distributed around the mean 0.3). 
   
   Perform 10,000 repetitions of each value of θ^IE to compute the average performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import time

from MedicalDecisionDiabetesModel import MedicalDecisionDiabetesModel as MDDM
from MedicalDecisionDiabetesPolicy import MDDMPolicy

def run_problem_4_17a():
    """
    문제 4.17(a) 실행 함수
    Prior: 모든 약물에 대해 μ_x^0 = 0.3 (정규 분포)
    Truth: μ̂_x = 0.3 + ε, ε ~ Uniform[-0.15, +0.15]
    """
    # 초기 파라미터
    seed = 19783167
    
    # 약물 이름
    x_names = ['M', 'Sens', 'Secr', 'AGI', 'PA']
    policy_names = ['IE']
    
    # Parameters for Problem 4.17(a)
    # Prior: μ_x^0 = 0.3 for all drugs, σ_x^0 = 0.10
    S0 = {
        'M': {
            'mu_0': 0.3, 'sigma_0': 0.10,
            'mu_truth': 0.25, 'sigma_truth': 0,  # Not used for fixed_uniform
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'Sens': {
            'mu_0': 0.3, 'sigma_0': 0.10,
            'mu_truth': 0.30, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'Secr': {
            'mu_0': 0.3, 'sigma_0': 0.10,
            'mu_truth': 0.28, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'AGI': {
            'mu_0': 0.3, 'sigma_0': 0.10,
            'mu_truth': 0.34, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        },
        'PA': {
            'mu_0': 0.3, 'sigma_0': 0.10,
            'mu_truth': 0.24, 'sigma_truth': 0,
            'mu_fixed': 0.3, 'fixed_uniform_a': -0.15, 'fixed_uniform_b': 0.15,
            'prior_mult_a': -0.5, 'prior_mult_b': 0.5
        }
    }
    
    # Additional parameters
    additional_params = {
        'sigma_w': 5.0,
        'N': 20,
        'L': 10000,  # 10,000 repetitions
        'theta_start': 0,
        'theta_end': 2.1,
        'increment': 0.2,
        'truth_type': 'fixed_uniform',  # Truth sampled as μ̂ = 0.3 + Uniform[-0.15, +0.15]
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
    print("문제 4.17(a): IE 정책 평가 - Prior와 Truth 분포 불일치")
    print("=" * 70)
    print("\n초기 설정:")
    print(f"Prior: 모든 약물에 대해 μ_x^0 = 0.3, σ_x^0 = 0.10")
    print(f"Truth: μ̂_x = 0.3 + ε, ε ~ Uniform[-0.15, +0.15]")
    print(f"Budget: N = {t_stop}")
    print(f"Repetitions: L = {L}")
    print(f"σ^W = {additional_params['sigma_w']}")
    Model.printTruth()
    Model.printState()
    
    # 정책 초기화
    P = MDDMPolicy(Model, policy_names, seed)
    
    # 결과 저장용 딕셔너리
    theta_obj_mean = []
    theta_obj_std = []
    
    # θ 값에 대한 루프
    print("\n시뮬레이션 시작...")
    total_start = time.time()
    
    for theta_idx, theta in enumerate(theta_range):
        Model.prng = np.random.RandomState(seed)
        P.prng = np.random.RandomState(seed)
        
        F_hat = []  # 각 샘플 경로의 성능 저장
        
        theta_start_time = time.time()
        
        # L번 시뮬레이션 (샘플 경로)
        for l in range(1, L+1):
            # 모델의 새 복사본 생성
            model_copy = copy(Model)
            
            # truth 샘플링 (fixed_uniform: μ̂ = 0.3 + Uniform[-0.15, +0.15])
            model_copy.exog_info_sample_mu()
            
            # 최적 치료법 결정 (이 샘플 경로의 truth에 대해)
            best_treatment = max(model_copy.mu, key=model_copy.mu.get)
            
            # N번의 실험 수행
            for n in range(t_stop):
                # IE 정책에 따라 의사결정
                decision = P.IE(model_copy, theta)
                
                # 시간 진행 및 상태 업데이트
                exog_info = model_copy.step(decision)
            
            # 이 샘플 경로의 성능 저장
            F_hat.append(model_copy.obj)
            
            # 진행상황 출력 (매 1000번째)
            if l % 1000 == 0:
                elapsed = time.time() - theta_start_time
                print(f"  θ = {theta:.1f}: {l}/{L} 완료 ({l/L*100:.1f}%), 경과시간: {elapsed:.1f}초")
        
        # θ에 대한 평균과 표준편차 계산
        F_hat_mean = np.array(F_hat).mean()
        F_hat_var = np.sum(np.square(np.array(F_hat) - F_hat_mean)) / (L - 1)
        F_hat_std = np.sqrt(F_hat_var / L)
        
        theta_obj_mean.append(F_hat_mean)
        theta_obj_std.append(F_hat_std)
        
        elapsed_time = time.time() - theta_start_time
        total_elapsed = time.time() - total_start
        remaining = (len(theta_range) - theta_idx - 1) * elapsed_time
        
        print(f"θ = {theta:.1f}: F̄^IE = {F_hat_mean:.4f}, Std = {F_hat_std:.4f}")
        print(f"  소요시간: {elapsed_time:.1f}초, 총 경과: {total_elapsed/60:.1f}분, 예상 남은 시간: {remaining/60:.1f}분\n")
    
    total_time = time.time() - total_start
    print(f"\n총 시뮬레이션 시간: {total_time/60:.2f}분")
    
    # 플롯 생성
    print("\n" + "=" * 70)
    print("θ에 따른 F^IE(θ) 플롯 생성")
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
    ax.set_title('Problem 4.17(a): IE Policy Performance\n' + 
                 'Prior: Normal(0.3), Truth: Uniform[0.15, 0.45]\n' +
                 f'N=20, L={L}', fontsize=14)
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
    plt.savefig('/Users/junchan/Documents/GitHub/stochastic-optimization/MedicalDecisionDiabetes/problem_4_17a_plot.png', dpi=300)
    print(f"\n플롯이 저장되었습니다: problem_4_17a_plot.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("결과 분석:")
    print("=" * 70)
    print(f"최적 θ 값: {best_theta:.1f}")
    print(f"최적 성능: {best_performance:.4f}")
    print("\n해석:")
    print("- Prior는 정규분포(μ=0.3 중심)를 가정하지만, Truth는 균등분포로 샘플링됩니다.")
    print("- 이는 Prior와 Truth의 분포 불일치(distribution mismatch) 상황입니다.")
    print("- Prior가 모든 약물에 대해 동일하므로, 초기에는 어떤 약물이 더 나은지 알 수 없습니다.")
    print("- 따라서 충분한 exploration(높은 θ)이 중요할 수 있습니다.")
    print("- 그래프의 형태는 exploration-exploitation 균형의 중요성을 보여줍니다.")
    
    # 결과 테이블 출력
    print("\n상세 결과 테이블:")
    print("θ\t\tF̄^IE(θ)\t\tStd")
    print("-" * 50)
    for theta, mean, std in zip(theta_range, theta_obj_mean, theta_obj_std):
        print(f"{theta:.1f}\t\t{mean:.6f}\t{std:.6f}")
    
    return theta_range, theta_obj_mean, theta_obj_std

if __name__ == "__main__":
    theta_range, theta_obj_mean, theta_obj_std = run_problem_4_17a()

