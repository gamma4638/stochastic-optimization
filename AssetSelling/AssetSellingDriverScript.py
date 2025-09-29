"""
Asset selling driver script
"""

from collections import namedtuple
import pandas as pd
import numpy as np
from AssetSellingModel import AssetSellingModel
from AssetSellingPolicy import AssetSellingPolicy
import matplotlib.pyplot as plt
from copy import copy
import math

if __name__ == "__main__":
    # Excel 없이 실행되도록 예시 변수 선언
    policy_selected = 'full_grid'  # 'sell_low' | 'high_low' | 'track' | 'time_series' | 'full_grid'
    T = 20
    initPrice = 16
    initBias = 'Neutral'  # 'Up' | 'Neutral' | 'Down'

    # 외생 파라미터 (예시값)
    UpStep = 1.0
    DownStep = -1.0
    Variance = 2.0

    # time_series full_grid용 범위 (예시값)
    theta_min = 0.0
    theta_max = 100.0
    theta_step = 0.1

    # 정책 파라미터 리스트 (예시값)
    # index 0: sell_low(lower_limit, placeholder)
    # index 1: high_low(lower_limit, upper_limit)
    # index 2: track(track_signal, alpha)
    # index 3: time_series(theta, placeholder)
    param_list = [
        (80.0, 0.0),
        (70.0, 130.0),
        (5.0, 0.3),
        (2.0, 0.0) # time_series(theta, placeholder) -> 우리 정책에서 얼마만큼의 변동까지 hold 할건지 결정하는는 theta 값
    ]

    # 편향 전이 확률표 (예시값): 첫 열을 인덱스로 사용하고, 나머지 열은 Up/Neutral/Down 확률
    biasdf = pd.DataFrame({
        'State': ['Up', 'Neutral', 'Down'],
        'Up': [0.9, 0.1, 0],
        'Neutral': [0.2, 0.6, 0.2],
        'Down': [0.1, 0.1, 0.9]
    })

    exog_params = {'UpStep': UpStep, 'DownStep': DownStep, 'Variance': Variance, 'biasdf': biasdf}

    # 반복 설정 (예시값)
    nIterations = 50
    printStep = 10
    printIterations = [0]
    printIterations.extend(list(reversed(range(nIterations-1,0,-printStep))))  
    
    
    print("exog_params ",exog_params)
   
    # initialize the model and the policy
    policy_names = ['sell_low', 'high_low', 'track', 'time_series']
    state_names = ['price', 'resource','bias']
    init_state = {'price': initPrice, 'resource': 1,'bias':initBias}
    decision_names = ['sell', 'hold']

    
    M = AssetSellingModel(state_names, decision_names, init_state,exog_params,T)
    P = AssetSellingPolicy(M, policy_names)
    t = 0
    prev_price = init_state['price']


    # make a policy_info dict object
    # param_list[3] is expected to hold theta for time_series
    # Initialize previous prices with initial price for p_{t-1} and p_{t-2}
    policy_info = {'sell_low': param_list[0],
                   'high_low': param_list[1],
                   'track': param_list[2] + (prev_price,)}
    # Add time_series only if param_list has enough elements
    if len(param_list) > 3:
        policy_info['time_series'] = (param_list[3][0], prev_price, prev_price)
    else:
        # Default theta=1.0 for time_series if not provided
        policy_info['time_series'] = (1.0, prev_price, prev_price)
    
    
    if (policy_selected != 'full_grid'):
        print("Selected policy {}, time horizon {}, initial price {} and number of iterations {}".format(policy_selected,T,initPrice,nIterations))
        contribution_iterations=[P.run_policy(param_list, policy_info, policy_selected, t) for ite in list(range(nIterations))]

        contribution_iterations = pd.Series(contribution_iterations)
        print("Contribution per iteration: ")
        print(contribution_iterations)
        cum_avg_contrib = contribution_iterations.expanding().mean()
        print("Cumulative average contribution per iteration: ")
        print(cum_avg_contrib)
        
        #plotting the results
       
        fig, axsubs = plt.subplots(1,2,sharex=True,sharey=True)
        fig.suptitle("Asset selling using policy {} with parameters {} and T {}".format(policy_selected,policy_info[policy_selected],T) )
        i = np.arange(0, nIterations, 1)
        
        axsubs[0].plot(i, cum_avg_contrib, 'g')
        axsubs[0].set_title('Cumulative average contribution')
          
        axsubs[1].plot(i, contribution_iterations, 'g')
        axsubs[1].set_title('Contribution per iteration')
        
    
        # Create a big subplot
        ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        ax.set_ylabel('USD', labelpad=0) # Use argument `labelpad` to move label downwards.
        ax.set_xlabel('Iterations', labelpad=10)
        
        plt.show()
        
    else:
        # full grid: time_series 파라미터를 1D 그리드로 탐색
        num_steps = int(round((theta_max - theta_min) / theta_step)) + 1
        theta_values = np.linspace(theta_min, theta_max, num_steps)
        contribution_iterations = [P.vary_theta(param_list, policy_info, 'time_series', t, theta_values) for ite in list(range(nIterations))]
        contribution_iterations_arr = np.array(contribution_iterations)
        cum_sum_contrib = contribution_iterations_arr.cumsum(axis=0)
        nElem = np.arange(1,cum_sum_contrib.shape[0]+1).reshape((cum_sum_contrib.shape[0],1))
        cum_avg_contrib=cum_sum_contrib/nElem
        print("cum_avg_contrib")
        print(cum_avg_contrib)
        P.plot_heat_map_time_series(cum_avg_contrib, theta_values, printIterations)
        
        