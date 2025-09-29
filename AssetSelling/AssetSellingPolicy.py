"""
Asset selling policy class

"""
from collections import namedtuple
import pandas as pd
import numpy as np
from AssetSellingModel import AssetSellingModel
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from copy import copy
import math

class AssetSellingPolicy():
    """
    Base class for decision policy
    """

    def __init__(self, model, policy_names):
        """
        Initializes the policy

        :param model: the AssetSellingModel that the policy is being implemented on
        :param policy_names: list(str) - list of policies
        """
        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)

    def build_policy(self, info):
        """
        this function builds the policies depending on the parameters provided

        :param info: dict - contains all policy information
        :return: namedtuple - a policy object
        """
        return self.Policy(*[info[k] for k in self.policy_names])

    def sell_low_policy(self, state, info_tuple):
        """
        this function implements the sell-low policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = info_tuple[0]
        new_decision = {'sell': 1, 'hold': 0} if state.price < lower_limit else {'sell': 0, 'hold': 1}
        return new_decision

    def high_low_policy(self, state, info_tuple):
        """
        this function implements the high-low policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = info_tuple[0]
        upper_limit = info_tuple[1]
        new_decision = {'sell': 1, 'hold': 0} if state.price < lower_limit or state.price > upper_limit \
            else {'sell': 0, 'hold': 1}
        return new_decision

    def track_policy(self, state, info_tuple):
        """
        this function implements the track policy

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        track_signal = info_tuple[0]
        alpha = info_tuple[1]
        prev_price = info_tuple[2]
        smoothed_price = (1-alpha) * prev_price + alpha * state.price
        new_decision = {'sell': 1, 'hold': 0} if state.price >= smoothed_price + track_signal \
            else {'sell': 0, 'hold': 1}
        return new_decision

    def time_series_policy(self, state, info_tuple):
        """
        [과제용] 시계열(time_series) 정책 구현
        - 목적: 현재가가 최근 3시점 가중평균(예상가격)에서 임계치 theta 이상 벗어나면 즉시 매도
        - 입력: info_tuple = (theta, p_{t-1}, p_{t-2})
        - 출력: 매도/보유 의사결정 딕셔너리
        """
        theta = info_tuple[0]
        p_tm1 = info_tuple[1]
        p_tm2 = info_tuple[2]
        p_t = state.price
        p_tilde = 0.7 * p_t + 0.2 * p_tm1 + 0.1 * p_tm2
        new_decision = {'sell': 1, 'hold': 0} if (p_t < p_tilde - theta) or (p_t > p_tilde + theta) \
            else {'sell': 0, 'hold': 1}
        return new_decision

    def run_policy(self, param_list, policy_info, policy, time):
        """
        [공통] 선택한 정책으로 1회 시뮬레이션을 수행하고 총 기여도(수익)를 반환
        - time_series인 경우, 각 스텝에서 과거가격(p_{t-1}, p_{t-2})을 갱신하여 다음 스텝에 사용
        - 만기 직전 시점(t=T-1)에는 강제 매도
        """
        model_copy = copy(self.model)

        while model_copy.state.resource != 0 and time < model_copy.initial_args['T']:
            # build decision policy
            p = self.build_policy(policy_info)

            # make decision based on chosen policy
            if policy == "sell_low":
                decision = self.sell_low_policy(model_copy.state, p.sell_low)
            elif policy == "high_low":
                decision = self.high_low_policy(model_copy.state, p.high_low)
            elif policy == "track":
                decision = {'sell': 0, 'hold': 1} if time == 0 else self.track_policy(model_copy.state, p.track)
            elif policy == "time_series":
                decision = self.time_series_policy(model_copy.state, p.time_series)

            if (time == model_copy.initial_args['T'] - 1):
                 decision = {'sell': 1, 'hold': 0}  

            x = model_copy.build_decision(decision)
            print("time={}, obj={}, s.resource={}, s.price={}, x={}".format(time, model_copy.objective,
                                                                            model_copy.state.resource,
                                                                            model_copy.state.price, x))
            # update previous price
            prev_price = model_copy.state.price
            # step the model forward one iteration
            model_copy.step(x)
            # update track policy info with new previous price
            policy_info.update({'track': param_list[2] + (prev_price,)})
            # [과제용] time_series의 과거가격 메모리 이동: (theta, p_{t-1}, p_{t-2}) ← (theta, p_t, p_{t-1})
            if 'time_series' in self.policy_names and hasattr(p, 'time_series'):
                theta = p.time_series[0]
                prev1_old = p.time_series[1]
                policy_info.update({'time_series': (theta, prev_price, prev1_old)})
            # increment time
            time += 1
        print("obj={}, state.resource={}".format(model_copy.objective, model_copy.state.resource))
        contribution = model_copy.objective
        return contribution


        


    def grid_search_theta_values(self, low_min, low_max, high_min, high_max, increment_size):
        """
        this function gives a list of theta values needed to run a full grid search

        :param low_min: the minimum value/lower bound of theta_low
        :param low_max: the maximum value/upper bound of theta_low
        :param high_min: the minimum value/lower bound of theta_high
        :param high_max: the maximum value/upper bound of theta_high
        :param increment_size: the increment size over the range of theta values
        :return: list - list of theta values
        """

        # Convert pandas Series/scalars to floats
        if hasattr(low_min, 'iloc'):
            low_min = float(low_min.iloc[0])
        else:
            low_min = float(low_min)
        if hasattr(low_max, 'iloc'):
            low_max = float(low_max.iloc[0])
        else:
            low_max = float(low_max)
        if hasattr(high_min, 'iloc'):
            high_min = float(high_min.iloc[0])
        else:
            high_min = float(high_min)
        if hasattr(high_max, 'iloc'):
            high_max = float(high_max.iloc[0])
        else:
            high_max = float(high_max)
        if hasattr(increment_size, 'iloc'):
            increment_size = float(increment_size.iloc[0])
        else:
            increment_size = float(increment_size)

        low_steps = int(round((low_max - low_min) / increment_size)) + 1
        high_steps = int(round((high_max - high_min) / increment_size)) + 1

        theta_low_values = np.linspace(low_min, low_max, low_steps)
        theta_high_values = np.linspace(high_min, high_max, high_steps)

        theta_values = []
        for x in theta_low_values:
            for y in theta_high_values:
                theta = (x, y)
                theta_values.append(theta)

        return theta_values, theta_low_values, theta_high_values

    def vary_theta(self, param_list, policy_info, policy, time, theta_values):
        """
        [공통] theta 그리드의 각 값에 대해 기여도(1회 시뮬레이션)를 계산
        - time_series: 1차원 theta 스윕
        - high_low: (low, high) 2차원 스윕
        """
        contribution_values = []

        # Support both high_low (2D theta tuples) and time_series (1D theta scalars)
        if policy == "time_series":
            # [과제용] time_series: 시뮬레이션 시작 시 p_{t-1}, p_{t-2} 초기값 설정
            if 'time_series' in policy_info and len(policy_info['time_series']) >= 2:
                init_p_tm1 = policy_info['time_series'][1]
                init_p_tm2 = policy_info['time_series'][2] if len(policy_info['time_series']) >= 3 else init_p_tm1
            else:
                init_p_tm1 = self.model.state.price
                init_p_tm2 = self.model.state.price

            for theta in theta_values:
                t = time
                policy_dict = policy_info.copy()
                policy_dict.update({'time_series': (float(theta), init_p_tm1, init_p_tm2)})
                print("policy_dict={}".format(policy_dict))
                contribution = self.run_policy(param_list, policy_dict, "time_series", t)
                contribution_values.append(contribution)
            return contribution_values

        # default: high_low as before (theta is a (low, high) tuple)
        for theta in theta_values:
            t = time
            policy_dict = policy_info.copy()
            policy_dict.update({'high_low': theta})
            print("policy_dict={}".format(policy_dict))
            contribution = self.run_policy(param_list, policy_dict, policy, t)
            contribution_values.append(contribution)
        
        return contribution_values

    def plot_heat_map(self, contribution_values, theta_low_values, theta_high_values):
        """
        this function plots a heat map

        :param contribution_values: list - list of contribution values
        :param theta_low_values: list - list of theta_low_values
        :param theta_high_values: list - list of theta_high_values
        :return: none (plots a heat map)
        """
        contributions = np.array(contribution_values)
        increment_count = len(theta_low_values)
        contributions = np.reshape(contributions, (-1, increment_count))

        fig, ax = plt.subplots()
        im = ax.imshow(contributions, cmap='hot')
        # create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        # we want to show all ticks...
        ax.set_xticks(np.arange(len(theta_low_values)))
        ax.set_yticks(np.arange(len(theta_high_values)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(theta_low_values)
        ax.set_yticklabels(theta_high_values)
        # rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_title("Heatmap of contribution values across different values of theta")
        fig.tight_layout()
        plt.show()
        return True


    def plot_heat_map_many(self, contribution_values, theta_low_values, theta_high_values,iterations):
        """
        this function plots a heat map

        :param contribution_values: list - list of contribution values
        :param theta_low_values: list - list of theta_low_values
        :param theta_high_values: list - list of theta_high_values
        :return: none (plots a heat map)
        """
        fig, axsubs = plt.subplots(math.ceil(len(iterations)/2), 2)
        fig.suptitle("Heatmap of contribution values across different values of theta", fontsize=10)

        for ite,n in zip(iterations,list(range(len(iterations)))):
            contributions = np.array(contribution_values[ite])
            
            
            increment_count = len(theta_high_values)
            contributions = np.reshape(contributions, (-1, increment_count))
            contributions=contributions[::-1]
            


            print("Ite {}, n {} and plot ({},{})".format(ite,n,n // 2,n % 2))
            if (math.ceil(len(iterations)/2)>1):
                ax = axsubs[n // 2,n % 2]
            else:
                ax = axsubs[n % 2]
            
            im = ax.imshow(contributions, cmap='hot')
            cbar = ax.figure.colorbar(im, ax=ax)
            ax.set_yticks(np.arange(len(theta_low_values)))
            ax.set_xticks(np.arange(len(theta_high_values)))
            ax.set_yticklabels(list(reversed(theta_low_values)))
            ax.set_xticklabels(theta_high_values)
            
            
            # get the current labels 
            labelsx = [item.get_text() for item in ax.get_xticklabels()]
            ax.set_xticklabels([str(round(float(label), 2)) for label in labelsx])

            # get the current labels 
            labelsy = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_yticklabels([str(round(float(label), 2)) for label in labelsy])


            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
            
            ax.set_title("Iteration {}".format(ite))

        # Create a big subplot
        ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        ax.set_ylabel('Theta sell low values', labelpad=0) # Use argument `labelpad` to move label downwards.
        ax.set_xlabel('Theta sell high values', labelpad=10)

        

            
        fig.tight_layout()
        plt.show()
        return True


    def grid_search_theta_values_1d(self, theta_min, theta_max, increment_size):
        """
        [과제용] 단일 파라미터 정책(time_series)용 1차원 theta 그리드 생성
        - 입력: 최소/최대/증분
        - 출력: theta 값 배열
        """
        # allow passing as Series from Excel
        if hasattr(theta_min, 'iloc'):
            theta_min = float(theta_min.iloc[0])
        if hasattr(theta_max, 'iloc'):
            theta_max = float(theta_max.iloc[0])
        if hasattr(increment_size, 'iloc'):
            increment_size = float(increment_size.iloc[0])

        num_steps = int(round((theta_max - theta_min) / increment_size)) + 1
        theta_values = np.linspace(theta_min, theta_max, num_steps)
        return theta_values

    def vary_theta_time_series(self, param_list, policy_info, time, theta_values):
        """
        [과제용] time_series 정책: 1차원 theta 그리드 각각에 대한 기여도 계산
        - 입력: (theta, p_{t-1}, p_{t-2}) 초기화 포함
        - 출력: 각 theta의 기여도 리스트
        """
        contribution_values = []
        # [과제용] 시뮬레이션 시작 시 과거가격 초기화
        if 'time_series' in policy_info and len(policy_info['time_series']) >= 2:
            init_p_tm1 = policy_info['time_series'][1]
            init_p_tm2 = policy_info['time_series'][2] if len(policy_info['time_series']) >= 3 else init_p_tm1
        else:
            # fallback: use current state's price twice if not provided
            init_p_tm1 = self.model.state.price
            init_p_tm2 = self.model.state.price

        for theta in theta_values:
            t = time
            policy_dict = policy_info.copy()
            policy_dict.update({'time_series': (float(theta), init_p_tm1, init_p_tm2)})
            contribution = self.run_policy(param_list, policy_dict, 'time_series', t)
            contribution_values.append(contribution)

        return contribution_values

    def plot_heat_map_time_series(self, cum_avg_contrib, theta_values, iterations):
        """
        [과제용] time_series 정책의 "objective(contribution) vs. theta" 선 그래프
        - 입력: cum_avg_contrib(반복×theta 누적평균), theta_values, iterations(겹쳐 그릴 반복 인덱스)
        - 출력: 최종 평균 곡선 + 선택 반복 곡선 + 최적 theta 표시
        """
        contributions = np.array(cum_avg_contrib)
        thetas = np.array(theta_values)

        if contributions.ndim != 2:
            raise ValueError("cum_avg_contrib must be 2D: (nIterations, nTheta)")
        if thetas.ndim != 1:
            raise ValueError("theta_values must be 1D")

        if contributions.shape[1] != thetas.shape[0]:
            raise ValueError("Second dim of cum_avg_contrib must equal len(theta_values)")

        final_avg = contributions[-1]

        fig, ax = plt.subplots()
        ax.plot(thetas, final_avg, 'o-', label='Final average')

        # 선택된 반복 인덱스를 점선으로 함께 표시(옵션)
        if iterations is not None:
            for ite in iterations:
                if isinstance(ite, (int, np.integer)) and 0 <= ite < contributions.shape[0]:
                    ax.plot(thetas, contributions[ite], '--', alpha=0.35, label=f'Iteration {ite}')

        # 최적 theta(최종 평균 최대)를 빨간 점으로 표시
        if final_avg.size > 0:
            best_index = int(np.nanargmax(final_avg))
            ax.scatter([thetas[best_index]], [final_avg[best_index]], color='red', zorder=5,
                       label=f'Best theta = {thetas[best_index]:.2f}')

        ax.set_xlabel('theta')
        ax.set_ylabel('Expected contribution')
        ax.set_title('Time-series policy: contribution vs. theta')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        plt.show()
        return True