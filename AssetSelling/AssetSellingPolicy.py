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

        과제 b) 문제에서 제공되는 정책함수의 코드 구현현

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - (theta, p_tm1, p_tm2)
        :return: a decision made based on the policy
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
        this function runs the model with a selected policy

        :param param_list: list of policy parameters in tuple form (read in from an Excel spreadsheet)
        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param time: float - start time
        :return: float - calculated contribution
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
            # update time_series policy info shifting previous prices
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
        this function calculates the contribution for each theta value in a list

        :param param_list: list of policy parameters in tuple form (read in from an Excel spreadsheet)
        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param time: float - start time
        :param theta_values: list - list of all possible thetas to be tested
        :return: list - list of contribution values corresponding to each theta
        """
        contribution_values = []

        # Support both high_low (2D theta tuples) and time_series (1D theta scalars)
        if policy == "time_series":
            # establish initial previous prices
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
        Builds a 1D grid of theta values for policies with a single parameter (e.g., time_series)

        :param theta_min: float or pandas.Series with one value
        :param theta_max: float or pandas.Series with one value
        :param increment_size: float or pandas.Series with one value
        :return: numpy.ndarray of theta values
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
        Calculates contributions for each theta in 1D list for the time_series policy

        :param param_list: list - not used directly, kept for signature consistency
        :param policy_info: dict - includes key 'time_series': (theta, p_tm1, p_tm2)
        :param time: float - start time
        :param theta_values: list/ndarray - theta candidates
        :return: list of contributions corresponding to each theta
        """
        contribution_values = []
        # initial previous prices
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
        Plots contribution vs. theta for the time_series policy.

        :param cum_avg_contrib: 2D array-like of shape (nIterations, nTheta)
                                 cumulative average contributions across iterations
        :param theta_values: 1D array-like of theta grid values
        :param iterations: list of iteration indices to optionally overlay (can be empty)
        :return: True when plotting completes
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

        # Optionally overlay selected iterations
        if iterations is not None:
            for ite in iterations:
                if isinstance(ite, (int, np.integer)) and 0 <= ite < contributions.shape[0]:
                    ax.plot(thetas, contributions[ite], '--', alpha=0.35, label=f'Iteration {ite}')

        # Mark best theta on final average curve
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