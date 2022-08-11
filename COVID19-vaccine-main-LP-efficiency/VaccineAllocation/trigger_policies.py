'''
This module includes different trigger policies that are simulated
'''
#breakpoint()
import json
import numpy as np
from VaccineAllocation import config
from itertools import product, permutations
import copy 
import iteround
import datetime as dt
CONSTANT_TR = 'constant'
STEP_TR = 'step'
LINEAR_TR = 'linear'
THRESHOLD_TYPES = [CONSTANT_TR, STEP_TR, LINEAR_TR]
datetime_formater = '%Y-%m-%d %H:%M:%S'

def build_multi_tier_policy_candidates(instance, tiers, threshold_type='constant', lambda_start=None):
    assert len(tiers) >= 2, 'At least two tiers need to be defined'
    threshold_candidates = []
    if threshold_type == CONSTANT_TR:
        #breakpoint()
        gz = config['grid_size']
        # lambda_start is given by the field pub; if it is none, then we use the square root staffing rule
        if lambda_start is None:
            if np.size(instance.epi.eq_mu) == 1:
                lambda_start = int(np.floor(instance.epi.eq_mu * instance.lambda_star))
            else:
                lambda_start = int(np.floor(np.max(instance.epi.eq_mu) * instance.lambda_star))
        params_trials = []
        for tier in tiers:
            if 'candidate_thresholds' in tier and isinstance(tier['candidate_thresholds'], list):
                params_trials.append(tier['candidate_thresholds'])
            else:
                candidates = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                params_trials.append(np.unique(candidates))
        for policy_params in product(*params_trials):
            is_valid = True
            for p_ix in range(len(policy_params) - 1):
                if policy_params[p_ix] >= policy_params[p_ix + 1]:
                    is_valid = False
                    break
            if is_valid:
                T = len(instance.cal)
                lockdown_thresholds = [[policy_params[i]] * T for i in range(len(policy_params))]
                threshold_candidates.append(lockdown_thresholds)
        #breakpoint()
        return threshold_candidates
    elif threshold_type == STEP_TR:
        #breakpoint()
        # TODO: we need to set a time and two levels
        gz = config['grid_size']
        lambda_start = int(np.floor(instance.epi.eq_mu * instance.lambda_star))
        params_trials = []
        for tier in tiers:
            if tier['candidate_thresholds'] is not None:
                # construct the trial parameters according to the candidate threshold
                # the candidate threshold should be a list of 2 lists
                if isinstance(tier['candidate_thresholds'][0], list):
                    candidates1 = tier['candidate_thresholds'][0]
                else:
                    candidates1 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                if isinstance(tier['candidate_thresholds'][1], list):
                    candidates2 = tier['candidate_thresholds'][1]
                else:
                    candidates2 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
            else:
                candidates1 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                candidates2 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
            params_trials.append([(t1, t2) for t1 in candidates1 for t2 in candidates2 if t1 <= t2])
        # obtain the possible stepping time points, limited to the start of months
        T_trials = instance.cal.month_starts
        for policy_params in product(*params_trials):
            is_valid = True
            for p_ix in range(len(policy_params) - 1):
                if (policy_params[p_ix][0] >= policy_params[p_ix + 1][0]) \
                        or (policy_params[p_ix][1] >= policy_params[p_ix + 1][1]):
                    is_valid = False
                    break
            if is_valid:
                T = len(instance.cal)
                for tChange in T_trials:
                    lockdown_thresholds = [[policy_params[i][0]] * tChange + [policy_params[i][1]] * (T - tChange)
                                           for i in range(len(policy_params))]
                    threshold_candidates.append(lockdown_thresholds)
        return threshold_candidates   
    elif threshold_type == LINEAR_TR:
        gz = config['grid_size'] 
        sgz = config['slope_grid_size'] # the grid size for slope search. 
        max_slope = config['max_slope']
        # The date the thresholds start to increase (constant threshold up until this point):
        slope_start = dt.datetime.strptime(config['slope_start_date'], datetime_formater)
        T_slope_start = np.where(np.array(instance.cal.calendar) == slope_start)[0][0]
        
        # lambda_start is given by the field pub; if it is none, then we use the square root staffing rule
        if lambda_start is None:
            if np.size(instance.epi.eq_mu) == 1:
                lambda_start = int(np.floor(instance.epi.eq_mu * instance.lambda_star))
            else:
                lambda_start = int(np.floor(np.max(instance.epi.eq_mu) * instance.lambda_star))
        params_trials = []
        for tier in tiers:
            # construct the trial parameters according to the candidate threshold
            if 'candidate_thresholds' in tier and isinstance(tier['candidate_thresholds'], list):
                params_trials.append(tier['candidate_thresholds'])
            else:
                candidates = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                params_trials.append(np.unique(candidates))
        for policy_params in product(*params_trials):
            is_valid = True
            for p_ix in range(len(policy_params) - 1):
                if policy_params[p_ix] >= policy_params[p_ix + 1]:
                    is_valid = False
                    break
            if is_valid:
                T = len(instance.cal)
                T_slope = T - T_slope_start
                if config['candidate_slopes'] is not None:
                    slope_candidates = config['candidate_slopes']
                else:    
                    slope_candidates = [sgz * i for i in range(0, int(max_slope / sgz) + 1)] + [max_slope]
                # If the intercept is valid, create grids according to slope:
                for slope in slope_candidates:  
                    # The lower bound for green will be constant at -1:
                    lockdown_thresholds = [[policy_params[0]] * T]
                    # The lower bound will be increasing for other stages:
                    lockdown_thresholds += [[policy_params[i]] * T_slope_start + \
                                           [policy_params[i] + (slope) * (t + 1) for t in range(T_slope)] 
                                           for i in range(1, len(policy_params))]
                        
                    threshold_candidates.append(lockdown_thresholds)
        #breakpoint()
        return threshold_candidates
    else:
        raise NotImplementedError
    
class MultiTierPolicy:
    '''
        A multi-tier policy allows for multiple tiers of lock-downs.
        Attrs:
            tiers (list of dict): a list of the tiers characterized by a dictionary
                with the following entries:
                    {
                        "transmission_reduction": float [0,1)
                        "cocooning": float [0,1)
                        "school_closure": int {0,1}
                    }
            
            lockdown_thresholds (list of list): a list with the thresholds for every
                tier. The list must have n-1 elements if there are n tiers. Each threshold
                is a list of values for evert time step of simulation.
            tier_type: functional form of the threshold (options are in THRESHOLD_TYPES)
            community_tranmission: (deprecated) CDC's old community tranmission threshold for staging. 
                                    Not in use anymore.
    '''
    def __init__(self, instance, tiers, lockdown_thresholds, tier_type, community_tranmission):
        assert len(tiers) == len(lockdown_thresholds)
        self.tiers = tiers
        self.tier_type = tier_type
        self.community_tranmission = community_tranmission
        self.lockdown_thresholds = lockdown_thresholds

        self.tier_history = None
        self._instance = instance
        
    def reset_tier_history(self):
        self.tier_history = None
    
    def compute_cost(self):
        return sum(self.tiers[i]['daily_cost'] for i in self.tier_history if i is not None and i in range(len(self.tiers)))
    
    def __repr__(self):
        return str(self.lockdown_thresholds)
    
    def __call__(self, t, ToIHT, IH, ToIY, ICU):
        '''
            Function that makes an instance of a policy a callable.
            Args:
                t (int): time period in the simulation
                z (object): deprecated, but maintained to avoid changes in the simulate function
                criStat (ndarray): the trigger statistics, previously daily admission, passed by the simulator
                IH (ndarray): hospitalizations admissions, passed by the simulator
                ** kwargs: additional parameters that are passed and are used elsewhere
        '''
        N = self._instance.N

        if self.tier_history is None:
            self.tier_history = [None for i in range(t)]

        if len(self.tier_history) > t:
            return

        ToIHT = np.array(ToIHT)
        IH = np.array(IH)
        ToIY = np.array(ToIY)
        ICU = np.array(ICU)

        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - config['moving_avg_len'])
        criStat_total = ToIHT.sum((1, 2))
        criStat_avg = criStat_total[moving_avg_start:].mean()

        # print(criStat_avg)
        
        # Compute new cases per 100k:
        if len(ToIY) > 0:
            ToIY_avg = ToIY.sum((1,2))[moving_avg_start:].sum()* 100000/np.sum(N, axis=(0,1))
        else:
            ToIY_avg = 0
            
        current_tier = self.tier_history[t-1]
        T = self._instance.T

        # find new tier
        counter = 0
        lb_threshold = 0
        for lt in self.lockdown_thresholds:
            if criStat_avg >= lt:
                lb_threshold = counter
                counter += 1
                if counter == len(self.lockdown_thresholds):
                    break

        new_tier = lb_threshold
        
        # Check if community tranmission rate is included:
        if self.community_tranmission == "blue":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else :
                        new_tier = 2
            elif new_tier == 1:
                if ToIY_avg > 10:
                    new_tier = 2
        elif self.community_tranmission == "green":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else :
                        new_tier = 2
                
        if current_tier is None:  # bump to the next tier
            t_end = t + self.tiers[new_tier]['min_enforcing_time']

        elif new_tier > current_tier:
            t_end = t + self.tiers[new_tier]['min_enforcing_time']

        elif new_tier < current_tier: # relax one tier, if safety trigger allows
            IH_total = IH[-1].sum()
            assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
            new_tier = new_tier if assert_safety_trigger else current_tier
            t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
            t_end = t + t_delta

        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = t + 1

        self.tier_history += [new_tier for i in range(t_end - t)]
