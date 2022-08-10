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

def build_ACS_policy_candidates(instance, tiers, acs_bounds, acs_time_bounds, threshold_type='constant', lambda_start=None):
    #breakpoint()
    assert len(tiers) >= 2, 'At least two tiers need to be defined'
    threshold_candidates = []
    if threshold_type == CONSTANT_TR:
        gz = config['grid_size']
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
                params_trials.append(candidates)
        # append the acs_trigger and acs_length
        acs_trigger_candidates = np.unique([gz * i for i in range(int(acs_bounds[0] / gz), int(acs_bounds[1] / gz) + 1)] + [acs_bounds[1]])
        acs_time_candidates = np.unique([gz * i for i in range(int(acs_time_bounds[0] / gz), int(acs_time_bounds[1] / gz) + 1)] + [acs_time_bounds[1]])
        params_trials.append(acs_trigger_candidates)
        params_trials.append(acs_time_candidates)
        
        for policy_params in product(*params_trials):
            is_valid = True
            for p_ix in range(len(policy_params) - 3):
                if policy_params[p_ix] >= policy_params[p_ix + 1]:
                    is_valid = False
                    break
            if is_valid:
                T = len(instance.cal)
                lockdown_thresholds = [[policy_params[i]] * T for i in range(len(policy_params) - 2)]
                output_trials = [lockdown_thresholds, policy_params[-2], policy_params[-1]]
                threshold_candidates.append(output_trials)
       # breakpoint()
        return threshold_candidates
    else:
        raise NotImplementedError

def build_CDC_policy_thresholds(instance, tiers):
    """
    Currently there is no optimization on CDC system.
    Implement the existing system. 
    Assuming only one candidate threshold for each metric.
    """
    
    nonsurge_staffed_bed, surge_staffed_bed = {}, {}
    nonsurge_hosp_adm,surge_hosp_adm = {}, {}
    for id_t, tier in enumerate(tiers):
        nonsurge_staffed_bed[id_t] = tier["nonsurge_thresholds"]["staffed_bed"][0]
        nonsurge_hosp_adm[id_t] = tier["nonsurge_thresholds"]["hosp_adm"][0]
        
        surge_staffed_bed[id_t] = tier["surge_thresholds"]["staffed_bed"][0]
        surge_hosp_adm[id_t] = tier["surge_thresholds"]["hosp_adm"][0]

    nonsurge_thresholds = {"hosp_adm":nonsurge_hosp_adm, "staffed_bed":nonsurge_staffed_bed}
    surge_thresholds = {"hosp_adm":surge_hosp_adm, "staffed_bed":surge_staffed_bed}   
      
    nonsurge_hosp_adm_ub = [nonsurge_hosp_adm[i] for i in range(1, len(nonsurge_hosp_adm))]
    nonsurge_hosp_adm_ub.append(np.inf)
    nonsurge_staffed_bed_ub = [nonsurge_staffed_bed[i] for i in range(1, len(nonsurge_staffed_bed))]
    nonsurge_staffed_bed_ub.append(np.inf)
    surge_hosp_adm_ub =[surge_hosp_adm[i] for i in range(1, len(surge_hosp_adm))]
    surge_hosp_adm_ub.append(np.inf)
    surge_staffed_bed_ub = [surge_staffed_bed[i] for i in range(1, len(surge_staffed_bed))]
    surge_staffed_bed_ub.append(np.inf) 
    
    nonsurge_thresholds_ub = {"hosp_adm":nonsurge_hosp_adm_ub, "staffed_bed":nonsurge_staffed_bed_ub}
    surge_thresholds_ub    = {"hosp_adm":surge_hosp_adm_ub, "staffed_bed":surge_staffed_bed_ub}

    return nonsurge_thresholds, surge_thresholds, nonsurge_thresholds_ub, surge_thresholds_ub
    
class CDCTierPolicy():
    """
    CDC's community levels. CDC system includes three tiers. Green and orange
    stages are deprecated but maintained for code consitency with our system.
    """
    def __init__(self, instance, 
                 tiers, 
                 case_threshold, 
                 nonsurge_thresholds, 
                 surge_thresholds,
                 nonsurge_thresholds_ub, 
                 surge_thresholds_ub):
        """
        instance : (Instance) data instance
        tiers (list of dict): a list of the tiers characterized by a dictionary
                with the following entries:
                    {
                        "transmission_reduction": float [0,1)
                        "cocooning": float [0,1)
                        "school_closure": int {0,1}
                    } 
        case_thresholds : (Surge threshold. New COVID-19 Cases Per 100,000 people 
                          in the past 7 days
        (non)surge_thresholds : (dict of dict) with entries:               
                   { hosp_adm : (list of list) a list with the thresholds for 
                                every tier. New COVID-19 admissions per 100,000 
                                population (7-day total)
                    staffed_bed : (list of list) a list with the thresholds for 
                                every tier.Percent of staffed inpatient beds 
                                occupied by COVID-19 patients (7-day average)
                   }
    
        """
        self.tiers = tiers
        self.case_threshold = case_threshold
        self.nonsurge_thresholds = nonsurge_thresholds
        self.surge_thresholds = surge_thresholds 
        self.nonsurge_thresholds_ub = nonsurge_thresholds_ub
        self.surge_thresholds_ub = surge_thresholds_ub
        self._n = len(self.tiers)
        self._tier_history = None
        self._surge_history = None
        self._intervention_history = None
        self._instance = instance
        self.red_counter = 0
         
          
    @classmethod
    def policy(cls, instance, tiers): 
        nonsurge_thresholds, surge_thresholds, nonsurge_thresholds_ub, surge_thresholds_ub = build_CDC_policy_thresholds(instance, tiers.tier)
        case_threshold = tiers.case_threshold
        return cls(instance, tiers.tier, 
                   case_threshold, 
                   nonsurge_thresholds, 
                   surge_thresholds,
                   nonsurge_thresholds_ub, 
                   surge_thresholds_ub)
        
    def deep_copy(self):
        p = CDCTierPolicy(self._instance, 
                          self.tiers,
                          self.case_threshold, 
                          self.nonsurge_thresholds, 
                          self.surge_thresholds,
                          self.nonsurge_thresholds_ub, 
                          self.surge_thresholds_ub)
        
        p.set_tier_history(self._tier_history_copy)
        p.set_intervention_history(self._intervention_history_copy)
        p.set_surge_history(self._surge_history_copy)
        return p
    
    def set_tier_history(self, history):
        # Set history and saves a copy to reset
        self._tier_history = history.copy()
        self._tier_history_copy = history.copy()
    
    def set_intervention_history(self, history):
        # Set history and saves a copy to reset
        self._intervention_history = history.copy()
        self._intervention_history_copy = history.copy()
    
    def set_surge_history(self, history):
        '''
        Creaate surge history array
        '''
        t = len(history)
        self._surge_history = np.zeros(t)
        self._surge_history_copy = np.zeros(t)


    def reset_history(self):
        # reset history so that a new simulation can be excecuted
        self.set_tier_history(self._tier_history_copy)
        self.set_intervention_history(self._intervention_history_copy)
        self.set_surge_history(self._surge_history_copy)
        
    def compute_cost(self):
        return sum(self.tiers[i]['daily_cost'] for i in self._tier_history[783:] if i is not None and i in range(self._n))
    
    def get_tier_history(self):
        return self._tier_history
    
    def get_interventions_history(self):
        return self._intervention_history
    
    def get_surge_history(self):
        return self._surge_history
 
    def __repr__(self):
        p_str =  str([("CDC", self.case_threshold)])
        p_str = p_str.replace(' ', '')
        p_str = p_str.replace(',', '_')
        p_str = p_str.replace("'", "")
        p_str = p_str.replace('[', '')
        p_str = p_str.replace('(', '')
        p_str = p_str.replace(']', '')
        p_str = p_str.replace(')', '')
        return p_str
      
    def __call__(self, t, ToIHT, IH, ToIY, ICU, *args, **kwargs):
        '''
            Function that makes an instance of a policy a callable.
            Args:
                t (int): time period in the simulation
                ToIHT (ndarray): daily hospital admission, passed by the simulator
                IH (ndarray): hospitalizations, passed by the simulator
                ToIY (ndarray): new symptomatic cases, passed by the simulator 
                ** kwargs: additional parameters that are passed and are used elsewhere
        '''
     
        if self._tier_history[t] is not None:
            return self._intervention_history[t],kwargs

        # Compute 7-day total new cases:
        N = self._instance.N
        moving_avg_start = np.maximum(0, t - config['moving_avg_len'])
        ToIY_total = ToIY.sum((1, 2))
        ToIY_total = ToIY_total[moving_avg_start:].sum()* 100000/np.sum(N, axis=(0,1)) 
        
        # Compute 7-day total daily admission:
        ToIHT_total = ToIHT.sum((1, 2))
        ToIHT_total = ToIHT_total[moving_avg_start:].sum()* 100000/np.sum(N, axis=(0,1))
        
        # Compute 7-day average percent of COVID beds:
        IHT_total = IH.sum((1, 2)) + ICU.sum((1,2))
        IHT_avg = IHT_total[moving_avg_start:].mean()/self._instance.hosp_beds
            
            
        current_tier = self._tier_history[t - 1]
        valid_interventions_t = kwargs['feasible_interventions'][t]
        T = self._instance.T
        effective_tiers = range(len(self.tiers))

        if ToIY_total < self.case_threshold:   # Nonsurge
            hosp_adm_thresholds = self.nonsurge_thresholds["hosp_adm"]
            staffed_bed_thresholds = self.nonsurge_thresholds["staffed_bed"]
            surge_state = 0
        else:
            hosp_adm_thresholds = self.surge_thresholds["hosp_adm"]
            staffed_bed_thresholds = self.surge_thresholds["staffed_bed"]
            surge_state = 1
        
        hosp_adm_thresholds_ub, staffed_bed_thresholds_ub  = {}, {} 
        for id_t in effective_tiers:
            if id_t!= len(effective_tiers) - 1:
                hosp_adm_thresholds_ub[id_t] = hosp_adm_thresholds[id_t + 1]
                staffed_bed_thresholds_ub[id_t] = staffed_bed_thresholds[id_t + 1]
            else:
                hosp_adm_thresholds_ub[id_t] = np.inf
                staffed_bed_thresholds_ub[id_t] = np.inf  
                    
        hosp_adm_tier = effective_tiers[[
            hosp_adm_thresholds[tier_ix] <= ToIHT_total < hosp_adm_thresholds_ub[tier_ix] 
            for tier_ix in effective_tiers].index(True)]
            
        staffed_bed_tier = effective_tiers[[
            staffed_bed_thresholds[tier_ix] <= IHT_avg < staffed_bed_thresholds_ub[tier_ix] 
            for tier_ix in effective_tiers].index(True)]
            
        new_tier = max(hosp_adm_tier, staffed_bed_tier)
        if new_tier > current_tier:  # bump to the next tier
            t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
            
        elif new_tier < current_tier: # relax one tier, if safety trigger allows
            IH_total = IH[-1].sum()
            assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
            new_tier = new_tier if assert_safety_trigger else current_tier
            t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
            t_end = np.minimum(t + t_delta, T)

        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = np.minimum(t + 1, T)
        
        self._surge_history[t:t_end] = surge_state
        self._intervention_history[t:t_end] = valid_interventions_t[new_tier]
        self._tier_history[t:t_end] = new_tier
        if new_tier == 4:
            self.red_counter += (t_end - t)
        else:
            self.red_counter = 0
 
        return self._intervention_history[t],kwargs        
    
class MultiTierPolicy():
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
        self.lockdown_thresholds_ub = [lockdown_thresholds[i] for i in range(1, len(lockdown_thresholds))]
        self.lockdown_thresholds_ub.append([np.inf] * len(lockdown_thresholds[0]))
        
        self._n = len(self.tiers)
        self._tier_history = None
        self._intervention_history = None
        self._instance = instance
        self.red_counter = 0
    
    @classmethod
    def constant_policy(cls, instance, tiers, constant_thresholds, community_tranmission):
        T = instance.T
        lockdown_thresholds = [[ct] * T for ct in constant_thresholds]
        return cls(instance, tiers, lockdown_thresholds, 'constant', community_tranmission)
    
    @classmethod
    def step_policy(cls, instance, tiers, constant_thresholds, change_date, community_tranmission):
        lockdown_thresholds = []
        for tier_ix, tier in enumerate(tiers):
            tier_thres = []
            for t, d in enumerate(instance.cal.calendar):
                if d < change_date:
                    tier_thres.append(constant_thresholds[0][tier_ix])
                else:
                    tier_thres.append(constant_thresholds[1][tier_ix])
            lockdown_thresholds.append(tier_thres)
        return cls(instance, tiers, lockdown_thresholds, 'step', community_tranmission)
    
    @classmethod
    def linear_policy(cls, instance, tiers, constant_thresholds, change_date, slope, community_tranmission):
        '''
        Linear threshold funtion.
        Parameters
        ----------
        constant_thresholds : the slope of the linear function. 
        change_date : The date the threshold starts to increase.   
        slope : The slope of the linear function.
        -------
        '''
        T = len(instance.cal)
        T_slope_start = np.where(np.array(instance.cal.calendar) == change_date)[0][0]
        T_slope = T - T_slope_start
       # The lower bound for green will be constant at -1:
        lockdown_thresholds = [[constant_thresholds[0]] * T]
        # The lower bound will be increasing for other stages:
        lockdown_thresholds += [[constant_thresholds[i]] * T_slope_start + \
                                [constant_thresholds[i] + (slope) * (t + 1) for t in range(T_slope)] 
                                for i in range(1,len(constant_thresholds))]
            
        return cls(instance, tiers, lockdown_thresholds, 'linear', community_tranmission)

    def deep_copy(self):
        p = MultiTierPolicy(self._instance, self.tiers, self.lockdown_thresholds, self.tier_type, self.community_tranmission)
        p.set_tier_history(self._tier_history_copy)
        p.set_intervention_history(self._intervention_history_copy)
        return p
    
    def set_tier_history(self, history):
        # Set history and saves a copy to reset
        self._tier_history = history.copy()
        self._tier_history_copy = history.copy()
    
    def set_intervention_history(self, history):
        # Set history and saves a copy to reset
        self._intervention_history = history.copy()
        self._intervention_history_copy = history.copy()
    
    def set_surge_history(self, history):
        '''
        Redundant function.
        '''
        self._surge_history = None
        self._surge_history_copy = None
        
    def reset_history(self):
        # reset history so that a new simulation can be excecuted
        self.set_tier_history(self._tier_history_copy)
        self.set_intervention_history(self._intervention_history_copy)
    
    def compute_cost(self):
        return sum(self.tiers[i]['daily_cost'] for i in self._tier_history[783:] if i is not None and i in range(self._n))
    
    def get_tier_history(self):
        return self._tier_history
    
    def get_interventions_history(self):
        return self._intervention_history
    
    def __repr__(self):
        p_str = str([(self.tiers[i]['name'], self.lockdown_thresholds[i][0], self.lockdown_thresholds[i][-1])
                     for i in range(len(self.tiers))])
        p_str = p_str.replace(' ', '')
        p_str = p_str.replace(',', '_')
        p_str = p_str.replace("'", "")
        p_str = p_str.replace('[', '')
        p_str = p_str.replace('(', '')
        p_str = p_str.replace(']', '')
        p_str = p_str.replace(')', '')
        return p_str
    
    def __call__(self, t, ToIHT, IH, ToIY, ICU, *args, **kwargs):
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
        if self._tier_history[t] is not None:
            return self._intervention_history[t],kwargs
        
        # if t >= 324:
        #     breakpoint()
        # enforce the tiers out
        effective_threshold = {}
        effective_threshold_ub = {}
        for itier in range(len(self.tiers)):
            effective_threshold[itier] = self.lockdown_thresholds[itier][t]
            effective_threshold_ub[itier] = self.lockdown_thresholds_ub[itier][t]
        
        effective_casecounts_threshold = {}
        effective_casecounts_threshold_ub = {}
        case_thresholds = [-1, -1, -1, -1, -1]
        case_thresholds_ub = [5, np.inf, np.inf, np.inf, np.inf]
        for itier in range(len(self.tiers)):
            effective_casecounts_threshold[itier] = case_thresholds[itier]
            effective_casecounts_threshold_ub[itier] = case_thresholds_ub[itier]
            
        effective_tiers = range(len(self.tiers))   
        
        for tier_ind in range(len(effective_tiers)):
            tier_ix = effective_tiers[tier_ind]
            if tier_ind != len(effective_tiers) - 1:
                effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
            else:
                effective_threshold_ub[tier_ix] = np.inf

        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - config['moving_avg_len'])
        criStat_total = ToIHT.sum((1, 2))
        criStat_avg = criStat_total[moving_avg_start:].mean()

        # if t > 783:
        #    print(criStat_avg)
        
        # Compute new cases per 100k:
        if len(ToIY) > 0:
            ToIY_avg = ToIY.sum((1,2))[moving_avg_start:].sum()* 100000/np.sum(N, axis=(0,1))
        else:
            ToIY_avg = 0  
            
            
        current_tier = self._tier_history[t - 1]
        valid_interventions_t = kwargs['feasible_interventions'][t]
        T = self._instance.T

        new_tier = effective_tiers[[
            effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix] 
            for tier_ix in effective_tiers].index(True)]
        
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
                
        if new_tier > current_tier:  # bump to the next tier
            t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
           # breakpoint()
        elif new_tier < current_tier: # relax one tier, if safety trigger allows
            IH_total = IH[-1].sum()
            assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
            new_tier = new_tier if assert_safety_trigger else current_tier
            t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
            t_end = np.minimum(t + t_delta, T)
           # breakpoint()
        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = np.minimum(t + 1, T)
        # if t > 700:    
        #     breakpoint()    
        self._intervention_history[t:t_end] = valid_interventions_t[new_tier]
        self._tier_history[t:t_end] = new_tier
        if new_tier == 4:
            self.red_counter += (t_end - t)
        else:
            self.red_counter = 0

        #print('new tier: ', new_tier)
        #print('criStat_avg', criStat_avg)
        #breakpoint()
        return self._intervention_history[t],kwargs

class MultiTierPolicy_ACS():
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
            threshold_type: functional form of the threshold (options are in THRESHOLD_TYPES)
    '''
    def __init__(self, instance, tiers, lockdown_thresholds, acs_thrs, acs_length, acs_lead_time, acs_Q, acs_type):
        assert len(tiers) == len(lockdown_thresholds)
        self.tiers = tiers
        self.lockdown_thresholds = lockdown_thresholds
        self.lockdown_thresholds_ub = [lockdown_thresholds[i] for i in range(1, len(lockdown_thresholds))]
        self.lockdown_thresholds_ub.append([np.inf] * len(lockdown_thresholds[0]))
        
        self._n = len(self.tiers)
        self._tier_history = None
        self._intervention_history = None
        self._instance = instance
        
        self.red_counter = 0
        
        # ACS parammeters
        self.acs_thrs = acs_thrs
        self.acs_length = acs_length
        self.acs_lead_time = acs_lead_time
        self.acs_Q = acs_Q
        self.acs_type = acs_type
    
    @classmethod
    def constant_policy(cls, instance, tiers, constant_thresholds, acs_thrs, acs_length, acs_lead_time, acs_Q, acs_type):
        T = instance.T
        lockdown_thresholds = [[ct] * T for ct in constant_thresholds]
        return cls(instance, tiers, lockdown_thresholds, acs_thrs, acs_length, acs_lead_time, acs_Q. acs_type)
    
    def deep_copy(self):
        p = MultiTierPolicy_ACS(self._instance, self.tiers, self.lockdown_thresholds, self.acs_thrs, self.acs_length, self.acs_lead_time, self.acs_Q, self.acs_type)
        p.set_tier_history(self._tier_history_copy)
        p.set_intervention_history(self._intervention_history_copy)
        return p
    
    def set_tier_history(self, history):
        # Set history and saves a copy to reset
        self._tier_history = history.copy()
        self._tier_history_copy = history.copy()
    
    def set_intervention_history(self, history):
        # Set history and saves a copy to reset
        self._intervention_history = history.copy()
        self._intervention_history_copy = history.copy()
    
    def reset_history(self):
        # reset history so that a new simulation can be excecuted
        self.set_tier_history(self._tier_history_copy)
        self.set_intervention_history(self._intervention_history_copy)
    
    def compute_cost(self):
        return sum(self.tiers[i]['daily_cost'] for i in self._tier_history[783:] if i is not None and i in range(self._n))
    
    def get_tier_history(self):
        return self._tier_history
    
    def get_cap_history(self):
        return self._capacity
    
    def get_interventions_history(self):
        return self._intervention_history
    
    def __repr__(self):
        p_str = str([(self.tiers[i]['name'], self.lockdown_thresholds[i][0], self.lockdown_thresholds[i][-1])
                     for i in range(len(self.tiers))])
        p_str = p_str.replace(' ', '')
        p_str = p_str.replace(',', '_')
        p_str = p_str.replace("'", "")
        p_str = p_str.replace('[', '')
        p_str = p_str.replace('(', '')
        p_str = p_str.replace(']', '')
        p_str = p_str.replace(')', '')
        return p_str

    def __call__(self, t, N, criStat, IH, ToIY, *args, **kwargs):
        '''
            Function that makes an instance of a policy a callable.
            Args:
                t (int): time period in the simulation
                z (object): deprecated, but maintained to avoid changes in the simulate function
                criStat (ndarray): the trigger statistics, previously daily admission, passed by the simulator
                IH (ndarray): hospitalizations admissions, passed by the simulator
                ** kwargs: additional parameters that are passed and are used elsewhere
        '''
        # Compute daily admissions moving average
       # breakpoint()
        moving_avg_start = np.maximum(0, t - config['moving_avg_len'])
        if len(kwargs["acs_criStat"]) > 0:
            acs_criStat_avg = kwargs["acs_criStat"].sum((1,2))[moving_avg_start:].mean()
        else:
            acs_criStat_avg = 0
        

            
        # check hospitalization trigger
        if t >= kwargs["t_start"]:
            if (not kwargs["acs_triggered"]) and (acs_criStat_avg > self.acs_thrs):
                kwargs["acs_triggered"] = True
                for tCap in range(t + self.acs_lead_time,t + self.acs_lead_time + self.acs_length):
                    if tCap < len(kwargs["_capacity"]):
                        kwargs["_capacity"][tCap] = kwargs["_capacity"][tCap] + self.acs_Q
        
        if self._tier_history[t] is not None:
            return self._intervention_history[t],kwargs
                
        # enforce the tiers out
        criStat_total = criStat.sum((1, 2))
        criStat_avg = criStat_total[moving_avg_start:].mean()

        effective_threshold = {}
        effective_threshold_ub = {}
        for itier in range(len(self.tiers)):
            effective_threshold[itier] = self.lockdown_thresholds[itier][0]
            effective_threshold_ub[itier] = self.lockdown_thresholds_ub[itier][0]
        if kwargs['fo_tiers'] is None:
            effective_tiers = range(len(self.tiers))
        else:
            if not(kwargs['changed_tiers']):
                effective_tiers = [i for i in range(len(self.tiers)) if i not in kwargs['fo_tiers']]
            else:
                effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers']]
        for tier_ind in range(len(effective_tiers)):
            tier_ix = effective_tiers[tier_ind]
            if tier_ind != len(effective_tiers) - 1:
                effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
            else:
                effective_threshold_ub[tier_ix] = np.inf
        
        current_tier = self._tier_history[t - 1]
        valid_interventions_t = kwargs['feasible_interventions'][t]
        T = self._instance.T
        
        ## NEW
        new_tier = effective_tiers[[
            effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
            for tier_ix in effective_tiers
        ].index(True)]
        
        if new_tier == current_tier:
            t_end = np.minimum(t + 1, T)
            # if it is the first time turning red
            if new_tier == 4:
                # forced out of the red tier
                if self.red_counter >= kwargs["redLimit"]:
                    # if it is the first time turning red
                    if (not(kwargs['changed_tiers'])):
                        kwargs['changed_tiers'] = True
                    effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers'] and i != 4]
                    for tier_ind in range(len(effective_tiers)):
                        tier_ix = effective_tiers[tier_ind]
                        if tier_ind != len(effective_tiers) - 1:
                            effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
                        else:
                            effective_threshold_ub[tier_ix] = np.inf
                    new_tier = effective_tiers[[
                        effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
                        for tier_ix in effective_tiers
                    ].index(True)]
                    t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        else:
            if new_tier < current_tier:
                IH_total = IH[-1].sum()
                if current_tier != 4:
                    assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
                    new_tier = new_tier if assert_safety_trigger else current_tier
                    t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
                else:
                    # if it is the first time turning red
                    if (not(kwargs['changed_tiers'])):
                        kwargs['changed_tiers'] = True
                        effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers']]
                        for tier_ind in range(len(effective_tiers)):
                            tier_ix = effective_tiers[tier_ind]
                            if tier_ind != len(effective_tiers) - 1:
                                effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
                            else:
                                effective_threshold_ub[tier_ix] = np.inf
                        new_tier = effective_tiers[[
                            effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
                            for tier_ix in effective_tiers
                        ].index(True)]
                    t_delta = self.tiers[new_tier]['min_enforcing_time']
                t_end = np.minimum(t + t_delta, T)
            else:
                t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        self._intervention_history[t:t_end] = valid_interventions_t[new_tier]
        self._tier_history[t:t_end] = new_tier
        if new_tier == 4:
            self.red_counter += (t_end - t)
        else:
            self.red_counter = 0
        
        return self._intervention_history[t],kwargs