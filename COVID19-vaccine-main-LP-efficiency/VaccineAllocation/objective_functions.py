import pickle
import numpy as np
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
from itertools import product
from trigger_policies import  build_multi_tier_policy_candidates, MultiTierPolicy
#from reporting.plotting import plot_stoch_simulations
from VaccineAllocation import config, logger, output_path
from utils import profile_log, print_profiling_log


def multi_tier_objective(instance, policy, sim_output, **kwargs):
    epi = instance.epi
    IYIH = np.sum(sim_output['IYIH'], axis=(-1, -2))
    IHT = np.sum(sim_output['IHT'], axis=(-1, -2))
    lambda_star_scaled = epi.eq_mu * instance.lambda_star
    diff_IYIH_above = np.sum(np.maximum(0, IYIH - lambda_star_scaled))
    exceeding_threshold = np.sum(np.maximum(IHT - instance.hosp_beds,0))
    inf_penalty = np.inf if diff_IYIH_above > 0 else 0
    policy_cost = policy.compute_cost()
    cap_cost = kwargs['over_capacity_cost']*exceeding_threshold
    if kwargs['icu_trigger']:
        ICU = np.sum(sim_output['ICU'], axis=(-1, -2))
        icu_exceeding_threshold = np.sum(np.maximum(ICU - instance.icu,0))
        cap_cost += kwargs['icu_capacity_cost']*icu_exceeding_threshold
    if kwargs['obj_over_included']:
        if kwargs['infeasible_penalty']:
            return policy_cost + cap_cost + inf_penalty, [policy_cost, cap_cost, inf_penalty]
        else:
            return policy_cost + cap_cost, [policy_cost, cap_cost, inf_penalty]
    else:
        if kwargs['infeasible_penalty']:
            return policy_cost + inf_penalty, [policy_cost, cap_cost, inf_penalty]
        else:
            return policy_cost, [policy_cost, cap_cost, inf_penalty]