'''
This module includes different vaccine allocation policies that are simulated
'''
import json
import numpy as np
from VaccineAllocation import config
from itertools import product, permutations
import copy 
import iteround
import datetime as dt
from trigger_policies import MultiTierPolicy
            
class VaccineAllocationPolicy():
    '''
        A vaccine allocation policy for different age-risk groups.

    '''
    def __init__(self, instance, vaccines, allocation, policy_name):
        # Initialize
        self._instance = instance
        self._vaccines = vaccines
        self._vaccine_groups = vaccines.define_groups()
        T, A, L = instance.T, instance.A, instance.L
        total_risk_gr = A*L
        N = instance.N

        # print(allocation)
                    
        #self._vaccine_distribution_date = np.unique(vaccines.vaccine_time)
        self._vaccine_policy_name = policy_name
        self._allocation = allocation

        # LP -- I guess this is a good double-check

        # Checking the feasibility
        print('Considering', self._vaccine_policy_name)
        total_allocation_p_a_r = np.zeros((A, L))
        total_allocation = np.zeros((A, L))
        for allocation_item in allocation['v_first']:
            total_allocation_p_a_r += allocation_item['within_proportion']
            total_allocation += allocation_item['assignment']


        print('Total allocated vaccine first dose (within-proportion):')
        print(np.round(total_allocation_p_a_r, 4))

        total_allocation_p_a_r = np.zeros((A, L))
        total_allocation = np.zeros((A, L))
        for allocation_item in allocation['v_wane']:
            total_allocation_p_a_r += allocation_item['within_proportion']
            total_allocation += allocation_item['assignment']


        print('Total allocated waneddose (within-proportion):')
        print(np.round(total_allocation_p_a_r, 4))

        total_allocation_p_a_r = np.zeros((A, L))
        total_allocation = np.zeros((A, L))
        for allocation_item in allocation['v_booster']:
            total_allocation_p_a_r += allocation_item['within_proportion']
            total_allocation += allocation_item['assignment']


        print('Total allocated booster dose (within-proportion):')
        print(np.round(total_allocation_p_a_r, 4))
   
    def reset_vaccine_history(self, instance, seed):
        '''
            reset vaccine history for a new simulation.
        '''
        for v_group in self._vaccine_groups:        
            v_group.reset_history(instance, seed)
        
        
    @classmethod
    def vaccine_policy(cls, instance, vaccines):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)
        return cls(instance, vaccines, vaccines.vaccine_allocation, 'fixed_policy')