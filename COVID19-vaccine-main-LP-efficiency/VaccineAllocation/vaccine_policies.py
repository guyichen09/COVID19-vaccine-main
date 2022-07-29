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

def find_allocation(vaccines, calendar, A, L, T, N, allocation):

    # LP Edit
    # flag (either False or True) corresponds to whether or not
    #   allocation_item["supply"]["time"] intersects with the simulation horizon
    # This variable is added to accommodate the cases in which we run
    #   the model for a short simulation horizon that terminates before
    #   vaccines begin.

    for allocation_type in allocation:
        for allocation_item in allocation[allocation_type]:

            flag = False
            id_t = np.where(calendar == allocation_item['supply']['time'])[0]

            vaccine_a = []
            for i in range(A):
                for j in range(L):
                    vaccine_a.append([0] * (T))

            k = 0
            for i in range(A):
                for j in range(L):

                    if flag == False:
                        # the simulation calendar does intersect with a vaccine event
                        # therefore, switch the flag to True and proceed as usual
                        if len(id_t) > 0:
                            flag = True
                            id_t = id_t[0]
                            vaccine_a[k][id_t] = allocation_item['assignment'][i, j]
                        # the simulation calendar does not intersect with a vaccine event
                        # id_t does not exist, so we cannot use it as an index
                    else:
                        vaccine_a[k][id_t] = allocation_item['assignment'][i, j]

                    k += 1
            allocation_item['assignment_daily'] = np.array(vaccine_a)
    
    return allocation
            
class VaccineAllocationPolicy():
    '''
        A vaccine allocation policy for different age-risk groups.

    '''
    def __init__(self, instance, vaccines, allocation, problem_type, policy_name):
        # Initialize
        self._instance = instance
        self._vaccines = vaccines
        self._vaccine_groups = vaccines.define_groups(problem_type)
        T, A, L = instance.T, instance.A, instance.L
        total_risk_gr = A*L
        N = instance.N

        # print(allocation)
                    
        #self._vaccine_distribution_date = np.unique(vaccines.vaccine_time)
        self._vaccine_policy_name = policy_name
        self._allocation = allocation

        # # Checking the feasibility
        # print('Considering', self._vaccine_policy_name)
        # total_allocation_p_a_r = np.zeros((A, L))
        # total_allocation = np.zeros((A, L))
        # for allocation_item in allocation['v_first']:
        #     total_allocation_p_a_r += allocation_item['within_proportion']
        #     total_allocation += allocation_item['assignment']
        #
        #
        # print('Total allocated vaccine first dose (within-proportion):')
        # print(np.round(total_allocation_p_a_r, 4))
        #
        # total_allocation_p_a_r = np.zeros((A, L))
        # total_allocation = np.zeros((A, L))
        # for allocation_item in allocation['v_wane']:
        #     total_allocation_p_a_r += allocation_item['within_proportion']
        #     total_allocation += allocation_item['assignment']
        #
        #
        # print('Total allocated waneddose (within-proportion):')
        # print(np.round(total_allocation_p_a_r, 4))
        #
        # total_allocation_p_a_r = np.zeros((A, L))
        # total_allocation = np.zeros((A, L))
        # for allocation_item in allocation['v_booster']:
        #     total_allocation_p_a_r += allocation_item['within_proportion']
        #     total_allocation += allocation_item['assignment']
        #
        #
        # print('Total allocated booster dose (within-proportion):')
        # print(np.round(total_allocation_p_a_r, 4))
        
        # Assigning from-to vaccine groups for the SEIR model
        for v_group in self._vaccine_groups:
            v_group.vaccine_flow(instance, allocation)
        
   
    def reset_vaccine_history(self, instance, seed):
        '''
            reset vaccine history for a new simulation.
        '''
        for v_group in self._vaccine_groups:        
            v_group.reset_history(instance, seed)
        
        
    @classmethod
    def vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)
        allocation = vaccines.vaccine_allocation
        #breakpoint()
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        #breakpoint()
        return cls(instance, vaccines, allocation, problem_type, 'fixed_policy')
    
    @classmethod
    def vaccine_rollout_policy(cls, instance, vaccines, temp_assignment, args):
        vaccine_time = args['vaccine_time']
        active_age, active_risk = args['active_group']
        prev_age, prev_risk = args['previous_group']
        problem_type = args['type']
        eligible_population = args['eligible_population'].copy()
        #print('eligible_population', eligible_population)
        #breakpoint()
        T, A, L, N = instance.T, instance.A, instance.L, instance.N
        
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)
        
        allocation = temp_assignment.copy()       
        for id_t, supply in enumerate(vaccines.vaccine_supply):
            total_daily_supply = sum(s_item['amount'] * N_total for s_item in supply)
            allocation_list = []
            
            for id_s,s_item in enumerate(supply):
                  # breakpoint()
                vac_amount = s_item['amount'] * N_total
                vac_remaining = s_item['amount'] * N_total
                vac_assignment = np.zeros((A, L))
                            
                if s_item['time'] == vaccine_time:
                        #breakpoint()
                        # if there are people left from the previous group finish them.
                    if eligible_population[prev_age][prev_risk] > 0 and eligible_population[prev_age][prev_risk] < total_daily_supply:
      
                        vac_assignment[prev_age][prev_risk] = min(eligible_population[prev_age][prev_risk], vac_remaining)
                        vac_remaining -= vac_assignment[prev_age][prev_risk]
                        eligible_population[prev_age][prev_risk] -= vac_assignment[prev_age][prev_risk]
                    
                    
                        # Assign the remaining vaccine to active group:
                    
                    vac_assignment[active_age][active_risk] = min(eligible_population[active_age][active_risk], vac_remaining)
                    vac_remaining -= vac_assignment[active_age][active_risk]
                    eligible_population[active_age][active_risk] -= vac_assignment[active_age][active_risk]
                        # breakpoint()
                if vac_amount == 0:
                    vac_noround = np.zeros((5, 2))
                else:
                    vac_noround = vac_assignment/vac_amount
                pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                vac_assignment = pro_round*vac_amount

                allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                allocation_list.append(allocation_item)
                
                if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
                    #print('assignment', vac_assignment)
            if allocation_list != []:
                allocation.append(allocation_list) 
                    
        
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)  
        return cls(instance, vaccines, allocation, problem_type, 'vaccine_rollout_policy')       
   
    @classmethod
    def phase_1b_policy(cls, instance, vaccines, problem_type, percentage):
        
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)
        
        uptake = 0.97 # uptake level for the vaccine.

        spilage = 0.5
        
        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
        
        # Vaccine is not suitable for 16 and younger
        eligible_population[0][0] = 0
        eligible_population[0][1] = 0
        eligible_population[1][0] = 0
        eligible_population[1][1] = 0
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    
                    vac_amount = s_item['amount'] * N_total
                    vac_remaining = s_item['amount'] * N_total
                    vac_assignment = np.zeros((A, L))
                    
                    # Allocate vaccine to 65+
                    senior_total = eligible_population[4][0] + eligible_population[4][1] # Total 65+
                    med_cond_total = eligible_population[2][1] + eligible_population[3][1] # Total under 65 with medical conditions.
                    if med_cond_total > 0:
                        senior_allocation = vac_remaining*percentage
                    else:  
                        senior_allocation = vac_remaining
                    
                    if senior_total != 0:
                        vac_assignment[4][1] = (eligible_population[4][1]/senior_total)*senior_allocation
                    
                    if vac_assignment[4][1] > eligible_population[4][1]:
                        vac_assignment[4][1] = eligible_population[4][1]  
                        
                    #print( vac_assignment[4][1])
                    eligible_population[4][1] -= vac_assignment[4][1]
                    vac_remaining -= vac_assignment[4][1]
                    senior_allocation -= vac_assignment[4][1]
                        
                    vac_assignment[4][0] = senior_allocation
                    if vac_assignment[4][0] > eligible_population[4][0]:
                        vac_assignment[4][0] = eligible_population[4][0] 
                    
                    vac_remaining -= vac_assignment[4][0]
                    eligible_population[4][0] -= vac_assignment[4][0]
                    
                    
                    # Allocate to under 65, high risk:
                    
                    med_cond_allocation = vac_remaining*(1 - spilage)
                    if med_cond_total != 0:
                        vac_assignment[3][1] = (eligible_population[3][1]/med_cond_total)*med_cond_allocation
                    
                    if vac_assignment[3][1] > eligible_population[3][1]:
                        vac_assignment[3][1] = eligible_population[3][1] 
                    
                    med_cond_allocation -= vac_assignment[3][1]
                    vac_remaining -= vac_assignment[3][1]
                    eligible_population[3][1] -= vac_assignment[3][1]
                    
                    vac_assignment[2][1] = med_cond_allocation
                    if vac_assignment[2][1] > eligible_population[2][1]:
                        vac_assignment[2][1] = eligible_population[2][1] 
                        
                    vac_remaining -= vac_assignment[2][1]
                    eligible_population[2][1] -= vac_assignment[2][1]
                    
                    spilage_population = eligible_population[2][0] + eligible_population[3][0]
                    # Allocate the spilage pro-rata:
                    if vac_remaining > 0:       
                        for age in range(2,4):
                            for risk in range(0,1):
                                if spilage_population != 0:
                                    vac_assignment[age, risk] = vac_remaining*(eligible_population[age, risk]/spilage_population)
                                    if vac_assignment[age, risk] > eligible_population[age, risk]:
                                        vac_assignment[age, risk] = eligible_population[age, risk] 
                                
                        for age in range(2,4):
                            for risk in range(0,1):
                                vac_remaining -= vac_assignment[age, risk]
                                eligible_population[age, risk] -= vac_assignment[age, risk]
                                
                    if np.sum(vac_assignment) > 0:
                        vac_noround = vac_assignment/np.sum(vac_assignment)
                    else:
                        vac_noround = np.zeros((5, 2))           
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
                if allocation_list != []:
                    allocation.append(allocation_list)   
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, 'deterministic', 'phase_1b_policy')

     
    @classmethod
    def no_vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)
  
        allocation, eligible_population  = fix_hist_allocation(vaccines, instance)
   
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    vac_amount = s_item['amount'] * N_total
                    vac_assignment = np.zeros((A, L))
                    if np.sum(vac_assignment) > 0:
                        vac_noround = vac_assignment/np.sum(vac_assignment)
                    else:
                        vac_noround = np.zeros((5, 2)) 
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
            if s_item['time'] >= vaccines.vaccine_start_time:
                    allocation.append(allocation_list)

        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, problem_type, 'no_vaccine')    
 
    
    @classmethod
    def min_death_vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)

        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
            
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    vac_amount = s_item['amount'] * N_total
                    vac_remaining = s_item['amount'] * N_total
    
                    vac_assignment = np.zeros((A, L))
                    
                    # assign high risk 65+, 50-64, 18-49 in order first
                    for age in range(4, 1, -1):
                        risk = 1
                        assigned_r_a = eligible_population[age, risk]
                        if assigned_r_a > vac_remaining:
                            assigned_r_a = vac_remaining
                        vac_remaining -= assigned_r_a
                        vac_assignment[age, risk] = assigned_r_a
                        eligible_population[age, risk] -= assigned_r_a
                                
                    # distribute 5-17 for both high and low risk proportional to population, age = 1
                    age = 1 
                    N_age_5_17 = N[age, 0] + N[age, 1]
                    age_5_17_0 = (N[age, 0] * vac_remaining)/N_age_5_17
                    age_5_17_1 = (N[age, 1] * vac_remaining)/N_age_5_17
                    
                    # high-risk
                    if age_5_17_1 > eligible_population[age, 1]:
                        assigned_r_a = eligible_population[age, 1]
                        age_5_17_0 += age_5_17_1 - eligible_population[age, 1]
                    else:
                        assigned_r_a = age_5_17_1
                    vac_remaining -= assigned_r_a
                    vac_assignment[age, 1] = assigned_r_a   
                    eligible_population[age, 1] -= assigned_r_a
                    
                    # low-risk
                    if age_5_17_0 > eligible_population[age, 0]:
                        assigned_r_a = eligible_population[age, 0]
                    else:
                        assigned_r_a = age_5_17_0 
                    vac_remaining -= assigned_r_a
                    vac_assignment[age, 0] = assigned_r_a
                    eligible_population[age, 0] -= assigned_r_a
                    
                    # distribute 18-49 for low risk
                    age = 2
                    if vac_remaining > eligible_population[age, 0]:
                        assigned_r_a = eligible_population[age, 0]
                    else:
                        assigned_r_a = vac_remaining
                    vac_remaining -= assigned_r_a
                    vac_assignment[age, 0] = assigned_r_a
                    eligible_population[age, 0] -= assigned_r_a
                    
                    # remaining populations: 65+, 50-64 low risk; 0-4 high and low
                    if vac_remaining > 0:
                        N_remaining = N[4, 0] + N[3, 0] + N[0, 0] + N[0, 1]
                        age_65_0 = (N[4, 0] * vac_remaining)/N_remaining
                        vac_assignment[4, 0] = age_65_0
                        eligible_population[4, 0] -= age_65_0
                        
                        age_50_64_0 = (N[3, 0] * vac_remaining)/N_remaining
                        vac_assignment[3, 0] = age_50_64_0 
                        eligible_population[3, 0] -= age_50_64_0
                        
                        age_0_4_0 = (N[0, 0] * vac_remaining)/N_remaining
                        vac_assignment[0, 0] = age_0_4_0 
                        eligible_population[0, 0] -= age_0_4_0
                        
                        age_0_4_1 = (N[0, 1] * vac_remaining)/N_remaining
                        vac_assignment[0, 1] = age_0_4_1 
                        eligible_population[0, 1] -= age_0_4_1
                    
                    if vac_amount == 0:
                        vac_noround = np.zeros((5, 2))
                    else:
                        vac_noround = vac_assignment/vac_amount
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item) 
                        
            if allocation_list != []:
                    allocation.append(allocation_list)
 
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, problem_type, 'min_death')
    
    
    @classmethod
    def high_risk_senior_first_vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)

        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    vac_amount = s_item['amount'] * N_total
                    vac_remaining = s_item['amount'] * N_total
                    vac_assignment = np.zeros((A, L))
                    for age in range(A - 1, -1, -1):
                        for risk in range(L - 1, -1, -1):
                            assigned_r_a = eligible_population[age, risk]
                            if assigned_r_a > vac_remaining:
                                assigned_r_a = vac_remaining
                            vac_remaining -= assigned_r_a
                            vac_assignment[age, risk] = assigned_r_a
                            eligible_population[age, risk] -= assigned_r_a
                            
                    if vac_amount == 0:
                        vac_noround = np.zeros((5, 2))
                    else:
                        vac_noround = vac_assignment/vac_amount
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item2 = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}    
                        allocation_list.append(allocation_item2)
                             
            if allocation_list != []:
                allocation.append(allocation_list)
            
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, problem_type, 'high_risk_senior_first')
    
    @classmethod
    def proportional_to_pop(cls, instance, vaccines):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)

        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    vac_amount = s_item['amount'] * N_total
                    vac_remaining = s_item['amount'] * N_total
                    vac_assignment = np.zeros((A, L))
                    for age in range(A):
                        for risk in range(L):
                            vac_assignment[age, risk] = vac_remaining*(eligible_population[age, risk]/np.sum(eligible_population))
                    for age in range(A):
                        for risk in range(L):
                            vac_remaining -= vac_assignment[age, risk]
                            eligible_population[age, risk] -= vac_assignment[age, risk]
                    if vac_amount == 0:
                        vac_noround = np.zeros((5, 2))
                    else:
                        vac_noround = vac_assignment/vac_amount
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
            if allocation_list != []:
                allocation.append(allocation_list)
            
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, 'proportional_to_pop')
    
    @classmethod
    def low_risk_young_first_vaccine_policy(cls, instance, vaccines):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)

        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    vac_amount = s_item['amount'] * N_total
                    vac_remaining = s_item['amount'] * N_total
                    vac_assignment = np.zeros((A, L))
                    for age in range(A):
                        for risk in range(L):
                            assigned_r_a = eligible_population[age, risk]
                            if assigned_r_a > vac_remaining:
                                assigned_r_a = vac_remaining
                            vac_remaining -= assigned_r_a
                            vac_assignment[age, risk] = assigned_r_a
                            eligible_population[age, risk] -= assigned_r_a
                    if vac_amount == 0:
                        vac_noround = np.zeros((5, 2))
                    else:
                        vac_noround = vac_assignment/vac_amount
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
            if allocation_list != []:
                allocation.append(allocation_list)
            
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, 'low_risk_young_first')

            
    @classmethod
    def senior_first_vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)

        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    vac_amount = s_item['amount'] * N_total
                    vac_remaining = s_item['amount'] * N_total
                    vac_assignment = np.zeros((A, L))
                    for age in range(A - 1, -1, -1):
                            N_age = N[age, 0] + N[age, 1]
                            age_0 = (N[age, 0] * vac_remaining)/N_age
                            age_1 = (N[age, 1] * vac_remaining)/N_age
                            if age_1 > eligible_population[age, 1]:
                                assigned_r_a = eligible_population[age, 1]
                                vac_remaining -= assigned_r_a
                                vac_assignment[age, 1] = assigned_r_a
                                age_0 += age_1 - eligible_population[age, 1]
                                eligible_population[age, 1] = 0
                            else:
                                vac_remaining -= age_1
                                vac_assignment[age, 1] = age_1
                                eligible_population[age, 1] -= age_1
                            if age_0 > eligible_population[age, 0]:
                                assigned_r_a = eligible_population[age, 0]
                                vac_remaining -= assigned_r_a
                                vac_assignment[age, 0] = assigned_r_a
                                eligible_population[age, 0] = 0
                            else:
                                vac_remaining -= age_0
                                vac_assignment[age, 0] = age_0
                                eligible_population[age, 0] -= age_0
                    if vac_amount == 0:
                        vac_noround = np.zeros((5, 2))
                    else:
                        vac_noround = vac_assignment/vac_amount
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
            if allocation_list != []:
                allocation.append(allocation_list)
            
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, problem_type, 'senior_first')



    @classmethod
    def sort_contact_matrix_vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)

        age_array = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        risk_array = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        demographics_normalized = N / N_total
        phi_weekday = instance.epi.effective_phi(0, 0.67947125, 0.5131765, demographics_normalized, day_type=1)
        #phi = (np.sum(phi_weekday, (2, 3)) * instance.N).reshape(10)
        #print('phi weekday', phi_weekday)
        phi = (np.sum(phi_weekday, (2, 3))).reshape(10)
        #print('contact matrix', phi)
        sorted_ids = np.argsort(phi)[::-1]

        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                vac_amount = s_item['amount'] * N_total
                vac_remaining = s_item['amount'] * N_total
                vac_assignment = np.zeros((A, L))
                for idx in sorted_ids:
                        age = age_array[idx]
                        risk = risk_array[idx]
                        
                        if vac_remaining > eligible_population[age, risk]:
                            vac_assignment[age, risk] = eligible_population[age, risk]
                            vac_remaining -= vac_assignment[age, risk]
                            eligible_population[age, risk] = 0
                        else:
                            vac_assignment[age, risk] = vac_remaining
                            eligible_population[age, risk] -= vac_remaining
                            vac_remaining = 0
                if vac_amount > 0:                   
                    vac_noround = vac_assignment/vac_amount
                else:
                    vac_noround = np.zeros((5, 2))     
                pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                vac_assignment = pro_round*vac_amount
                allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                allocation_list.append(allocation_item)
                
                if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
            if allocation_list != []:
                allocation.append(allocation_list)
            
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, problem_type, 'sort_contact_matrix')
          
    @classmethod
    def min_infected_vaccine_policy(cls, instance, vaccines, problem_type):
        T, A, L = instance.T, instance.A, instance.L
        N = instance.N
        N_total = np.sum(N)
        calendar = np.array(instance.cal.calendar)
        allocation, eligible_population = fix_hist_allocation(vaccines, instance)
      
        
        #allocation, eligible_population = fix_hist_allocation(vaccines, instance)
        for supply in vaccines.vaccine_supply:
            allocation_list = []
            for s_item in supply:
                if s_item['time'] >= vaccines.vaccine_start_time:
                    vac_amount = s_item['amount'] * N_total
                    vac_remaining = s_item['amount'] * N_total
                    vac_assignment = np.zeros((A, L))
                    # assign high+low risk 5-17, 18-49, 50-64 in order first
                    for age in range(1, 4, 1):
                        N_age = N[age, 0] + N[age, 1]
                        age_0 = (N[age, 0] * vac_remaining)/N_age
                        age_1 = (N[age, 1] * vac_remaining)/N_age
                        if age_1 > eligible_population[age, 1]:
                            assigned_r_a = eligible_population[age, 1]
                            age_0 += age_1 - eligible_population[age, 1]
                        else:
                            assigned_r_a = age_1
                        vac_remaining -= assigned_r_a
                        vac_assignment[age, 1] = assigned_r_a   
                        eligible_population[age, 1] -= assigned_r_a
                            
                        if age_0 > eligible_population[age, 0]:
                            assigned_r_a = eligible_population[age, 0]
                        else:
                            assigned_r_a = age_0
                        vac_remaining -= assigned_r_a
                        vac_assignment[age, 0] = assigned_r_a   
                        eligible_population[age, 0] -= assigned_r_a
                        # assign remanining
                        # remaining populations: 65+, 0-4 high and low
                    if vac_remaining > 0:
                        N_remaining = N[4, 0] + N[4, 1] + N[0, 0] + N[0, 1]
                        age_65_0 = (N[4, 0] * vac_remaining)/N_remaining
                        vac_assignment[4, 0] = age_65_0
                        eligible_population[4, 0] -= age_65_0
                        age_65_1 = (N[4, 1] * vac_remaining)/N_remaining
                        vac_assignment[4, 1] = age_65_1 
                        eligible_population[4, 1] -= age_65_1
                        age_0_4_0 = (N[0, 0] * vac_remaining)/N_remaining
                        vac_assignment[0, 0] = age_0_4_0 
                        eligible_population[0, 0] -= age_0_4_0
                        age_0_4_1 = (N[0, 1] * vac_remaining)/N_remaining
                        vac_assignment[0, 1] = age_0_4_1 
                        eligible_population[0, 1] -= age_0_4_1
    
                    if vac_amount == 0:
                        vac_noround = np.zeros((5, 2))
                    else:
                        vac_noround = vac_assignment/vac_amount
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                    
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                        
            if allocation_list != []:
                allocation.append(allocation_list)
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)
        return cls(instance, vaccines, allocation, problem_type, 'min_infected')        

        
    def vaccine_group_update(self, new_vaccine_groups):
        self._vaccine_groups = new_vaccine_groups.copy()
        
        p = MultiTierPolicy(self._instance, self.tiers, self.lockdown_thresholds)
        p.set_tier_history(self._tier_history_copy)
        p.set_intervention_history(self._intervention_history_copy)
        return p
    
def fix_hist_allocation(vaccines, instance):
    A, L, N = instance.A, instance.L, instance.N
    uptake = 1
    eligible_population = N.copy()*uptake
    eligible_population[0][0] = 0
    eligible_population[0][1] = 0
    eligible_population[1][0] = 0
    eligible_population[1][1] = 0
        
    allocation = []
    # Fixed the historical allocations.
    if vaccines.vaccine_allocation_file_name is not None: 
        for allocation_daily in vaccines.fixed_vaccine_allocation:
            for allocation_item in allocation_daily:
                vac_assignment = allocation_item['assignment']
                allocation_item['within_proportion'] = vac_assignment/N
                if allocation_item['which_dose'] == 1:
                    for age in range(A):
                        for risk in range(L):
                            eligible_population[age][risk] -= vac_assignment[age][risk]
                                       
        allocation = vaccines.fixed_vaccine_allocation.copy() 
    else:
        allocation = []
        
    #breakpoint()    
    return allocation, eligible_population

def find_rollout_allocation(instance, vaccines, v_time_increment, allocation, candidate_group,  args):
        # Assign vaccine to the active group untill the group end time or the whole group is vaccinated.    

        vaccine_time = args['vaccine_time']
        calendar = np.array(instance.cal.calendar)
        group_end_time = vaccine_time + dt.timedelta(days = v_time_increment)
        active_age, active_risk = candidate_group
        prev_age, prev_risk = args['previous_group']
        T, A, L, N = instance.T, instance.A, instance.L, instance.N

        eligible_population = args['eligible_population']
        #print('eligible_population', eligible_population)
        
        N_total = np.sum(N)
        for id_t, supply in enumerate(vaccines.vaccine_supply):
            total_daily_supply = sum(s_item['amount'] * N_total for s_item in supply)
            allocation_list = []
            allocation_pro_list = []
           
            for id_s,s_item in enumerate(supply):
                #print('inside policy time',s_item['time'])
                vac_amount = s_item['amount'] * N_total
                vac_remaining = s_item['amount'] * N_total
                vac_assignment = np.zeros((A, L))
                            
                if (vaccine_time <= s_item['time']) and (s_item['time'] < group_end_time):
                    #print('eligible population', eligible_population)                        
                    # if there are people left from the previous group finish them.
                    if eligible_population[prev_age][prev_risk] > 0 and eligible_population[prev_age][prev_risk] < total_daily_supply:
                        vac_assignment[prev_age][prev_risk] = min(eligible_population[prev_age][prev_risk], vac_remaining)
                        vac_remaining -= vac_assignment[prev_age][prev_risk]
                        eligible_population[prev_age][prev_risk] -= vac_assignment[prev_age][prev_risk]
                        total_daily_supply -= vac_assignment[prev_age][prev_risk]

                    # Assign the remaining vaccine to the active group:
                    if eligible_population[active_age][active_risk] < vac_remaining:
                        # The active group is about to be finished. finish the vaccination earlier:
                        group_end_time = s_item['time']
                    else:
                        vac_assignment[active_age][active_risk] = min(eligible_population[active_age][active_risk], vac_remaining)
                        vac_remaining -= vac_assignment[active_age][active_risk]
                        eligible_population[active_age][active_risk] -= vac_assignment[active_age][active_risk]
                    
                    if vac_amount == 0:
                        vac_noround = np.zeros((5, 2))
                    else:
                        vac_noround = vac_assignment/vac_amount
                    pro_round = np.reshape(iteround.saferound(vac_noround.reshape(10), 10), (5, 2))
                    vac_assignment = pro_round*vac_amount
                
                    allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item, 'which_dose': 1, 'within_proportion': vac_assignment/N}    
                    allocation_list.append(allocation_item)
                
                    if s_item['dose_type'] == 2:
                        allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'supply': s_item['second_dose_supply'], 'which_dose': 2, 'within_proportion': vac_assignment/N}  
                        allocation_list.append(allocation_item)
                    
            if allocation_list != []:
                allocation.append(allocation_list) 
                    
        allocation = find_allocation(vaccines, calendar, A, L, T, N, allocation)  
        return allocation,  group_end_time, eligible_population
        