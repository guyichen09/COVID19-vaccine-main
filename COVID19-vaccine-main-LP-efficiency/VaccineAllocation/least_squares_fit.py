import numpy as np
import pandas as pd
import datetime as dt
from pipelinemultitier import read_hosp
from SEIYAHRD import simulate_p
from VaccineAllocation import config, logger
from objective_functions import multi_tier_objective
from trigger_policies import MultiTierPolicy as MTP
from scipy.optimize import least_squares
import copy

def deterministic_path(instance,
                       tiers,
                       vaccines,
                       obj_func,
                       n_replicas_train=100,
                       n_replicas_test=100,
                       instance_name=None,
                       policy_class='constant',
                       policy=None,
                       vaccine_policy=None,
                       mp_pool=None,
                       crn_seeds=[],
                       unique_seeds_ori=[],
                       forcedOut_tiers=None,
                       redLimit=1000,
                       after_tiers=[0,1,2,3,4],
                       policy_field="ToIHT",
                       policy_ub=None):
  
    fixed_TR = list(filter(None, instance.cal.fixed_transmission_reduction))
    tier_TR = [item['transmission_reduction'] for item in tiers]
    uniquePS = sorted(np.unique(np.append(fixed_TR, np.unique(tier_TR))))
    sc_levels = np.unique([tier['school_closure'] for tier in tiers] + [0, 1])
    fixed_CO = list(filter(None, instance.cal.fixed_cocooning))
    tier_CO = np.unique([tier['cocooning'] for tier in tiers])
    uniqueCO = sorted(np.unique(np.append(fixed_CO, np.unique(tier_CO))))
    intervention_levels = create_intLevel(sc_levels, uniqueCO, uniquePS)
    interventions_train = form_interventions(intervention_levels, instance.epi, instance.N)
    
    config['det_history'] = True
    # Build an iterator of all the candidates to be simulated by simulate_p
    sim_configs = policy_multi_iterator(instance,
                                        tiers,
                                        vaccines, 
                                        obj_func,
                                        interventions_train,
                                        policy_class=policy_class,
                                        fixed_policy=policy,
                                        fixed_vaccine_policy=vaccine_policy,
                                        policy_field=policy_field,
                                        policy_ub=policy_ub)
    
    all_outputs = simulate_p(mp_pool, sim_configs)
    sim_output, cost, _trigger_policy, _vac_policy, seed_0, kwargs_out = all_outputs[0]
        
    return sim_output

def residual_error(x_beta, **kwargs):
    change_dates = kwargs['change_dates']
    instance = kwargs['instance']
    vaccines = kwargs['vaccines']
    tiers = kwargs['tiers']
    hosp_ad = kwargs['hosp_ad']
    real_icu = kwargs['real_icu']
    real_death_from_hosp = kwargs['real_death_from_hosp']
    real_death_total = kwargs['real_death_total']
    selected_vaccine_policy = kwargs['selected_vaccine_policy']
    t_start = kwargs['t_start']
    #############Change the transmission reduction and cocconing accordingly
    beta = [ 0.052257,
             0.787752,
             0.641986,
             0.827015,
             0.778334,
             0.752980,
             0.674321,
             0.801538,
             0.811144,
             0.6849,
             0.5551,
             0.6446,
             0.6869,
             0.7186,
             x_beta[4],
             x_beta[5],
             x_beta[6],
             x_beta[7]]
    
    cocoon = [0,
              0.787752,
              0.787752,
              0.827015,
              0.827015,
              0.787752,
              0.827015,
              0.801538,
              0.811144,
              0.6849,
              0.5551,
              0.6446,
              0.6869,
              0.7186,
              x_beta[4],
              x_beta[5],
              x_beta[6],
              x_beta[7]]

    tr_reduc = []
    date_list = []
    cocoon_reduc = []
    for idx in range(len(change_dates[:-1])):
        tr_reduc.extend([beta[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        date_list.extend([str(change_dates[idx] + dt.timedelta(days=x)) for x in range((change_dates[idx + 1] - change_dates[idx]).days)])
        cocoon_reduc.extend([cocoon[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
    
    nodaysrem = (instance.end_date.date() - change_dates[-1]).days 
    date_list.extend([str(change_dates[-1] + dt.timedelta(days=x)) for x in range(nodaysrem)])
    tr_reduc.extend([beta[0]] * nodaysrem)
    cocoon_reduc.extend([cocoon[0]] * nodaysrem)
        
    d = {'date': pd.to_datetime(date_list), 'transmission_reduction': tr_reduc}
    df_transmission = pd.DataFrame(data=d)
    transmission_reduction = [(d, tr) for (d, tr) in zip(df_transmission['date'], df_transmission['transmission_reduction'])]
    instance.cal.load_fixed_transmission_reduction(transmission_reduction, present_date=instance.end_date)
    

    d = {'date': pd.to_datetime(date_list), 'cocooning': cocoon_reduc}
    df_cocooning = pd.DataFrame(data=d)
    cocooning = [(d, c) for (d, c) in zip(df_cocooning['date'], df_cocooning['cocooning'])]
    instance.cal.load_fixed_cocooning(cocooning, present_date=instance.end_date)
    #############
    #logger.info(f'beta: {str(beta)}')

    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = 1
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = 1
    n_replicas_test = 1

    given_threshold = eval('[-1,0,5,20,70]')
    given_date = None
    # if a threshold/threshold+stepping date is given, then it carries out a specific task
    # if not, then search for a policy
    selected_policy = None
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP.constant_policy(instance, tiers.tier, given_threshold, tiers.community_transmision)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date, tiers.community_transmision)
    
    # Define additional variables included in the fit
    
    instance.epi.alpha1_omic = x_beta[0]
    instance.epi.alpha2_omic = x_beta[1]
    instance.epi.alpha3_omic = x_beta[2]
    instance.epi.alpha4_omic= x_beta[3]
    instance.epi.immune_escape_rate = 0.22
    print('new value: ', x_beta)
    selected_vaccine_policy.reset_vaccine_history(instance, -1)
  
                    
    sim_output = deterministic_path(instance=instance,
                                    tiers=tiers.tier,
                                    vaccines = vaccines,
                                    obj_func=multi_tier_objective,
                                    n_replicas_train=n_replicas_train,
                                    n_replicas_test=n_replicas_test,
                                    instance_name='det_path',
                                    policy_class=tiers.tier_type,
                                    policy=selected_policy,
                                    vaccine_policy=selected_vaccine_policy,
                                    mp_pool=None,
                                    crn_seeds=[],
                                    unique_seeds_ori=[],
                                    forcedOut_tiers=eval('[]'),
                                    redLimit=100000,
                                    after_tiers=eval('[0,1,2,3,4]'),
                                    policy_field='ToIHT',
                                    policy_ub=None)

 
    if instance.city == 'austin':
        hosp_benchmark = None
        real_hosp = [a_i - b_i for a_i, b_i in zip(instance.cal.real_hosp, real_icu)] 
        hosp_benchmark = [sim_output['IH'][t].sum() for t in range(t_start, len(instance.cal.real_hosp))]
        residual_error_IH = [a_i - b_i for a_i, b_i in zip(real_hosp, hosp_benchmark)]
        
        
        icu_benchmark = [sim_output['ICU'][t].sum() for t in range(t_start, len(instance.cal.real_hosp))]
        w_icu = 1.5
        residual_error_ICU = [a_i - b_i for a_i, b_i in zip(real_icu, icu_benchmark)]
        residual_error_ICU = [element * w_icu for element in residual_error_ICU]
        residual_error_IH.extend(residual_error_ICU)
    
        w_iyih = 7.3*(1 - 0.10896) + 9.9*0.10896
        daily_ad_benchmark = [sim_output['ToIHT'][t].sum() for t in range(t_start, len(instance.cal.real_hosp) - 1)] 
        print('hospital admission: ', sum(daily_ad_benchmark))
        residual_error_IYIH = [a_i - b_i for a_i, b_i in zip(hosp_ad, daily_ad_benchmark)]
        residual_error_IYIH = [element * w_iyih for element in residual_error_IYIH]
        residual_error_IH.extend(residual_error_IYIH)
        
        w_d = w_iyih*5
        daily_death_benchmark = [sim_output['D'][t+1].sum() - sim_output['D'][t].sum() for t in range(t_start, len(instance.cal.real_hosp) - 1)] 
        daily_death_benchmark.insert(0, 0)
        daily_death_benchmark = [sim_output['ToICUD'][t].sum() for t in range(len(instance.cal.real_hosp) - 1)] 
        daily_death_benchmark.insert(0, 0)
        residual_error_death = [a_i - b_i for a_i, b_i in zip(real_death_from_hosp, daily_death_benchmark)]
        residual_error_death = [element * w_d for element in residual_error_death]
        residual_error_IH.extend(residual_error_death)     
        
        w_iyd = w_d
        real_toIYD = [a_i - b_i for a_i, b_i in zip(real_death_total, real_death_from_hosp)]
        daily_toIYD_benchmark = [sim_output['ToIYD'][t].sum() for t in range(t_start, len(instance.cal.real_hosp) - 1)] 
        daily_death_benchmark.insert(0, 0)
        residual_error_death = [a_i - b_i for a_i, b_i in zip(real_toIYD, daily_toIYD_benchmark)]
        residual_error_death = [element * w_iyd for element in residual_error_death]
        residual_error_IH.extend(residual_error_death)  
    #breakpoint()
    print('residual error:', )
    return residual_error_IH 

  
def least_squares_fit(initial_guess, kwargs):
    # Function that runs the least squares fit
    result = least_squares(residual_error,
                           initial_guess,
                           bounds = ([0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 10, 1, 1, 1, 1, 1]),
                           method='trf', verbose=2,
                           kwargs = kwargs)
    return result 
    
def run_fit(instance,
            tiers,
            vaccines,
            obj_func,
            n_replicas_train=100,
            n_replicas_test=100,
            instance_name=None,
            policy_class='constant',
            policy=None,
            vaccine_policy=None,
            mp_pool=None,
            crn_seeds=[],
            unique_seeds_ori=[],
            forcedOut_tiers=None,
            redLimit=1000,
            after_tiers=[0,1,2,3,4],
            policy_field="IYIH",
            policy_ub=None,
            method="lsq",
            dfo_obj=None,
            initial=None,
            start_date=None):
 
    if instance.city == 'austin':
        daily_admission_file_path = instance.path_to_data  / "austin_hosp_ad_updated.csv"
        hosp_ad = read_hosp(daily_admission_file_path, start_date, "admits")
        
        daily_icu_file_path = instance.path_to_data  / "austin_real_icu_updated.csv"
        real_icu = read_hosp(daily_icu_file_path, start_date)  

        daily_death_file_path = instance.path_to_data  / "austin_real_death_from_hosp_updated.csv"
        real_death_from_hosp = read_hosp(daily_death_file_path, start_date) 
        
        daily_total_death_file_path = instance.path_to_data  / "austin_real_total_death.csv"
        real_death_total = read_hosp(daily_total_death_file_path, start_date)          
  
        #time blocks
        change_dates = [dt.date(2020, 2, 15),
                        dt.date(2020, 3, 24),
                        dt.date(2020, 5, 21),
                        dt.date(2020, 6, 26),
                        dt.date(2020, 8, 20),
                        dt.date(2020, 10, 29),
                        dt.date(2020, 11, 30),
                        dt.date(2020, 12, 31),
                        dt.date(2021, 1, 12),
                        dt.date(2021, 3, 13),
                        dt.date(2021, 6, 20),
                        dt.date(2021, 7, 31),
                        dt.date(2021, 8, 22),
                        dt.date(2021, 9, 24),
                        dt.date(2021, 10, 25),
                        dt.date(2021, 12, 8),
                        dt.date(2022, 1, 5),
                        dt.date(2022, 3, 10),
                        dt.date(2022, 4, 20)] 

  
        x = np.array([0.6, 0.15, 3.5, 0.002, 0.57, 0.55, 0.68, 0.55])       

    selected_vaccine_policy = VAP.vaccine_policy(instance, vaccines, 'deterministic')
    
    kwargs  = {'change_dates' : change_dates,
               'instance' : instance,
               'tiers' : tiers,
               'hosp_ad': hosp_ad,
               'real_icu': real_icu,
               'real_death_from_hosp': real_death_from_hosp,
               'real_death_total': real_death_total,
               'vaccines' : vaccines,
               'selected_vaccine_policy': selected_vaccine_policy,
               't_start':instance.cal.calendar.index(start_date)
               }
    ########## ########## ##########
    #Run least squares and choose a method
    res = least_squares_fit(x, kwargs)
    SSE = res.cost

    ########## ########## ##########
    #Get variable value
    opt_tr_reduction = res.x
    contact_reduction = np.array([  0.052257,
             0.787752,
             0.641986,
             0.827015,
             0.778334,
             0.752980,
             0.674321,
             0.801538,
             0.811144,
             0.6849,
             0.5551,
             0.6446,
             0.6869,
             0.7186,
             opt_tr_reduction[4],
             opt_tr_reduction[5],
             opt_tr_reduction[6],
             opt_tr_reduction[7]])
    
    cocoon = np.array([0,
              0.787752,
              0.787752,
              0.827015,
              0.827015,
              0.787752,
              0.827015,
              0.801538,
              0.811144,
              0.6849,
              0.5551,
              0.6446,
              0.6869,
              0.7186,
              opt_tr_reduction[4],
              opt_tr_reduction[5],
              opt_tr_reduction[6],
              opt_tr_reduction[7]])
                                                                                
    betas = instance.epi.beta*(1 - (contact_reduction))
    end_date = []
    for idx in range(len(change_dates[1:])):
        end_date.append(str(change_dates[1:][idx] - dt.timedelta(days=1)))
    
    print('beta_0:', instance.epi.beta)   
    print('SSE:', SSE)   
    print('alpha1_omic=', opt_tr_reduction[0])
    print('alpha2_omic=', opt_tr_reduction[1])
    print('alpha3_omic=', opt_tr_reduction[2])
    print('alpha4_omic=', opt_tr_reduction[3])
    print('immune escape', 0.22)
    
    #breakpoint()
    table = pd.DataFrame({'start_date': change_dates[:-1], 'end_date': end_date, 'contact_reduction': contact_reduction, 'beta': betas, 'cocoon': cocoon})
    print(table)
    
    
    
    #Save optimized values to transmission_new.csv
    tr_reduc = []
    date_list = []
    cocoon_reduc = []
    for idx in range(len(change_dates[:-1])):
        tr_reduc.extend([contact_reduction[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        date_list.extend([str(change_dates[idx] + dt.timedelta(days=x)) for x in range((change_dates[idx + 1] - change_dates[idx]).days)])
        cocoon_reduc.extend([cocoon[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
    
    d = {'date': pd.to_datetime(date_list), 'transmission_reduction': tr_reduc, 'cocooning': cocoon_reduc}
    df_transmission = pd.DataFrame(data=d)

    return df_transmission
 
