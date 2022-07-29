import pickle
import numpy as np
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
from interventions import create_intLevel, form_interventions
from itertools import product
from SEIYAHRD import simulate_p
from trigger_policies import MultiTierPolicy
from VaccineAllocation import config, logger, output_path
from utils import profile_log, print_profiling_log
from threshold_policy import run_multi_calendar, policy_multi_iterator, stoch_simulation_iterator
from objective_functions import multi_tier_objective
from vaccine_policies import VaccineAllocationPolicy as VAP
from vaccine_policies import  fix_hist_allocation
import iteround
datetime_formater = '%Y-%m-%d %H:%M:%S'
date_formater = '%Y-%m-%d'

def trigger_policy_search(instance,
                  tiers,
                  vaccines,
                  obj_func,
                  n_replicas_train=100,
                  n_replicas_test=100,
                  instance_name=None,
                  policy_class='constant',
                  community_transmision = "green",
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
    '''
    Search for optimal trigger policy with a fixed vaccine policy.
    '''
    
    # Set up for policy search: build interventions according to input tiers
    fixed_TR = list(filter(None, instance.cal.fixed_transmission_reduction))
    tier_TR = [item['transmission_reduction'] for item in tiers]
    uniquePS = sorted(np.unique(np.append(fixed_TR, np.unique(tier_TR))))
    sc_levels = np.unique([tier['school_closure'] for tier in tiers] + [0, 1])
    fixed_CO = list(filter(None, instance.cal.fixed_cocooning))
    co_levels = np.unique(np.append([tier['cocooning'] for tier in tiers], np.unique(fixed_CO)) + [0])
    intervention_levels = create_intLevel(sc_levels, co_levels, uniquePS)
    interventions_train = form_interventions(intervention_levels, instance.epi, instance.N)
    t_start = len(instance.real_hosp)
    print(t_start)
    
    selected_vaccine_policy = vaccine_policy
    # Build an iterator of all the candidate trigger policies (with given fixed vaccine policy) to be simulated by simulate_p
    sim_configs = policy_multi_iterator(instance,
                                        tiers,
                                        vaccines, 
                                        obj_func,
                                        interventions_train,
                                        policy_class = policy_class,
                                        community_transmision = community_transmision,
                                        fixed_policy = policy,
                                        fixed_vaccine_policy = selected_vaccine_policy,
                                        policy_field = policy_field,
                                        policy_ub = policy_ub)
    
    # Launch parallel simulation
    all_outputs = simulate_p(mp_pool, sim_configs)
    
    logger.info(f'Simulated candidates: {len(all_outputs)}: {profile_log["simulate_p"]}')
    
    # Search of the best feasible candidate
    best_cost, best_sim, best_policy, best_params = np.Inf, None, None, None
    hosp_benchmark = instance.real_hosp
    if len(all_outputs) == 1:   
        # Skip search if there is only one candidate
        sim_output, cost, best_policy, _vac_policy, seed_0, kwargs_out = all_outputs[0]
        best_cost = cost
        best_sim = sim_output
        best_params = kwargs_out
        cost_record = {}
    else:
       # Check feasibility and optimality of all the simulated candidates
       #   - Feasibility: chance constraint feasible
       #   - Optimality: according to obj_func, but ignoring penalties
        
        SRS_pruned = 0  # Tally of candidates pruned by square root staffing
        found_feasible_policy = False  # Feasible solution found flag
        chance_constrain_vio = np.inf  # Min violation of the CC
        best_cost_inf = np.inf  # Cost of the least inf solution
        cost_record = {}  # Costs information (lockdown cost, over-capacity cost) for each candidate


        for ix, output_i in enumerate(all_outputs):
            # Loop through all the
            sim_output, cost, policy_i, _vac_policy, seed_i, kwargs_out = output_i
            cost_record_ij = []
            logger.info(f'seed_i: {str(seed_i)}')
    
            if cost < np.inf or not found_feasible_policy:
                # Staffing rule feasible
                logger.info(f'Considering: {str(policy_i)}')
                kwargs_out['opt_phase'] = False  # not longer optimizing
                kwargs_out['infeasible_penalty'] = False  # don't penalize with staffing rule
                kwargs_out['over_capacity_cost'] = config['over_capacity_cost']  # add the over capacity penalty
                kwargs_out['obj_over_included'] = config['obj_over_included']  # whether the objective function includes over-capacity cost
                kwargs_out['sim_method'] = config['sim_method']
                kwargs_out['fo_tiers'] = forcedOut_tiers
                kwargs_out['changed_tiers'] = False
                kwargs_out['redLimit'] = redLimit
                kwargs_out['after_tiers'] = after_tiers
                kwargs_out['active_intervention'] = config['active_intervention']
                kwargs_out['extra_capacity_cost'] = config['extra_capacity_cost']
                kwargs_out['capacity_setup_cost'] = config['capacity_setup_cost']
                kwargs_out['icu_trigger'] = config['icu_trigger']
                kwargs_out['icu_capacity_cost'] = config['icu_capacity_cost']
                
                try:
                    start_date = dt.datetime.strptime(config["rd_start"], datetime_formater)
                    kwargs_out["rd_start"] = instance.cal.calendar_ix[start_date]
                    end_date = dt.datetime.strptime(config["rd_end"], datetime_formater)
                    kwargs_out["rd_end"] = instance.cal.calendar_ix[end_date]
                    kwargs_out["rd_rate"] = config["rd_rate"]
                except:
                    kwargs_out["rd_start"] = -1
                    kwargs_out["rd_end"] = -1
                    kwargs_out["rd_rate"] = 1

            # LOOPING THROUGH ALL FEASIBLE SOLUTIONS (POLICIES?)
                
                stoch_outputs_i = []  # List of valid samples if policy_i
                crn_seeds_i = []  # Common random number seed of policy_i (set once)
                deviation_output = []  # print out the deviation of the seeds
                total_train_reps = 0  # Total samples executed in the filtering procedure
                
                # =================================================
                # Sample until required number of replicas achieved
                n_loops = 0
                while len(stoch_outputs_i) < n_replicas_train:
                   # breakpoint()
                    chunksize = 1 if mp_pool is None else mp_pool._processes
                    #chunksize = chunksize if crn_seeds == [] else n_replicas_train
                    total_train_reps += chunksize
                    n_loops += chunksize + 1 if crn_seeds == [] else 0
                    if crn_seeds == []:
                        # no input seeds
                        seed_shift_var=n_loops
                        crn_input = None
                        chunkinput = chunksize
                    else:
                        seed_shift_var = 0
                        crn_input = crn_seeds[total_train_reps-chunksize:total_train_reps]
                        if len(crn_input) == 0:
                            # if the given seeds are run out, need to generate new seeds
                            #breakpoint()
                            crn_input = None
                            chunkinput = chunksize
                            seed_shift_var = crn_seeds[-1] + 1 + total_train_reps
                        else:
                            chunkinput = len(crn_input)
                   # breakpoint()
                    # Simulate n=chunksize samples of policy_i
                    out_sample_configs = stoch_simulation_iterator(instance,
                                                                   vaccines,
                                                                   policy_i,
                                                                   selected_vaccine_policy,
                                                                   obj_func,
                                                                   interventions_train, 
                                                                   det_sample_path = False,
                                                                   crn_seeds=crn_input,
                                                                   seed_shift=seed_shift_var,
                                                                   n_replicas=chunkinput, 
                                                                   **kwargs_out) 
                    
                    out_sample_outputs = simulate_p(mp_pool, out_sample_configs)               
                    `
                    # Eliminate invalid samples according to R-squared. Invalid samples are those that
                    # deviate from recent hospitalization data
                    
                    for sample_ij in out_sample_outputs:
                        sim_j, cost_j, policy_j, _vac_policy, seed_j, kwargs_j = sample_ij
                        real_hosp_end_ix = len(hosp_benchmark) 
                        IH_sim = sim_j['IHT'][0:real_hosp_end_ix]
                        IH_sim = IH_sim.sum(axis=(2,1))
                        f_benchmark = hosp_benchmark
                    
                        rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark))**2))/sum((np.array(f_benchmark) - np.mean(np.array(f_benchmark)))**2)
                        print('rsq', rsq)
                 
        
                        if rsq > 0.75:
                            stoch_outputs_i.append(sample_ij)
                            crn_seeds_i.append(seed_j)
                            if len(stoch_outputs_i) == n_replicas_train:
                                break
                 
                            
                        
                # Save CRN seeds for all policies yet to be evaluated      
                if crn_seeds == []:
                    assert len(np.unique(crn_seeds_i)) == n_replicas_train
                    crn_seeds = crn_seeds_i.copy()
                    logger.info(f'\tCRN SEEDS {str(crn_seeds)}, Deviation {str(deviation_output)}')
                # End of samples filtering procedure
                # =================================================
                
                # Check feasibility with valid traning samples
                logger.info(f'\tgot {len(stoch_outputs_i)} replicas sampling {total_train_reps}')
                stoch_replicas = [rep_i[0] for rep_i in stoch_outputs_i]
                stoch_costs = [rep_i[1] for rep_i in stoch_outputs_i]
                cost_record[str(policy_i)] = cost_record_ij
                infeasible_cap_field = {"ICU": instance.icu,
                                        "IHT": instance.hosp_beds,
                                        "IH": instance.hosp_beds - instance.icu
                                        }
                infeasible_replicas = np.sum([
                    np.any(stoch_replicas[rep_i][config['infeasible_field']].sum(axis=(1, 2))[t_start:] > infeasible_cap_field[config['infeasible_field']])
                    for rep_i in range(len(stoch_outputs_i))
                ])
                IH_feasible = infeasible_replicas <= int(config['chance_constraint_epsilon'] * n_replicas_train)
                expected_cost = np.mean(stoch_costs)
                logger.info(
                    f'\tInf reps: {infeasible_replicas}  Expected Cost: {expected_cost:.0f} best cost: {best_cost}')
                
                # Update incunbent solution: best_policy
                if not found_feasible_policy and not IH_feasible:
                    # Accept infeasible policies if no feasible policy found yet
                    # If an infeasible policy is accepted, it is the least infeasible
                    cc_vio = infeasible_replicas - int(config['chance_constraint_epsilon'] * n_replicas_train)
                    if cc_vio < chance_constrain_vio or (cc_vio == chance_constrain_vio and expected_cost < best_cost_inf):
                        chance_constrain_vio = cc_vio
                        best_sim, best_cost_inf, best_policy, best_params = sim_output, expected_cost, policy_i, kwargs_out
                        print('Least inf : ', policy_i, '  ', cc_vio, '   ', expected_cost)
                else:
                    # Feasible solution replace incumbent according to expected cost
                    if expected_cost < best_cost and IH_feasible:
                        best_sim = sim_output
                        best_cost = expected_cost
                        best_policy = policy_i
                        best_params = kwargs_out
                        found_feasible_policy = True  # Flag update, inf policies not longer accepted
                        logger.info(
                            f'\tNew feasible solution -> inf reps {infeasible_replicas} : exp. cost: {expected_cost}')
            else:
                # Policy infeasible w.r.t square root staffing rule, solution discarded.
                logger.info(f'Discarded: {str(policy_i)}')
                SRS_pruned += 1
        logger.info(f'SRS pruned {SRS_pruned} out of {len(all_outputs)} candidates')
    print_profiling_log(logger)
    
    # ===================================================================

    # Final evaluation of the best policy using sample filter for samples
    best_params['opt_phase'] = False  # not longer optimizing
    best_params['infeasible_penalty'] = False  # don't penalize with staffing rule
    best_params['over_capacity_cost'] = config['over_capacity_cost']  # add the over capacity penalty
    best_params['obj_over_included'] = config['obj_over_included']
    best_params['sim_method'] = config['sim_method']
    best_params['active_intervention'] = config['active_intervention']
    best_params['fo_tiers'] = forcedOut_tiers
    best_params['changed_tiers'] = False
    best_params['redLimit'] = redLimit
    best_params['after_tiers'] = after_tiers
    best_params['extra_capacity_cost'] = config['extra_capacity_cost']
    best_params['capacity_setup_cost'] = config['capacity_setup_cost']
    best_params['icu_trigger'] = config['icu_trigger']
    best_params['icu_capacity_cost'] = config['icu_capacity_cost']
    try:
        start_date = dt.datetime.strptime(config["rd_start"], datetime_formater)
        best_params["rd_start"] = instance.cal.calendar_ix[start_date]
        end_date = dt.datetime.strptime(config["rd_end"], datetime_formater)
        best_params["rd_end"] = instance.cal.calendar_ix[end_date]
        best_params["rd_rate"] = config["rd_rate"]
    except:
        best_params["rd_start"] = -1
        best_params["rd_end"] = -1
        best_params["rd_rate"] = 1
    #breakpoint()    
    total_test_reps = 0
    det_path_computed = False
    cost_record_ij = []
    stoch_outputs_test = []
    #stoch_outputs_test.append(all_outputs[0])
    unique_seeds = []
   # unique_seeds.append(seed_0)
    while len(stoch_outputs_test) < n_replicas_test:
        #breakpoint()
        chunksize = 1 if mp_pool is None else 1 * mp_pool._processes
        total_test_reps += chunksize
        if unique_seeds_ori == []:
            # no input seeds
            seed_shift_var=10_00000 + total_test_reps
            crn_input = None
            chunkinput = chunksize
        else:
            seed_shift_var = 0
            crn_input = unique_seeds_ori[total_test_reps-chunksize:total_test_reps]
            if len(crn_input) == 0:
                #breakpoint()
                # if the given seeds are run out, need to generate new seeds
                chunkinput = chunksize
                crn_input = None
                seed_shift_var = unique_seeds_ori[-1] + 1 + total_test_reps
            else:
                chunkinput = len(crn_input)   
        out_sample_configs = stoch_simulation_iterator(instance,
                                                       vaccines,
                                                       best_policy,
                                                       selected_vaccine_policy,
                                                       obj_func,
                                                       interventions_train,
                                                       det_sample_path=not det_path_computed,
                                                       crn_seeds=crn_input,
                                                       seed_shift=seed_shift_var,
                                                       n_replicas=chunkinput,
                                                       **best_params)
        
        out_sample_outputs = simulate_p(mp_pool, out_sample_configs)
        
        for sample_ij in out_sample_outputs:
            sim_j, cost_j, policy_j,  _vac_policy, seed_j, kwargs_j = sample_ij
           
            real_hosp_end_ix = len(hosp_benchmark) 
            IH_sim = sim_j['IHT'][0:real_hosp_end_ix]
            IH_sim = IH_sim.sum(axis=(2,1))
            f_benchmark = hosp_benchmark
            # # time_blocks = [dt.datetime(2020, 5, 1),
            # #             dt.datetime(2020, 9, 1),
            # #             dt.datetime(2020, 12, 1), 
            # #             dt.datetime(2021, 4, 1),
            # #             dt.datetime(2021, 7, 1),
            # #             dt.datetime(2021, 11, 1),
            # #             dt.datetime(2021, 12, 1),
            # #             dt.datetime(2022, 4, 4)] 
            # time_blocks = [dt.datetime(2020, 5, 1)]
            # valid_seed = True
            # for tbx in range(len(time_blocks)):
            #     T_end = instance.cal.calendar_ix[time_blocks[tbx]]
            #     rsq = 1 - np.sum(((np.array(IH_sim[:T_end]) - np.array(f_benchmark[:T_end]))**2))/sum((np.array(f_benchmark[:T_end]) - np.mean(np.array(f_benchmark[:T_end])))**2)
            #     print('test rsq', rsq, 'until', time_blocks[tbx])
            #     if rsq <= 0.75:  
            #         valid_seed = False
            #     #breakpoint()
            
            rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark))**2))/sum((np.array(f_benchmark) - np.mean(np.array(f_benchmark)))**2)
           
                
            print('rsq', rsq)
            #print('')   
                       
            if rsq > 0.75:    
                stoch_outputs_test.append(sample_ij)
                unique_seeds.append(seed_j)
           # else:
            #    breakpoint()
            if len(stoch_outputs_test) == n_replicas_test:
                break
            
            
    assert len(np.unique(unique_seeds)) == n_replicas_test
    bpStr = str(best_policy)
    if bpStr not in cost_record.keys():
        cost_record[str(best_policy)] = cost_record_ij
    logger.info(f'Got {len(stoch_outputs_test)} replicas sampling {total_test_reps}')
    stoch_replicas = [rep_i[0] for rep_i in stoch_outputs_test]
    stoch_costs = [rep_i[1] for rep_i in stoch_outputs_test]
    infeasible_cap_field = {"ICU": instance.icu,
                            "IHT": instance.hosp_beds,
                            "IH": instance.hosp_beds - instance.icu
                            }
    infeasible_replicas = np.sum([
        np.any(stoch_replicas[rep_i][config['infeasible_field']].sum(axis=(1, 2))[t_start:] > infeasible_cap_field[config['infeasible_field']])
        for rep_i in range(len(stoch_replicas))
    ])
    expected_cost = np.mean(stoch_costs)
    logger.info(f'Optimized policy: {str(best_policy)}')
    logger.info(f'Cost: {expected_cost}')
    logger.info(f'Inf scenarios: {infeasible_replicas} out of {len(stoch_costs)}')
    print_profiling_log(logger)
    # Save solution
    instance_name = instance_name if instance_name is not None else f'output_{instance.city}.p'
    file_path = output_path / f'{instance_name}.p'
    if file_path.is_file():
        file_path = output_path / f'{instance_name}_{str(dt.datetime.now())}.p'
    crns_out = np.array(crn_seeds)
    unique_out = np.array(unique_seeds)
    
    selected_vaccine_policy = []
    
    with open(str(file_path), 'wb') as outfile:
        pickle.dump(
            (instance, interventions_train, best_params, best_policy, vaccines, stoch_replicas, 
             best_sim, cost_record,  config,
             (crns_out[crns_out >= 0], unique_out[unique_out >= 0])),
            outfile, pickle.HIGHEST_PROTOCOL)
    print(unique_seeds)
    return stoch_replicas, best_policy, file_path