#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:57:28 2021

@author: haoxiangyang
"""

if __name__ == '__main__':
    import multiprocessing as mp
    from utils import parse_arguments, parse_arguments_acs
    from VaccineAllocation import load_config_file, change_paths
    import datetime as dt
    # Parse arguments
    args = parse_arguments_acs()
    # Load config file
    load_config_file(args.f_config)
    # Adjust paths
    change_paths(args)
    
    from instances import load_instance, load_tiers, load_seeds, load_vaccines
    from objective_functions import multi_tier_objective, multi_tier_objective_ACS
    from trigger_policies import MultiTierPolicy as MTP, MultiTierPolicy_ACS as MTP_ACS
    from policy_search_functions import trigger_policy_search, capacity_policy_search
    
    # Parse city and get corresponding instance
    instance = load_instance(args.city, setup_file_name=args.f, transmission_file_name=args.tr, hospitalization_file_name=args.hos)
    train_seeds, test_seeds = load_seeds(args.city, args.seed)
    print('test seeds', test_seeds)
    print('train seeds', train_seeds)
    tiers = load_tiers(args.city, tier_file_name=args.t)
    #vaccines = load_vaccines(args.city, vaccine_file_name=args.v)
    vaccines = load_vaccines(args.city, instance, vaccine_file_name=args.v, booster_file_name = args.v_boost, vaccine_allocation_file_name = args.v_allocation)
   
    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = args.n_proc
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = args.train_reps
    n_replicas_test = args.test_reps
    
    # Create the pool (Note: pool needs to be created only once to run on a cluster)
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None
    
    # check if the "do-nothing" / 'Stage 1 option is in the tiers. If not, add it
    originInt = {
        "name": "Stage 1",
        "transmission_reduction": 0,
        "cocooning": 0,
        "school_closure": 0,
        "min_enforcing_time": 14,
        "daily_cost": 0,
        "color": 'green'
    }
    
    # read in acs related information
    acs_bounds = eval(args.acs_bounds) if args.acs_bounds is not None else None
    acs_times = eval(args.acs_times) if args.acs_times is not None else None
    acs_leadT = args.acs_leadT
    acs_Q = args.acs_Q
    acs_type = args.acs_type
    
    if tiers.tier_type == 'constant' or tiers.tier_type == 'linear':
        originInt["candidate_thresholds"] = [-1]  # Means that there is no lower bound
    elif tiers.tier_type == 'step':
        originInt["candidate_thresholds"] = [[-1], [-0.5]]
    
    if not (originInt in tiers.tier):
        tiers.tier.insert(0, originInt)
    
    given_vaccine_policy_no = args.n_policy if args.n_policy is not None else False
    given_threshold = eval(args.gt) if args.gt is not None else None
    given_date = eval('dt.datetime({})'.format(args.gd)) if args.gd is not None else None
    
    
    # if a threshold/threshold+stepping date is given, then it carries out a specific task
    # if not, then search for a policy
    selected_policy = None
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP_ACS.constant_policy(instance, tiers.tier, given_threshold)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date)
    elif tiers.tier_type == 'linear':
        if given_threshold is not None:
            selected_policy = MTP.linear_policy(instance, tiers.tier, given_threshold, given_date, args.gs)
    #breakpoint()        
    # if a vaccine allocation is given, then it carries out a specific task
    # if not, then search for a policy
    selected_vaccine_policy = None
    if given_vaccine_policy_no is not False:
        if given_vaccine_policy_no == 0:
            selected_vaccine_policy = VAP.high_risk_senior_first_vaccine_policy(instance, vaccines)
        elif given_vaccine_policy_no == 1:
            selected_vaccine_policy = VAP.senior_first_vaccine_policy(instance, vaccines)
        elif given_vaccine_policy_no == 2:
            selected_vaccine_policy = VAP.min_death_vaccine_policy(instance, vaccines)
        elif given_vaccine_policy_no == 3:
            selected_vaccine_policy = VAP.min_infected_vaccine_policy(instance, vaccines)
        elif given_vaccine_policy_no == 4:
            selected_vaccine_policy = VAP.sort_contact_matrix_vaccine_policy(instance, vaccines)
        elif given_vaccine_policy_no == 5:
            selected_vaccine_policy = VAP.no_vaccine_policy(instance, vaccines)
        elif given_vaccine_policy_no == 6:
            selected_vaccine_policy = VAP.phase_1b_policy(instance, vaccines, args.percentage)
        elif given_vaccine_policy_no == 7:
            selected_vaccine_policy = VAP.vaccine_policy(instance, vaccines, 'deterministic')
            
    task_str = str(selected_policy) if selected_policy is not None else f'opt{len(tiers.tier)}'
    instance_name = f'{args.f_config[:-5]}_{args.t[:-5]}_{task_str}__{args.tr[:-4]}_{args.f[:-5]}'
    
    # read in the policy upper bound
    if args.pub is not None:
        policy_ub = eval(args.pub)
    else:
        policy_ub = None
    
    best_policy_replicas, best_policy, out_file = capacity_policy_search(instance=instance,
                                                                tiers=tiers.tier,
                                                                vaccines = vaccines,
                                                                obj_func=multi_tier_objective_ACS,
                                                                acs_bounds=acs_bounds,
                                                                acs_time_bounds=acs_times,
                                                                acs_lead_time=acs_leadT,
                                                                acs_Q=acs_Q,
                                                                acs_type=acs_type,
                                                                n_replicas_train=n_replicas_train,
                                                                n_replicas_test=n_replicas_test,
                                                                instance_name=instance_name,
                                                                policy_class=tiers.tier_type,
                                                                policy=selected_policy,
                                                                vaccine_policy= selected_vaccine_policy,
                                                                mp_pool=mp_pool,
                                                                crn_seeds=train_seeds,
                                                                unique_seeds_ori=test_seeds,
                                                                forcedOut_tiers=eval(args.fo),
                                                                redLimit=args.rl,
                                                                after_tiers=eval(args.aftert),
                                                                policy_field=args.field,
                                                                policy_ub=policy_ub
                                                                )
        

