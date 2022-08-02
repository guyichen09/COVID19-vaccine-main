if __name__ == '__main__':

    import time
    start = time.time()

    import sys
    # sys.path.append("/home/lpei/scratch/06202022/COVID19-vaccine-main/VaccineAllocation")
    # sys.path.append("/home/lpei/scratch/06202022/COVID19-vaccine-main")

    sys.path.append("/Users/lindapei/Dropbox/RESEARCH/Summer2022/COVID19-vaccine-main/Original Code from Nazlican/VaccineAllocation")
    sys.path.append("/Users/lindapei/Dropbox/RESEARCH/Summer2022/COVID19-vaccine-main/Original Code from Nazlican")

    import multiprocessing as mp
    from utils import parse_arguments
    from VaccineAllocation import load_config_file, change_paths
    import datetime as dt
    import numpy as np
    # Parse arguments
    args = parse_arguments()
    # Load config file
    load_config_file(args.f_config)
    # Adjust paths
    change_paths(args)
    
    from instances import load_instance, load_tiers, load_seeds, load_vaccines
    from objective_functions import multi_tier_objective
    from trigger_policies import MultiTierPolicy as MTP
    from trigger_policies import CDCTierPolicy as CTP
    from vaccine_policies import VaccineAllocationPolicy as VAP
    from policy_search_functions import trigger_policy_search, trigger_policy_search_det
    
    # Parse city and get corresponding instance
    instance = load_instance(args.city, setup_file_name=args.f, transmission_file_name=args.tr, hospitalization_file_name=args.hos)
    train_seeds, test_seeds = load_seeds(args.city, args.seed)
    tiers = load_tiers(args.city, tier_file_name=args.t)
    vaccines = load_vaccines(args.city, instance, vaccine_file_name=args.v, booster_file_name = args.v_boost, vaccine_allocation_file_name = args.v_allocation)
    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = args.n_proc
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = args.train_reps
    n_replicas_test = args.test_reps
    
    # Create the pool (Note: pool needs to be created only once to run on a cluster)
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None

    given_vaccine_policy_no = args.n_policy if args.n_policy is not None else False
    given_threshold = eval(args.gt) if args.gt is not None else None
    given_date = eval('dt.datetime({})'.format(args.gd)) if args.gd is not None else None
    
    # if a threshold/threshold+stepping date is given, then it carries out a specific task
    # if not, then search for a policy
    selected_policy = None
    if tiers.tier_type == "CDC":
        selected_policy = CTP.policy(instance, tiers)
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP.constant_policy(instance, tiers.tier, given_threshold, tiers.community_transmision)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date, tiers.community_transmision)
    elif tiers.tier_type == 'linear':
        if given_threshold is not None:
            selected_policy = MTP.linear_policy(instance, tiers.tier, given_threshold, given_date, args.gs, tiers.community_transmision)
      
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
            selected_vaccine_policy = VAP.sort_contact_matrix_vaccine_policy(instance, vaccines, "deterministic")
        elif given_vaccine_policy_no == 5:
            selected_vaccine_policy = VAP.no_vaccine_policy(instance, vaccines, 'deterministic')
        elif given_vaccine_policy_no == 6:
            selected_vaccine_policy = VAP.phase_1b_policy(instance, vaccines, args.percentage)
        elif given_vaccine_policy_no == 7:
            selected_vaccine_policy = VAP.vaccine_policy(instance, vaccines, 'deterministic')
            
    task_str = str(selected_policy) if selected_policy is not None else f'opt{len(tiers.tier)}'
    instance_name = f'{args.f_config[:-5]}_{args.t[:-5]}_{task_str}_{args.tr}_{args.f}'
    # read in the policy upper bound
    if args.pub is not None:
        policy_ub = eval(args.pub)
    else:
        policy_ub = None

    if args.det == True:
        best_triger_policy, file_path = trigger_policy_search_det(instance = instance,
                                                                          tiers = tiers.tier,
                                                                          vaccines = vaccines,
                                                                          obj_func = multi_tier_objective,
                                                                          n_replicas_train= n_replicas_train,
                                                                          n_replicas_test= n_replicas_test,
                                                                          instance_name= instance_name,   
                                                                          policy_class = tiers.tier_type,
                                                                          policy=selected_policy,
                                                                          vaccine_policy= selected_vaccine_policy,
                                                                          mp_pool= mp_pool,
                                                                          crn_seeds= train_seeds,
                                                                          unique_seeds_ori=test_seeds,
                                                                          forcedOut_tiers= eval(args.fo),
                                                                          redLimit=args.rl,
                                                                          after_tiers=eval(args.aftert),
                                                                          policy_field = args.field,
                                                                          policy_ub= policy_ub)
    elif args.det == False: 
        stoch_replicas, best_triger_policy, file_path = trigger_policy_search(instance = instance,
                                                                          tiers = tiers.tier,
                                                                          vaccines = vaccines,
                                                                          obj_func = multi_tier_objective,
                                                                          n_replicas_train= n_replicas_train,
                                                                          n_replicas_test= n_replicas_test,
                                                                          instance_name= instance_name,   
                                                                          policy_class = tiers.tier_type,
                                                                          policy=selected_policy,
                                                                          vaccine_policy= selected_vaccine_policy,
                                                                          mp_pool= mp_pool,
                                                                          crn_seeds= train_seeds,
                                                                          unique_seeds_ori=test_seeds,
                                                                          forcedOut_tiers= eval(args.fo),
                                                                          redLimit=args.rl,
                                                                          after_tiers=eval(args.aftert),
                                                                          policy_field = args.field,
                                                                          policy_ub= policy_ub)
    else:
        print('Wrong input argument for det')
    
    # print(stoch_replicas, best_triger_policy, file_path)

    print("Time")
    print(time.time() - start)
    print("=====================")

    print(best_triger_policy)
