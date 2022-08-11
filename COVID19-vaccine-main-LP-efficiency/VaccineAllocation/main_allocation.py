if __name__ == '__main__':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    num_workers = size - 1
    master_rank = size - 1

    import time
    start = time.time()

    import sys
    # sys.path.append("/home/lpei/scratch/06202022/COVID19-vaccine-main/VaccineAllocation")
    # sys.path.append("/home/lpei/scratch/06202022/COVID19-vaccine-main")

    sys.path.append("/Users/lindapei/Dropbox/RESEARCH/Summer2022/COVID19-vaccine-main/COVID19-vaccine-main-LP-efficiency/VaccineAllocation")
    sys.path.append("/Users/lindapei/Dropbox/RESEARCH/Summer2022/COVID19-vaccine-main/COVID19-vaccine-main-LP-efficiency/")

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
    from policy_search_functions import trigger_policy_search, LP_trigger_policy_search

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
            
    task_str = str(selected_policy) if selected_policy is not None else f'opt{len(tiers.tier)}'
    instance_name = f'{args.f_config[:-5]}_{args.t[:-5]}_{task_str}_{args.tr}_{args.f}'
    # read in the policy upper bound
    if args.pub is not None:
        policy_ub = eval(args.pub)
    else:
        policy_ub = None

    LP_trigger_policy_search(instance=instance,
                          tiers=tiers.tier,
                          vaccines=vaccines)

    print(time.time() - start)
