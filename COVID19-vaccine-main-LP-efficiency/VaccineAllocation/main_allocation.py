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

    from utils import parse_arguments
    from VaccineAllocation import load_config_file

    # Parse arguments
    args = parse_arguments()
    # Load config file
    load_config_file(args.f_config)
    
    from instances import load_instance, load_tiers, load_vaccines
    from policy_search_functions import LP_trigger_policy_search

    # Parse city and get corresponding instance
    instance = load_instance(args.city, setup_file_name=args.f, transmission_file_name=args.tr, hospitalization_file_name=args.hos)
    tiers = load_tiers(args.city, tier_file_name=args.t)

    vaccines = load_vaccines(args.city, instance, vaccine_file_name=args.v, booster_file_name = args.v_boost, vaccine_allocation_file_name = args.v_allocation)
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = args.train_reps
    n_replicas_test = args.test_reps

    given_threshold = eval(args.gt) if args.gt is not None else None

    if args.pub is not None:
        policy_ub = eval(args.pub)
    else:
        policy_ub = None

    LP_trigger_policy_search(instance=instance,
                          tiers=tiers.tier,
                          vaccines=vaccines)

    print(time.time() - start)
