if __name__ == '__main__':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    num_workers = size - 1
    master_rank = size - 1

    import time
    start = time.time()

    from SimObjects import MultiTierPolicy
    from DataObjects import ProblemInstance, TierInformation, VaccineInstance
    from SimModel import SimulationReplication

    austin = ProblemInstance("austin",
                             "austin_test_IHT.json",
                             "calendar.csv",
                             "setup_data_Final.json",
                             "transmission.csv",
                             "austin_real_hosp_updated.csv",
                             "delta_prevalence.csv",
                             "omicron_prevalence.csv",
                             "variant_prevalence.csv"
                             )

    tiers = TierInformation("austin", "tiers5_opt_Final.json")

    vaccines = VaccineInstance(austin,
                               "austin",
                               "vaccines.json",
                               "booster_allocation_fixed.csv",
                               "vaccine_allocation_fixed.csv")

    print(time.time() - start)

    start = time.time()

    thresholds = (-1,5,15,30,50)

    mtp = MultiTierPolicy(austin, tiers, thresholds, "constant", "green")

    test = SimulationReplication(austin, vaccines, mtp, 100)

    test.simulate_time_period(0,900,None)
    test.simulate_time_period(900,945,None)

    print(test.policy.compute_cost())

    print(test.compute_rsq())
    print(test.compute_ICU_violation())

    print(time.time() - start)
