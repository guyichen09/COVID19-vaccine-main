if __name__ == '__main__':
    # from mpi4py import MPI
    #
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    # num_workers = size - 1
    # master_rank = size - 1

    import time

    start = time.time()

    from SimObjects import MultiTierPolicy
    from DataObjects import City, TierInfo, Vaccine
    from SimModel import SimReplication

    from InputOutputTools import SimReplicationIO

    austin = City("austin",
                  "austin_test_IHT.json",
                  "calendar.csv",
                  "setup_data_Final.json",
                  "transmission.csv",
                  "austin_real_hosp_updated.csv",
                  "delta_prevalence.csv",
                  "omicron_prevalence.csv",
                  "variant_prevalence.csv"
                  )

    tiers = TierInfo("austin", "tiers5_opt_Final.json")

    vaccines = Vaccine(austin,
                       "austin",
                       "vaccines.json",
                       "booster_allocation_fixed.csv",
                       "vaccine_allocation_fixed.csv")

    thresholds = (-1, 100, 200, 5000, 10000)
    mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
    rep500 = SimReplication(austin, vaccines, mtp, 500)
    rep5000 = SimReplication(austin, vaccines, mtp, 5000)

    # rep500.simulate_time_period(0, 100)
    # rep500.simulate_time_period(100, 783)
    # rep500.simulate_time_period(783, 945)
    #
    # print(rep500.compute_rsq())
    # print(rep500.policy.compute_cost())

    rep5000.simulate_time_period(0, 945)

    print(rep5000.compute_rsq())
    print(rep5000.policy.compute_cost())

    rep500b = SimReplication(austin, vaccines, None, 500)
    rep500b.simulate_time_period(0, 783)
    print(rep500b.compute_rsq())

    # print(time.time() - start)
    #
    # start = time.time()
    #
    # thresholds = (-1,5,15,30,50)
    #
    # a_mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
    # a = SimReplication(austin, vaccines, a_mtp, 100)
    #
    # a.simulate_time_period(0,100)
    # a.rng_generator = np.random.RandomState(1000)
    # a.simulate_time_period(100, 945)
    #
    # SimReplicationIO.export_rep_to_file(a, "sim_rep.json", "mtp.json", "v0.json", "v1.json", "v2.json", "v3.json")
    #
    # b_mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
    # b = SimReplication(austin, vaccines, b_mtp, 1000)
    #
    SimReplicationIO.load_vars_from_file(b, "sim_rep.json", "mtp.json", "v0.json", "v1.json", "v2.json", "v3.json")

    b.simulate_time_period(100,945)
    #
    # print(a.policy.compute_cost())
    # print(b.policy.compute_cost())
    #
    # print(a.compute_rsq())
    # print(b.compute_rsq())
    #
    # print(time.time() - start)
