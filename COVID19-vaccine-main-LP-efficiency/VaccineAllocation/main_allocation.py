if __name__ == '__main__':
    import multiprocessing
    import mpi4py

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    num_workers = size - 1
    master_rank = size - 1

    import time
    import numpy as np
    import copy

    start = time.time()

    from SimObjects import MultiTierPolicy
    from DataObjects import City, TierInfo, Vaccine
    from SimModel import SimReplication
    import InputOutputTools

    austin = City("austin",
                  "austin_test_IHT.json",
                  "calendar.csv",
                  "setup_data_Final.json",
                  "transmission.csv",
                  "austin_real_hosp_updated.csv",
                  "delta_prevalence.csv",
                  "omicron_prevalence.csv",
                  "variant_prevalence.csv")

    tiers = TierInfo("austin", "tiers5_opt_Final.json")

    vaccines = Vaccine(austin,
                       "austin",
                       "vaccines.json",
                       "booster_allocation_fixed.csv",
                       "vaccine_allocation_fixed.csv")

    #############################################################

    def get_sample_paths(rank):

        print(rank)

        rep = SimReplication(austin, vaccines, None, rank)

        training_set_reps = 0
        total_reps = 0
        num_elim_per_stage = [0, 0, 0, 0, 0]
        timepoints = (25, 100, 200, 400, 783)
        all_rsq = []

        start = time.time()

        while training_set_reps < 300:
            total_reps += 1
            if total_reps % 10 == 0:
                print(time.time() - start)
                print(num_elim_per_stage)
                start = time.time()
            valid = True
            for i in range(5):
                rep.simulate_time_period(rep.next_t, timepoints[i])
                rsq = rep.compute_rsq()
                if rsq < 0.75:
                    num_elim_per_stage[i] += 1
                    valid = False
                    all_rsq.append(rsq)
                    # print(rsq)
                    break
            if valid == True:
                training_set_reps += 1
                all_rsq.append(rsq)
                # print(rsq)
                identifier = str(rank) + "_" + str(training_set_reps)
                InputOutputTools.export_rep_to_file(rep,
                                                    identifier + "_sim.json",
                                                    None,
                                                    identifier + "_v0.json",
                                                    identifier + "_v1.json",
                                                    identifier + "_v2.json",
                                                    identifier + "_v3.json",
                                                    identifier + "_epi_params.json")
            next_rng = rep.rng
            rep = SimReplication(austin, vaccines, None, 1)

            rep.rng = next_rng
            epi_rand = copy.deepcopy(rep.instance.base_epi)
            epi_rand.sample_random_params(rep.rng)
            epi_rand.setup_base_params()
            rep.epi_rand = epi_rand

            if total_reps % 1000 == 0:
                np.savetxt(str(rank) + "_num_elim_per_stage.csv", np.array(num_elim_per_stage), delimiter=",")
                np.savetxt(str(rank) + "_all_rsq.csv", np.array(all_rsq), delimiter=",")

    # for i in range(multiprocessing.cpu_count()):
    #     p = multiprocessing.Process(target=get_sample_paths, args=(i,))
    #     p.start()

    #############################################################

    # det = SimReplication(austin, vaccines, None, -1)
    # det.simulate_time_period(0, det.t_historical_data_end)
    # InputOutputTools.export_rep_to_file(det, "det_sim_rep.json", None,
    #                                     "det_v0.json", "det_v1.json", "det_v2.json", "det_v3.json",
    #                                     "det_epi_params.json")
    #
    # base = SimReplication(austin, vaccines, None, -1)
    # InputOutputTools.load_vars_from_file(base, "det_sim_rep.json", None,
    #                                     "det_v0.json", "det_v1.json", "det_v2.json", "det_v3.json",
    #                                     "det_epi_params.json")



    # # p1 < 14
    # # p1 14
    #
    # grid = [1000 + (i+1)*1000 for i in range(20)]
    # grid.reverse()
    # eps = 1e-6
    #
    # list_of_candidate_policies = []
    #
    # grid = [np.inf]
    #
    # grid2 = [i+1 for i in range(10)]
    # grid2.reverse()
    #
    # for p1 in grid2:
    #     print(p1)
    #     for g in grid:
    #         threshold = (-1, p1, g-eps, g-eps, g)
    #         rep = copy.deepcopy(base)
    #         rep.policy = MultiTierPolicy(austin, tiers, threshold, None)
    #         rep.simulate_time_period(rep.next_t, 945)
    #         if rep.compute_ICU_violation():
    #             print(threshold)
    #             break
    #
    # breakpoint()

    #############################################################

    # thresholds = (-1, 10, 20, 500, 1000)
    # mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
    # old = SimReplication(austin, vaccines, mtp, 2000)
    # old.simulate_time_period(0, 150)
    # start = time.time()
    # InputOutputTools.export_rep_to_file(old,
    #                                     "sim_rep.json",
    #                                     "policy.json",
    #                                     "v0.json",
    #                                     "v1.json",
    #                                     "v2.json",
    #                                     "v3.json",
    #                                     "random_epi_params_historical_period.json")
    # print(time.time() - start)
    # print(old.compute_rsq())
    # old.rng = np.random.RandomState(100)
    # old.simulate_time_period(150, 945)
    # print(old.compute_rsq())
    # print(old.policy.compute_cost())
    #
    # thresholds = (-1, 10, 20, 500, 1000)
    # mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
    # new = SimReplication(austin, vaccines, mtp, 2000)
    # # new.simulate_time_period(0, 150)
    #
    # start = time.time()
    # InputOutputTools.load_vars_from_file(new, "sim_rep.json", "policy.json",
    #                                     "v0.json", "v1.json",
    #                                     "v2.json", "v3.json",
    #                                     "random_epi_params_historical_period.json")
    # print(time.time() - start)
    # print(new.compute_rsq())
    # new.rng = np.random.RandomState(100)
    # new.simulate_time_period(150, 945)
    # print(new.compute_rsq())
    # print(new.policy.compute_cost())
    #
    # breakpoint()
    #
    # for i in range(4):
    #     for attribute in vars(old.vaccine_groups[i]).keys():
    #         print(attribute)
    #         print(getattr(old.vaccine_groups[i], attribute))
    #         print(getattr(new.vaccine_groups[i], attribute))
    #         print("~~~~~~~~~~~~~~~~~")
    #
    # breakpoint()
    #
    # for attribute in vars(old).keys():
    #     print(attribute)
    #     print(getattr(old, attribute))
    #     print(getattr(new, attribute))
    #     print("~~~~~~~~~~~~~~~~~")
    #
    # for attribute in vars(old.epi_rand).keys():
    #     print(attribute)
    #     print(getattr(old.epi_rand, attribute))
    #     print(getattr(new.epi_rand, attribute))
    #     print("~~~~~~~~~~~~~~~~~")
    #
    # breakpoint()
    #
    # # for k in new.state_vars:
    # #     print(k)
    # #     # print(getattr(new, k))
    # #     print(getattr(newnew, k))
    # #     print("~~~~~~~~~~~~~~")

    thresholds = (-1, 100, 200, 500, 1000)
    mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
    rep = SimReplication(austin, vaccines, mtp, 500)
    rep.simulate_time_period(0, 945)
    print(rep.compute_rsq())
    print(rep.policy.compute_cost())