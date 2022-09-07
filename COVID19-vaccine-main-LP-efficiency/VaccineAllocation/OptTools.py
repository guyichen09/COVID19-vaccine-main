# This file is very much under construction and only the function
# get_sample_paths is useable right now.

from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
import multiprocessing
import copy

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

def get_sample_paths(city, vaccine_data, rsq_cutoff, goal_num_good_reps, processor_rank=0, seed_assignment_func=(lambda rank: rank), timepoints=(25, 100, 200, 400, 783)):

    seed = seed_assignment_func(processor_rank)
    rep = SimReplication(city, vaccine_data, None, seed)

    num_good_reps = 0
    total_reps = 0
    num_elim_per_stage = [0, 0, 0, 0, 0]
    all_rsq = []

    while num_good_reps < goal_num_good_reps:
        total_reps += 1
        valid = True
        for i in range(len(timepoints)):
            rep.simulate_time_period(timepoints[i])
            rsq = rep.compute_rsq()
            if rsq < rsq_cutoff:
                num_elim_per_stage[i] += 1
                valid = False
                all_rsq.append(rsq)
                break
        if valid == True:
            num_good_reps += 1
            all_rsq.append(rsq)
            identifier = str(processor_rank) + "_" + str(num_good_reps)
            InputOutputTools.export_rep_to_file(rep,
                                                identifier + "_sim.json",
                                                identifier + "_v0.json",
                                                identifier + "_v1.json",
                                                identifier + "_v2.json",
                                                identifier + "_v3.json",
                                                None,
                                                identifier + "_epi_params.json")
        next_rng = rep.rng
        rep = SimReplication(city, vaccine_data, None, None)

        rep.rng = next_rng
        epi_rand = copy.deepcopy(rep.instance.base_epi)
        epi_rand.sample_random_params(rep.rng)
        epi_rand.setup_base_params()
        rep.epi_rand = epi_rand

        if total_reps % 1000 == 0:
            np.savetxt(str(processor_rank) + "_num_elim_per_stage.csv", np.array(num_elim_per_stage), delimiter=",")
            np.savetxt(str(processor_rank) + "_all_rsq.csv", np.array(all_rsq), delimiter=",")

    # for i in range(multiprocessing.cpu_count()):
    #     p = multiprocessing.Process(target=get_sample_paths, args=(i,))
    #     p.start()

def policy_generator(stage1_info, stage2_info, stage3_info, stage4_info, stage5_info):
    '''
    :param stage1_info: [3-tuple] with elements corresponding to
        start point, end point, and step size (all must be integers)
        for candidate values for stage 1
    :param stage2_info: same as above but for stage 2
    :param stage3_info: same as above but for stage 3
    :param stage4_info: same as above but for stage 4
    :param stage5_info: same as above but for stage 5
    :return:
    '''


