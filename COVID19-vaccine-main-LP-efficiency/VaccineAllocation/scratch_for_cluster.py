from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from OptTools import thresholds_generator
from InputOutputTools import import_rep_from_json

# Import other Python packages
import numpy as np
from mpi4py import MPI
import csv
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
num_workers = size - 1
master_rank = size - 1

###############################################################################

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

###############################################################################

# In this code, "threshold" refers to a 5-tuple of the thresholds for a policy
#   and "policy" is an instance of MultiTierPolicy -- there's a distinction
#   between the identifier for an object versus the actual object.

def evaluate_policies_on_sample_paths(thresholds_array, RNG, num_reps, base_filename, processor_rank,
                                      processor_count_total):

    policies_array = np.array([MultiTierPolicy(austin, tiers, thresholds, "green") for thresholds in thresholds_array])
    num_policies = len(policies_array)
    num_policies_per_processor = int(np.floor(num_policies / processor_count_total))

    policies_ix_processor = np.arange(processor_rank * num_policies_per_processor,
                                      (processor_rank + 1) * num_policies_per_processor)
    if processor_rank == processor_count_total - 1:
        policies_ix_processor = np.arange(processor_rank * num_policies_per_processor,
                                          processor_rank * num_policies_per_processor + num_policies % processor_count_total)

    if rank == 0:
        print(num_policies_per_processor)
        start = time.time()

    for rep in range(num_reps):
        base_identifier = base_filename + str(rep+1) + "_"
        base_rep = SimReplication(austin, vaccines, None, 1)
        import_rep_from_json(base_rep, base_identifier + "sim.json",
                             base_identifier + "v0.json",
                             base_identifier + "v1.json",
                             base_identifier + "v2.json",
                             base_identifier + "v3.json",
                             None,
                             base_identifier + "epi_params.json")
        if rep == 0:
            base_rep.rng = RNG

        thresholds_identifiers = []
        costs_data = []
        feasibility_data = []

        for policy in policies_array[policies_ix_processor]:
            base_rep.policy = policy
            base_rep.simulate_time_period(945)

            thresholds_identifiers.append(base_rep.policy.lockdown_thresholds)
            costs_data.append(base_rep.compute_cost())
            feasibility_data.append(base_rep.compute_feasibility())

            base_rep.policy.reset()
            base_rep.reset()

        thresholds_identifiers = np.array(thresholds_identifiers)
        costs_data = np.array(costs_data)
        feasibility_data = np.array(feasibility_data)

        base_csv_filename = "proc" + str(processor_rank) + "_rep" + str(rep+1) + "_"
        np.savetxt(base_csv_filename + "thresholds_identifiers.csv", thresholds_identifiers, delimiter=",")
        np.savetxt(base_csv_filename + "costs_data.csv", costs_data, delimiter=",")
        np.savetxt(base_csv_filename + "feasibility_data.csv", feasibility_data, delimiter=",")

        if rank == 0:
            print(time.time() - start)

thresholds_array = thresholds_generator((0,14,1), (0,100,10), (0,100,10), (0,100,10))
# thresholds_array = [(-1, 1, 10, 100, 200), (-1, 5, 15, 25, 50)]
rng = np.random.RandomState(100+rank)
base_identifier = "0_"
num_reps = 300

evaluate_policies_on_sample_paths(thresholds_array, rng, num_reps, base_identifier, rank, size)