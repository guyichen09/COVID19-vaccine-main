from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from OptTools import thresholds_generator, evaluate_policies_on_sample_paths
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

###############################################################################s

# thresholds_array = thresholds_generator((0, 14, 1), (0, 100, 10), (0, 100, 10), (0, 100, 20))

thresholds_array = thresholds_generator((0, 1000, 100), (0, 1000, 100), (0, 1000, 100), (0, 1000, 100))

new_thresholds_array = []

for threshold in thresholds_array:
    if threshold[1] == threshold[2] and threshold[3] == threshold[4]:
        new_thresholds_array.append(threshold)

base_identifier = "0_"
num_reps = 1

# breakpoint()

evaluate_policies_on_sample_paths(austin, tiers, vaccines, thresholds_array, 945,
                                  tuple(np.full(len(thresholds_array), -1)), num_reps, base_identifier, rank, size)

breakpoint()
