###############################################################################

# Examples.py
# This document contains examples of how to use the simulation code.

# To launch the examples, either
# (1) Run the following command in the OS command line or Terminal:
#   python3 Examples.py
# (2) Copy and paste the code of this document into an interactive
#   Python console.

# Note that if modules cannot be found, this is a path problem.
# In both fixes below, replace <NAME_OF_YOUR_DIRECTORY> with a string
#   containing the directory in which the modules reside.
# (1) The following path can be updated via the OS command line or
#   Terminal (e.g. on a cluster):
#   export PYTHONPATH=<NAME_OF_YOUR_DIRECTORY>:$PYTHONPATH
# (2) The following code can be added to the main .py script file
#   (e.g. can be added to the top of this document):
#   import sys
#   sys.path.append(<NAME_OF_YOUR_DIRECTORY>)

# Linda Pei 2022

###############################################################################

# Import other code modules
# SimObjects contains classes of objects that change within a simulation
#   replication.
# DataObjects contains classes of objects that do not change
#   within simulation replications and also do not change *across*
#   simulation replications -- they contain data that are "fixed"
#   for an overall problem.
# SimModel contains the class SimReplication, which runs
#   a simulation replication and stores the simulation data
#   in its object attributes.
# InputOutputTools contains utility functions that act on
#   instances of SimReplication to load and export
#   simulation states and data.
# OptTools contains utility functions for optimization purposes.

import copy
from faulthandler import is_enabled
from os import listdir
from matplotlib import pyplot as plt

from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from ParamFittingTools import run_fit, save_output
from SimModel import SimReplication
from InputOutputTools import export_rep_to_json
import OptTools
from Plotting import plot_from_file, plot_debug
import datetime as dt

# Import other Python packages
import numpy as np

###############################################################################

# Mandatory definitions from user-input files
# In general, these following lines must be in every script that uses
#   the simulation code. The specific names of the files used may change.
# Each simulation replication requires these 3 instances
#   (built from reading .json and .csv files)
# (1) City instance that holds calendar, city-specific epidemiological
#   parameters, historical hospital data, and fitted transmission parameters
# (2) TierInfo instance that contains information about the tiers in a
#   social distancing threshold policy
# (3) Vaccine instance that holds vaccine groups and historical
#   vaccination data

cook = City("cook",
              "cook_test_IHT.json",
              "calendar.csv",
              "setup_data_param_fit_final.json",
              "transmission_lsq_estimated_data.csv",
              "cook_hosp_region_sum_estimated.csv",
              "cook_icu_estimated_adjusted.csv",
              "sensitive_line_list_admission_processed_7_day.csv",
              "cook_deaths_from_hosp_est.csv",
              "cook_deaths.csv",
              "delta_prevalence.csv",
              "omicron_prevalence.csv",
              "variant_prevalence.csv")

# tiers = TierInfo("cook", "tiers5_opt_Final.json")

vaccines = Vaccine(cook,
                   "cook",
                   "vaccines.json",
                   "booster_allocation_fixed_scaled.csv",
                   "vaccine_allocation_fixed_scaled.csv")

###############################################################################

austin = City("austin",
              "austin_test_IHT.json",
              "calendar.csv",
              "setup_data_Final.json",
              "transmission.csv",
              "austin_real_hosp_updated.csv",
              "austin_real_icu_updated.csv",
              "austin_hosp_ad_updated.csv",
              "austin_real_death_from_hosp_updated.csv",
              "austin_real_total_death.csv",
              "delta_prevalence.csv",
              "omicron_prevalence.csv",
              "variant_prevalence.csv")

# tiers = TierInfo("austin", "tiers5_opt_Final.json")

vaccines_austin = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example A: Simulating a threshold policy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In general, simulating a policy requires the steps
# (1) Create a MultiTierPolicy instance with desired thresholds.
# (2) Create a SimReplication instance with  aforementioned policy
#   and a random number seed -- this seed dictates the randomly sampled
#   epidemiological parameters for that replication as well as the
#   binomial random variable transitions between compartments in the
#   SEIR-type model.
# (3) Advance simulation time.


rep = SimReplication(cook, vaccines, None, None)
rep_austin = SimReplication(austin, vaccines_austin, None, None)

# Advance simulation time until a desired end day.
# Currently, any non-negative integer between 0 and 963 (the length
#   of the user-specified "calendar.csv") works.
# Attributes in the SimReplication instance are updated in-place
#   to reflect the most current simulation state.
# rep.simulate_time_period(155)
# rep_austin.simulate_time_period(154)
# After simulating, we can query the R-squared.
# If the simulation has been simulated for fewer days than the
#   timeframe of the historical time period, the R-squared is
#   computed for this subset of days.
# print(rep.compute_rsq())

# After simulating, we expert it to json file
# export_rep_to_json(rep, "./output/cook/output_cook_admission_param_fit_ToIH.json", "v0_a.json", "v1_a.json", "v2_a.json", "v3_a.json")
# export_rep_to_json(rep_austin, "output_austin.json", "v0_a.json", "v1_a.json", "v2_a.json", "v3_a.json")


# If we want to test the same policy on a different sample path,
#   we can still use the same policy object as long as we clear it.
# mtp.reset()

# plot_from_file("output_austin.json","output_austin.json",austin)

plot_from_file("./output/cook/output_cook_admission_param_fit_ToIH.json", "output_cook_admission_param_fit.json", cook)
# icu_rsq = []
# # for rIH in np.linspace(0.70, 0.736, 15):
# rIH = 0.712
# for alpha_init in np.linspace(0.5, 0.7, 20):
#     # alpha_init = 0.67777778
#     alpha = None
#     hicur = None
#     cook = City("cook",
#                 "cook_test_IHT.json",
#                 "calendar.csv",
#                 "setup_data_param_fit_3.json",
#                 "transmission_lsq_estimated_data.csv",
#                 "cook_hosp_region_sum_estimated.csv",
#                 "cook_icu_estimated_adjusted.csv",
#                 "sensitive_line_list_admission_processed_7_day.csv",
#                 "cook_deaths_from_hosp_est.csv",
#                 "cook_deaths.csv",
#                 "delta_prevalence.csv",
#                 "omicron_prevalence.csv",
#                 "variant_prevalence.csv", alpha, alpha_init, False, hicur, rIH)
#     vaccines = Vaccine(cook,
#                             "cook",
#                             "vaccines.json",
#                             "booster_allocation_fixed_scaled.csv",
#                             "vaccine_allocation_fixed_scaled.csv")
#     rep = SimReplication(cook, vaccines, None, None)
#     rep.simulate_time_period(155)
#     # rsq = rep.compute_icu_rsq()
#     rsq = rep.compute_gw_rsq()
#     mu = []
#     for arr in cook.base_epi._mu:
#         if arr[3] < 0:
#             is_negative = True
#             break
#         # if arr[3] > 10:
#         #     is_greater_than_10 = True
#         #     break
#         mu.append(arr[3])
#     print("====================================")
#     print(rsq)
#     icu_rsq.append(rsq)
#     # if rsq > 0.80:
#     outfile_dir = "./output/cook/"
#     out_file = "gw_rsq=" + str(rsq)[:7] + "_a_init=" + str(alpha_init)[:5] + "_" + "a=" + str(alpha)[:5] + "_" + "rIH=" + str(rIH)[:5] +"_" + "hicur=" + str(hicur)[:5] + "_" + "mu=" + str(mu)[:10] + ".json"
#     out_file_path = outfile_dir + out_file

#     export_rep_to_json(rep, out_file_path, "v0_a.json", "v1_a.json", "v2_a.json", "v3_a.json")
# # if out_file in listdir(outfile_dir):
#     plot_from_file(out_file_path, out_file, cook)

# plt.plot(np.linspace(0.5, 0.7, 20), icu_rsq)
# plt.plot(np.linspace(0.70, 0.736, 15), icu_rsq)
# plt.show()


# # # alpha = None
# rIH = 0.712
# alpha_init = 0.647
# # for hicur in np.linspace(0.05, 0.15, 50):
# for hicur in [0.05]:
#     for alpha in np.linspace(0.5, 0.67, 100):
#         cook = City("cook",
#                 "cook_test_IHT.json",
#                 "calendar.csv",
#                 "setup_data_param_fit_3.json",
#                 "transmission_lsq_estimated_data.csv",
#                 "cook_hosp_region_sum_estimated.csv",
#                 "cook_icu_estimated_adjusted.csv",
#                 "sensitive_line_list_admission_processed_7_day.csv",
#                 "cook_deaths_from_hosp_est.csv",
#                 "cook_deaths.csv",
#                 "delta_prevalence.csv",
#                 "omicron_prevalence.csv",
#                 "variant_prevalence.csv", alpha, alpha_init, True, hicur, rIH)
#         is_negative = False
#         is_greater_than_10 = False
#         mu = []
#         for arr in cook.base_epi._mu:
#             if arr[3] < 0:
#                 is_negative = True
#                 break
#             # if arr[3] > 10:
#             #     is_greater_than_10 = True
#             #     break
#             mu.append(arr[3])
#         if is_negative is False and is_greater_than_10 is False:
#             vaccines = Vaccine(cook,
#                             "cook",
#                             "vaccines.json",
#                             "booster_allocation_fixed_scaled.csv",
#                             "vaccine_allocation_fixed_scaled.csv")
#             rep = SimReplication(cook, vaccines, None, None)
#             rep.simulate_time_period(155)
#             rsq = rep.compute_icu_rsq()
#             gw_rsq = rep.compute_gw_rsq()
#             print("====================================")
#             print(rsq)
#             if rsq > 0.80:
#                 outfile_dir = "./output/cook/"
#                 out_file = "f_icu_rsq=" + str(rsq)[:5] + "_gw_rsq=" + str(gw_rsq)[:5] + "_a_init=" + str(alpha_init)[:5] + "_" + "a=" + str(alpha)[:5] + "_" + "hicur=" + str(hicur)[:5] + "_" + "mu=" + str(mu)[:10] + ".json"
#                 out_file_path = outfile_dir + out_file

#                 export_rep_to_json(rep, out_file_path, "v0_a.json", "v1_a.json", "v2_a.json", "v3_a.json")
#             # if out_file in listdir(outfile_dir):
#                 plot_from_file(out_file_path, out_file, cook)


# outfile_dir = "./output/cook/"
# out_file = "rsq=0.8565463284164533_a_init=0.67777778_a=0.657979797979798_hicur=0.07040816326530612_mu=[10.226774327291405, 10.226774327291405, 10.908562246726644, 10.908562246726644, 11.931237011601608].json"
# out_file1 = "rsq=0.8732794733645752_a_init=0.67777778_a=0.6682828282828284_hicur=0.05204081632653061_mu=[12.575959548985267, 12.575959548985267, 13.414360497456638, 13.414360497456638, 14.67195317167005].json"
# out_file_path = outfile_dir + out_file
# out_file_path1 = outfile_dir + out_file1

# plot_debug(out_file_path, out_file_path1, out_file,cook)

###############################################################################