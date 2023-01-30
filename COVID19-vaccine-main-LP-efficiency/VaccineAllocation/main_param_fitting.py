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

from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from ParamFittingTools import run_fit, save_output
from SimModel import SimReplication
from InputOutputTools import export_rep_to_json
import OptTools
from Plotting import plot_from_file
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

# for c in range(0, 1):
c = 0
cocoon_factor = 1 + c / 100
transmission_file_name = "transmission_lsq_" + str(cocoon_factor) + "_apr" + ".csv"

cook = City("cook",
              "cook_test_IHT.json",
              "calendar.csv",
              "setup_data_param_fit_YHR.json",
              "transmission_lsq_1.0_apr.csv",
              "cook_hosp_region_sum_estimated.csv",
              "cook_icu_estimated_adjusted.csv",
              "sensitive_line_list_admission_processed.csv",
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



rep = SimReplication(cook, vaccines, None, None)
# rep_austin = SimReplication(austin, vaccines_austin, None, None)

# Advance simulation time until a desired end day.
# Currently, any non-negative integer between 0 and 963 (the length
#   of the user-specified "calendar.csv") works.
# Attributes in the SimReplication instance are updated in-place
#   to reflect the most current simulation state.
rep.simulate_time_period(115)
# rep_austin.simulate_time_period(154)
# After simulating, we can query the R-squared.
# If the simulation has been simulated for fewer days than the
#   timeframe of the historical time period, the R-squared is
#   computed for this subset of days.
# print(rep.compute_rsq())

# After simulating, we expert it to json file

# export_rep_to_json(rep, "./output/cook/ToPY_" + transmission_file_name[:-4] + ".json", "v0_a.json", "v1_a.json", "v2_a.json", "v3_a.json")
# # export_rep_to_json(rep_austin, "output_austin.json", "v0_a.json", "v1_a.json", "v2_a.json", "v3_a.json")


# # If we want to test the same policy on a different sample path,
# #   we can still use the same policy object as long as we clear it.
# # mtp.reset()

# plot_from_file("./output/cook/" + "ToPY_transmission_lsq_1.0_apr" + ".json", "ToPY_transmission_lsq_1.0_apr" + ".json", cook)



# Note that calling rep.compute_rsq() if rep has not yet
#   been simulated, or it has been cleared, leads to an error.
# Calling rep.compute_cost() if there is no policy attached
#   to the replication (i.e., if rep.policy = None) leads to an error.
#   Similarly, we also need to simulate rep (with a policy attached)
#   *past* the historical time period so that the policy goes
#   into effect before querying its cost.

# Clearing an instance of SimReplication is a bit tricky, so
#   be careful of this nuance. The following reset() method
#   clears the replication ("zero-ing" any saved data
#   as well as the current time).
# However, the randomly sampled parameters remain the same!
#   These are untouched.
# The random number generator is also untouched after
#   reset(), so simulating rep will draw random numbers
#   from where the random number generator last left off
#   (before the reset).
# rep.reset()


###############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example B: Parameter fitting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
change_dates = [dt.date(2020, 2, 17),
                        dt.date(2020, 3, 20),
                        dt.date(2020, 4, 1), 
                        dt.date(2020, 4, 24),
                        # dt.date(2020, 4, 10),
                        # dt.date(2020, 5, 15),
                        # dt.date(2020, 6, 1),
                        dt.date(2020, 6, 13),
                       ]  

change_dates_austin = [dt.date(2020, 2, 15),
                        dt.date(2020, 3, 24),
                        dt.date(2020, 5, 21),
                        dt.date(2020, 6, 26),]

param1 = 7.3*(1 - 0.10896) + 9.9*0.10896
param2 = (7.3*(1 - 0.10896) + 9.9*0.10896) * 5
# initial_guess = np.array([0.5, 0.85, 0.6, 0.5, 0.5, 0.85,])
# x_bound = ([ 0, 0, 0, 0, 0, 0, 0, 0,],
#                                      [ 1, 1, 1, 1, 1, 1,1, 1,])

initial_guess = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
x_bound = ([ 0, 0, 0, 0, 0],
                                     [ 1, 1, 1, 1, 1],)

# initial_guess = np.array([ 14, 15, 15, 18, 20 ])
# x_bound = ([ 10, 10, 10, 10, 10],
#                                      [ 23, 23, 23, 23, 23])

# initial_guess = np.array([ 11, 12, 13, 13, 14 ])
# x_bound = ([ 10, 10, 10, 10, 10],
#                                      [ 20, 20, 20, 20, 20])


# for c in range(0, 1):
c = 0
cocoon_factor = 1 + c / 100
# transmission = run_fit(cook, vaccines, change_dates, x_bound, initial_guess, 1, 2.5, 10, 0, 0, cocoon_factor, dt.datetime(2020, 2, 17), dt.datetime(2020, 6, 13))
# save_output(transmission, str(cocoon_factor) + "_apr_debug", cook)

# # transmission = run_fit(austin, vaccines_austin, change_dates_austin,x_bound, initial_guess, 1.5, param1, param2, param2, dt.datetime(2020, 2, 28), dt.datetime(2020, 6, 26))
# # save_output(transmission, austin)
###############################################################################



