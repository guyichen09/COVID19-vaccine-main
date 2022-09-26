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

cook = City("cook",
              "cook_test_IHT.json",
              "calendar.csv",
              "setup_data_param_fit.json",
              "transmission_lsq_test.csv",
              "hosp.csv",
              "icu.csv",
              "admission.csv",
              "cook_deaths_from_hosp_est.csv",
              "cook_deaths.csv",
              "delta_prevalence.csv",
              "omicron_prevalence.csv",
              "variant_prevalence.csv")

tiers = TierInfo("cook", "tiers5_opt_Final.json")

vaccines = Vaccine(cook,
                   "cook",
                   "vaccines.json",
                   "booster_allocation_fixed_scaled.csv",
                   "vaccine_allocation_fixed_scaled.csv")

###############################################################################

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


# Note that specifying a seed of -1 creates a simulation replication
#   with average values for the "random" epidemiological parameter
#   values and deterministic binomial transitions
#   (also taking average values).
rep = SimReplication(cook, vaccines, None, None)

# Advance simulation time until a desired end day.
# Currently, any non-negative integer between 0 and 963 (the length
#   of the user-specified "calendar.csv") works.
# Attributes in the SimReplication instance are updated in-place
#   to reflect the most current simulation state.
rep.simulate_time_period(155)

# After simulating, we can query the R-squared.
# If the simulation has been simulated for fewer days than the
#   timeframe of the historical time period, the R-squared is
#   computed for this subset of days.
print(rep.compute_rsq())

# After simulating, we expert it to json file
export_rep_to_json(rep, "output.json", "v0.json", "v1.json", "v2.json", "v3.json")


# If we want to test the same policy on a different sample path,
#   we can still use the same policy object as long as we clear it.
# mtp.reset()

plot_from_file("output.json", cook)


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
change_dates = [dt.date(2020, 1, 10),
                        dt.date(2020, 3, 24),
                        dt.date(2020, 4, 12),
                        dt.date(2020, 6, 13),
                       ]  
param1 = 7.3*(1 - 0.10896) + 9.9*0.10896
param2 = (7.3*(1 - 0.10896) + 9.9*0.10896) * 5
initial_guess = np.array([0.02, 0.73, 0.83, 0.75, 0.65, 0.5, 0.65])
x_bound = ([ 0, 0, 0, 0,0, 0, 0],
                                     [ 1, 1, 1, 1, 1, 1, 1])


# transmission = run_fit(cook, vaccines, change_dates,x_bound, initial_guess, 4.8, 17, 150, 150, dt.datetime(2020, 3, 29), dt.datetime(2020, 6, 13))

# save_output(transmission, cook)
###############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example B: Stopping and starting a simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Other examples to add -- stay tuned
# Stopping and starting simulation rep within a Python session
# Externally exporting and importing a simulation rep across computers and sessions
# A note on how the instance of EpiSetup is kind of a simulation object
#   and a data object...


