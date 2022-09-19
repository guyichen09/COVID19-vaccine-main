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
from ParamFittingTools import run_fit
from SimModel import SimReplication
import InputOutputTools
import OptTools
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

tiers = TierInfo("austin", "tiers5_opt_Final.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

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

# Specify the 5 thresholds for a 5-tier policy
thresholds = (-1, 100, 200, 500, 1000)

# Create an instance of MultiTierPolicy using
#   austin, tiers (defined above)
#   thresholds (defined above)
#   "green" as the community_transmission toggle
# Prof Morton mentioned that setting community_transmission to "green"
#   was a government official request to stop certain "drop-offs"
#   in active tiers.
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")

# Create an instance of SimReplication with seed 500.
# rep = SimReplication(austin, vaccines, mtp, 500)

# Note that specifying a seed of -1 creates a simulation replication
#   with average values for the "random" epidemiological parameter
#   values and deterministic binomial transitions
#   (also taking average values).

# Advance simulation time until a desired end day.
# Currently, any non-negative integer between 0 and 963 (the length
#   of the user-specified "calendar.csv") works.
# Attributes in the SimReplication instance are updated in-place
#   to reflect the most current simulation state.
# rep.simulate_time_period(800)

# After simulating, we can query the R-squared.
# If the simulation has been simulated for fewer days than the
#   timeframe of the historical time period, the R-squared is
#   computed for this subset of days.
# print(rep.compute_rsq())

# After simulating, we can query the cost of the specified policy.
# print(rep.compute_cost())

# If we want to test the same policy on a different sample path,
#   we can still use the same policy object as long as we clear it.
# mtp.reset()

# Now we create an instance of SimReplication with seed 1010.
# rep = SimReplication(austin, vaccines, mtp, 1010)

# Compare the R-squared and costs of seed 1010 versus seed 500.
# rep.simulate_time_period(800)
# print(rep.compute_rsq())
# print(rep.compute_cost())

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
change_dates = [dt.date(2020, 2, 15),
                        dt.date(2020, 3, 24),
                        dt.date(2020, 5, 21),
                        dt.date(2020, 6, 26),
                        dt.date(2020, 8, 20),
                        dt.date(2020, 10, 29),
                        dt.date(2020, 11, 30),
                        dt.date(2020, 12, 31),
                        dt.date(2021, 1, 12),
                        dt.date(2021, 3, 13),
                        dt.date(2021, 6, 20),
                        dt.date(2021, 7, 31),
                        dt.date(2021, 8, 22),
                        dt.date(2021, 9, 24),
                        dt.date(2021, 10, 25),
                        dt.date(2022, 1, 5),
                        dt.date(2022, 3, 10),
                        dt.date(2022, 4, 4)] 
param1 = 7.3*(1 - 0.10896) + 9.9*0.10896
param2 = (7.3*(1 - 0.10896) + 9.9*0.10896) * 5
initial_guess = np.array([0.6, 0.15, 3.5, 0.002, 0.425, 0.57, 0.68, 0.55])
x_bound = ([0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 10, 1, 1, 1, 1, 1])


run_fit(austin, vaccines, change_dates,x_bound, initial_guess, 1.5, param1 , param2, param2, dt.datetime(2020, 4, 20), dt.datetime(2022, 4, 4))


###############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example B: Stopping and starting a simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Other examples to add -- stay tuned
# Stopping and starting simulation rep within a Python session
# Externally exporting and importing a simulation rep across computers and sessions
# A note on how the instance of EpiSetup is kind of a simulation object
#   and a data object...


