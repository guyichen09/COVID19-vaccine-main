###############################################################################

# main_param_fitting_example.py
# This document contains examples of how to use the least square fitting tool.
# Nazlican Arslan 2022

###############################################################################

from DataObjects import City, TierInfo, Vaccine
from Paramfitting_nazlican import ParameterFitting

# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd

###############################################################################
# We need to define city, vaccine and tier object as the least square fit will
# run the deterministic simulation model.

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

tiers = TierInfo("cook", "tiers5_opt_Final.json")

vaccines = Vaccine(cook,
            "cook",
            "vaccines.json",
            "booster_allocation_fixed_scaled.csv",
            "vaccine_allocation_fixed_scaled.csv")

# We need to define time blocks where the transmission reduction (or the behaviour in the population) changes:
change_dates = [dt.date(2020, 2, 17),
                        dt.date(2020, 3, 20),
                        dt.date(2020, 4, 1), 
                        dt.date(2020, 4, 24),
                        # dt.date(2020, 4, 10),
                        # dt.date(2020, 5, 15),
                        # dt.date(2020, 6, 1),
                        dt.date(2020, 6, 13),
                       ] 

# We don't fit all the transmission reduction values from scratch as the least square fit cannot handle too many
# decision variables. We used the existing fitted values for the earlier fit data. In transmission_reduction and
# cocoon lists you can input the already existing values and for the new dates you need to estimate just input None.
transmission_reduction = [
                          None,
                          None,
                          None,
                          None]
# for the high risk groups uses cocoon instead of contact reduction
cocoon = np.array([
                   None,
                   None,
                   None,
                   None])

end_date = []
for idx in range(len(change_dates[1:])):
    end_date.append(str(change_dates[1:][idx] - dt.timedelta(days=1)))

table = pd.DataFrame(
    {
        "start_date": change_dates[:-1],
        "end_date": end_date,
    }
)
# The initial guess of the variables to estimate:
initial_guess = np.array([0.073749, 0.296143, 1.80139, 0.003, 0.75, 0.85, 0.75, 0.75])
# Lower and upper bound tuple:
x_bound = ([0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 10, 1, 1, 1, 1, 1])

# Austin weights for the least-square fit:
# You can input the data you would like to use in the process and corresponding weights. Different data have different
# scales that's why we use weights.
objective_weights = {"IH_history": 1,
                     "ICU_history": 2.5,
                     "ToIHT_history": 10,
                     "ToICUD_history": 0,
                     "ToIYD_history": 0}

# We generally use the least square fit to find transmission reduction and cocooning in a population. But time to time
# we may need to estimate other parameters. Fitting transmission reduction is optional. In the current version of
# the parameter fitting you can input the name of parameter you would like to fit, and you don't need to change anything
# else in the source code.
variables = ["transmission_reduction"]

# We can define the time frame we would like to use data from as follows:
time_frame = (cook.cal.calendar.index(dt.datetime(2020, 2, 17)), cook.cal.calendar.index(dt.datetime(2020, 6, 1)))
print(time_frame)
param_fitting = ParameterFitting(cook,
                                 vaccines,
                                 variables,
                                 initial_guess,
                                 x_bound,
                                 objective_weights,
                                 time_frame,
                                 change_dates,
                                 transmission_reduction,
                                 cocoon)
solution = param_fitting.run_fit()