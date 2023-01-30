###############################################################################

# Examples.py
# This document contains examples of how to use the simulation code.

# To launch the examples, either
# (1) Run the following command in the OS command line or Terminal:
#   python3 Examples.py
# (2) Copy and paste the code of this document into an interactive
#   Python console.

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
from OptTools import get_sample_paths
from Plotting import plot_from_file, plot_debug
import datetime as dt
import json 
# Import other Python packages
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import binom
import pandas as pd
import math
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

def linear_regression(simulation_filename, x_var_names, y_var_name):
    # plt.clf()
    # fig, ax = plt.subplots()

    total_x_vars = []
    total_y_vars = []
    total_risk_age_x_vars = []
    total_risk_age_y_vars = []
    for lead_time in [5]:
        if "cook" in simulation_filename:
            dir = "cook/"
        elif "austin" in simulation_filename:
            dir = "austin/"
        # with open(simulation_filename) as file:
        #     data = json.load(file)

        risk_age_x_vars = []
        risk_age_y_vars = []
        fig, ax = plt.subplots()
        with open("./output/cook/ToPY_transmission_lsq_1.0_apr.json") as file:
            data = json.load(file)
        for x_var_name in x_var_names:
            risk_age_x_vars = data[x_var_name]
            x_vars = np.sum(np.sum(risk_age_x_vars, axis=1), axis=1)
        risk_age_y_vars = data[y_var_name]
        y_vars = np.sum(np.sum(risk_age_y_vars, axis=1), axis=1)
        if len(x_vars) != len(y_vars):
            y_vars = y_vars[1:]
        x_vars = x_vars[:-lead_time]
        risk_age_x_vars = risk_age_x_vars[:-lead_time]
        y_vars = y_vars[lead_time:]
        risk_age_y_vars = risk_age_y_vars[lead_time:]
        total_x_vars.extend(x_vars)
        total_y_vars.extend(y_vars)
        total_risk_age_x_vars.extend(risk_age_x_vars)
        total_risk_age_y_vars.extend(risk_age_y_vars)
        ax.scatter(x_vars, y_vars, s= 0.5)
        ax.set_xlabel(x_var_name)
        ax.set_ylabel(y_var_name)

        # for i in range(1, 301):
        #     with open("./output/cook/sample_paths_2323/2323_" + str(i) + "_sim.json") as file:
        #         data = json.load(file)
        #     for x_var_name in x_var_names:
        #         risk_age_x_vars = data[x_var_name]
        #         x_vars = np.sum(np.sum(risk_age_x_vars, axis=1), axis=1)
        #     risk_age_y_vars = data[y_var_name]
        #     y_vars = np.sum(np.sum(risk_age_y_vars, axis=1), axis=1)
        #     if len(x_vars) != len(y_vars):
        #         y_vars = y_vars[1:]
        #     x_vars = x_vars[:-lead_time]
        #     risk_age_x_vars = risk_age_x_vars[:-lead_time]
        #     y_vars = y_vars[lead_time:]
        #     risk_age_y_vars = risk_age_y_vars[lead_time:]
        #     total_x_vars.extend(x_vars)
        #     total_y_vars.extend(y_vars)
        #     total_risk_age_x_vars.extend(risk_age_x_vars)
        #     total_risk_age_y_vars.extend(risk_age_y_vars)
        #     ax.scatter(x_vars, y_vars, s= 0.5)
        # ax.set_xlabel(x_var_name)
        # ax.set_ylabel(y_var_name)
        
        
        total_x_vars = np.array(total_x_vars).reshape(-1, 1)
        model = LinearRegression().fit(total_x_vars, total_y_vars)
        r_sq = model.score(total_x_vars, total_y_vars)
        print('coefficient of determination:', r_sq)

        # Print the Intercept:
        print('intercept:', model.intercept_)

        # Print the Slope:
        print('slope:', model.coef_) 

        predicted_ys = model.coef_[0] * total_x_vars + model.intercept_

        for age in range(5):
            for risk in range(2):
                fig, ax = plt.subplots()
                # print(total_risk_age_x_vars)
                x_vars = np.array(np.array(total_risk_age_x_vars)[:, age, risk]).reshape(-1, 1)
                model = LinearRegression().fit(x_vars, np.array(total_risk_age_y_vars)[:,age, risk])
                r_sq = model.score(x_vars, np.array(total_risk_age_y_vars)[:, age, risk])
                print("age:", age, "risk:", risk)
                print('coefficient of determination:', r_sq)
                print("hospital admission", np.sum(np.array(total_risk_age_y_vars)[:,age, risk]))
                print("PY", np.sum(x_vars))
                # Print the Intercept:
                print('intercept:', model.intercept_)

                # Print the Slope:
                print('slope:', model.coef_) 

                predicted_ys = model.coef_[0] * x_vars + model.intercept_

                ax.plot(x_vars, predicted_ys, color="red")
                ax.scatter(x_vars, np.array(total_risk_age_y_vars)[:,age, risk])
                ax.set_title("age" + str(age) + " risk" + str(risk))
                plt.savefig("age" + str(age) + "risk " + str(risk) + "_300_samples.png")



        # sum_errs = np.sum((total_y_vars - predicted_ys)**2)
        # stdev = np.sqrt(1/(len(total_y_vars)-2) * sum_errs)
        # # calculate prediction interval
        # interval = 1.28 * stdev
        # low_ys = predicted_ys - interval
        # upper_ys = predicted_ys + interval
        df = pd.DataFrame({"PY": np.array(total_x_vars).reshape(33600), "ToIHT": list(total_y_vars)}, index=None)
        df.to_csv("py_toiht.csv")
        
        low_ys = []
        upper_ys = []
        low_counter = 0
        upper_counter = 0
        for idx, x in enumerate(total_x_vars):
            low_y = binom.ppf(0.05, x, model.coef_[0])
            upper_y = binom.ppf(0.95, x, model.coef_[0])
            if total_y_vars[idx] >= low_y:
                low_counter += 1
            if total_y_vars[idx] <= upper_y:
                upper_counter += 1

            low_ys.append(low_y)
            upper_ys.append(upper_y)
        print("low counter", low_counter)
        print("upper counter", upper_counter)
        
        ax.plot(total_x_vars, predicted_ys, color="white")
        ax.plot(total_x_vars, low_ys, color="white")
        ax.plot(total_x_vars, upper_ys, color="white")


        print("lower line slope:", (max(low_ys) - min(low_ys)) / (max(total_x_vars) - min(total_x_vars)))
        print("upper line slope:", (max(upper_ys) - min(upper_ys)) / (max(total_x_vars) - min(total_x_vars)))

        plt.savefig(x_var_name + "_vs_" + y_var_name + str(lead_time) +  "_300_samples.png")
        # plt.show()
        plt.close()
def empirical_90_pi_lr(simulation_filename, x_var_names, y_var_name):
    total_x_vars = []
    total_y_vars = []
    for lead_time in [5]:
        if "cook" in simulation_filename:
            dir = "cook/"
        elif "austin" in simulation_filename:
            dir = "austin/"
        # with open(simulation_filename) as file:
        #     data = json.load(file)
        fig, ax = plt.subplots()
        x_y_pairs = []
        for i in range(1, 301):
            with open("./output/cook/sample_paths_2323/2323_" + str(i) + "_sim.json") as file:
                data = json.load(file)
            for x_var_name in x_var_names:
                x_vars = data[x_var_name]
                x_vars = np.sum(np.sum(x_vars, axis=1), axis=1)
            y_vars = data[y_var_name]
            y_vars = np.sum(np.sum(y_vars, axis=1), axis=1)
            if len(x_vars) != len(y_vars):
                y_vars = y_vars[1:]
            x_vars = x_vars[:-lead_time]
            y_vars = y_vars[lead_time:]
            total_x_vars.extend(x_vars)
            total_y_vars.extend(y_vars)
            x_y_pairs.extend((a, b) for a, b in zip(x_vars,y_vars))
            ax.scatter(x_vars, y_vars, s=0.5)
        ax.set_xlabel("daily pre-symptomatic individuals")
        ax.set_ylabel("daily hospital admission")
        
        
        total_x_vars = np.array(total_x_vars).reshape(-1, 1)
        model = LinearRegression().fit(total_x_vars, total_y_vars)
        r_sq = model.score(total_x_vars, total_y_vars)
        print('coefficient of determination:', r_sq)

        # Print the Intercept:
        print('intercept:', model.intercept_)

        # Print the Slope:
        print('slope:', model.coef_) 
        
        x_vars_less_11000 = [x for x in total_x_vars if x < 11000]
        x_vars_less_11000 = np.array(x_vars_less_11000).reshape(-1, 1)
        predicted_ys = model.coef_[0] * x_vars_less_11000 + model.intercept_

        # sum_errs = np.sum((total_y_vars - predicted_ys)**2)
        # stdev = np.sqrt(1/(len(total_y_vars)-2) * sum_errs)
        # # calculate prediction interval
        # interval = 1.28 * stdev
        # low_ys = predicted_ys - interval
        # upper_ys = predicted_ys + interval
        df = pd.DataFrame({"PY": np.array(total_x_vars).reshape(33600), "ToIHT": list(total_y_vars)}, index=None)
        df.to_csv("py_toiht.csv")
        ax.plot(x_vars_less_11000, predicted_ys, color="green", label="Linear regression fit")
        total_x_vars = list(total_x_vars)
        # x_y_pairs = [(x, y) for x, y in zip(total_x_vars, total_y_vars)]
        sorted_x_pairs = sorted(x_y_pairs)
        bin_width = 50
        num_samples = len(total_x_vars) / bin_width
        five_percentile = math.floor(num_samples * 0.10)
        low_xs = []
        low_ys = []
        upper_xs = []
        upper_ys = []
        for i in range(bin_width):
            samples = sorted_x_pairs[int(i * num_samples) : int((i + 1) * num_samples)]
            assert(len(samples) == num_samples)
            sorted_y_pairs = sorted(samples, key=lambda tup: tup[1])
            # print(i, sorted_y_pairs[:five_percentile])
            low_y = sorted_y_pairs[five_percentile]
            high_y = sorted_y_pairs[-five_percentile]
            low_xs.append(low_y[0])
            low_ys.append(low_y[1])
            upper_xs.append(high_y[0])
            upper_ys.append(high_y[1])
        
        ax.plot(low_xs[:], low_ys[:], color="black", label="Upper 90% empirical prediction")
        ax.plot(upper_xs[:], upper_ys[:], color="black", label="Lower 90% empirical prediction")
        ax.set_title("Pre-symptomatic vs. hospital admission with 5-day lead time")
        ax.legend()
        # print("lower line slope:", (max(low_ys) - min(low_ys)) / (max(total_x_vars) - min(total_x_vars)))
        # print("upper line slope:", (max(upper_ys) - min(upper_ys)) / (max(total_x_vars) - min(total_x_vars)))

        plt.savefig(x_var_name + "_vs_" + y_var_name + str(lead_time) + "_" + str(bin_width) + "_300_samples_95pi.png")
        # plt.show()
        plt.close()

def empirical_yhr(simulation_filename):
    total_risk_age_x_vars = []
    total_risk_age_y_vars = []
    for lead_time in [5]:
        if "cook" in simulation_filename:
            dir = "cook/"
        elif "austin" in simulation_filename:
            dir = "austin/"
        with open(simulation_filename) as file:
            data = json.load(file)
        risk_age_x_vars = data["PY_history"]
        risk_age_y_vars = data["ToIHT_history"]
        for age in range(5):
            for risk in range(2):
                print("deterministic path")
                print("age:", age, "risk:", risk)
                x_vars = np.array(np.array(risk_age_x_vars)[:, age, risk]).reshape(-1, 1)
                mean_x_vars = np.mean(x_vars)
                y_vars = np.array(np.array(risk_age_y_vars)[:, age, risk]).reshape(-1, 1)
                mean_y_vars = np.mean(y_vars)
                print("empirical yhr:", mean_y_vars / mean_x_vars)

        data = None
        risk_age_x_vars = []
        risk_age_y_vars = []
        fig, ax = plt.subplots()
        for i in range(1, 301):
            with open("./output/cook/sample_paths_2323/2323_" + str(i) + "_sim.json") as file:
                data = json.load(file)
            
            risk_age_x_vars = data["PY_history"]
            x_vars = np.sum(np.sum(risk_age_x_vars, axis=1), axis=1)
            risk_age_y_vars = data["ToIHT_history"]
            y_vars = np.sum(np.sum(risk_age_y_vars, axis=1), axis=1)
            if len(x_vars) != len(y_vars):
                y_vars = y_vars[1:]
            x_vars = x_vars[:-lead_time]
            risk_age_x_vars = risk_age_x_vars[:-lead_time]
            y_vars = y_vars[lead_time:]
            risk_age_y_vars = risk_age_y_vars[lead_time:]
            total_risk_age_x_vars.extend(risk_age_x_vars)
            total_risk_age_y_vars.extend(risk_age_y_vars)
            # ax.scatter(x_vars, y_vars, s= 0.5)
    
    for age in range(5):
        for risk in range(2):
            # print(total_risk_age_x_vars)
            print("age:", age, "risk:", risk)
            x_vars = np.array(np.array(total_risk_age_x_vars)[:, age, risk]).reshape(-1, 1)
            mean_x_vars = np.mean(x_vars)
            y_vars = np.array(np.array(total_risk_age_y_vars)[:, age, risk]).reshape(-1, 1)
            mean_y_vars = np.mean(y_vars)
            print("empirical yhr:", mean_y_vars / mean_x_vars)

def empirical_yhr_toPY(simulation_filename):
    total_risk_age_x_vars = []
    total_risk_age_y_vars = []
    with open(simulation_filename) as file:
        data = json.load(file)
    risk_age_x_vars = data["ToIY_history"]
    risk_age_y_vars = data["ToIHT_history"]
    print("deterministic path")
   
            # x_vars = np.array(np.array(risk_age_x_vars)[:, age, risk]).reshape(-1, 1)
    mean_x_vars = np.sum(np.array(data['ToIY_history']), axis=0)
    # y_vars = np.array(np.array(risk_age_y_vars)[:, age, risk]).reshape(-1, 1)
    mean_y_vars = np.sum(np.array(data['ToIHT_history']), axis=0)
    print("empirical To_IH:", mean_y_vars)
    print("empirical ToIY:", mean_x_vars)
    print("empirical yhr:", mean_y_vars / mean_x_vars)



    


# linear_regression("./output/cook/output_cook_more_var.json", ["ToIY_history"], "IH_history")
# linear_regression("./output/cook/output_cook_more_var.json", ["ToIY_history"], "ToIH_history")
# linear_regression("./output/cook/output_cook_more_var.json", ["E_history"], "IH_history")
# linear_regression("./output/cook/output_cook_more_var.json", ["E_history"], "ToIH_history")
# linear_regression("./output/cook/output_cook_more_var.json", ["PY_history"], "IH_history")
# linear_regression("./output/cook/output_cook_more_var.json", ["PY_history"], "ToIHT_history")
linear_regression("./output/cook/output_cook_more_var.json", ["ToPY_history"], "ToIHT_history")
# linear_regression("./output/cook/output_cook_more_var.json", ["ToIY_history"], "IH_history")
# linear_regression("./output/cook/output_cook_more_var.json", ["ToIY_history"], "ToIH_history")


# empirical_90_pi_lr("./output/cook/output_cook_more_var.json", ["PY_history"], "ToIHT_history")

# empirical_yhr("./output/cook/output_cook_more_var.json")
# empirical_yhr_toPY("./output/cook/ToPY_transmission_lsq_1.0_apr.json")
