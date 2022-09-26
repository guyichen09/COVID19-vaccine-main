from cProfile import label
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from InputOutputTools import SimReplication_IO_list_of_arrays_var_names
from scipy.signal import find_peaks

def plot_from_file(simulation_filename, instance):
    with open(simulation_filename) as file:
        data = json.load(file)
    
    for var in SimReplication_IO_list_of_arrays_var_names:
        y_data = data[var]
        print(var)
        print(len(y_data))
        plt.clf()
        fig, ax = plt.subplots()
        start_date = instance.start_date
        num_days = len(y_data)
        x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
        if var == "ICU_history":
            ax.scatter(x_axis, instance.real_hosp_icu[0:num_days])
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "ICU_history.png")
        elif var == "IH_history":
            real_hosp = [ai - bi for (ai, bi) in zip(instance.real_hosp, instance.real_hosp_icu)]
            ax.scatter(x_axis, real_hosp[0:num_days])
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "IH_history.png")
        elif var == "ToIHT_history":
            ax.scatter(x_axis, instance.real_hosp_ad[0:num_days])
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "admission_history.png")
        elif var == "D_history":
            ax.scatter(x_axis, np.cumsum(instance.real_death_total[0:num_days]))
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "death_history.png")

        # plt.show()


def plot_austin_vs_cook():
    austin_files = ["austin_real_hosp_updated.csv",
              "austin_real_icu_updated.csv",
              "austin_hosp_ad_updated.csv",
              "austin_real_death_from_hosp_updated.csv",
              "austin_real_total_death.csv"]
    cook_files = ["cook_hosp.csv",
              "cook_hosp_icu.csv",
              "cook_hosp_ad.csv",
              "cook_deaths_from_hosp_est.csv",
              "cook_deaths.csv"]

    outfile_names = ["hosp", "icu", "admission", "death_from_hosp", "death_total"]

    for (austin_file, cook_file, outfile_name) in zip(austin_files, cook_files, outfile_names):
        df_austin_hosp = pd.read_csv(
                "./instances/austin/" + austin_file,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        df_cook_hosp = pd.read_csv(
                "./instances/cook/" + cook_file,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        austin_peaks, _ = find_peaks(df_austin_hosp["hospitalized"], distance=60)
        second_austin_peak_idx = austin_peaks[1]
        austin_date = df_austin_hosp["date"]
        austin_hosp_data = df_austin_hosp["hospitalized"]
        austin_peak_date = austin_date[second_austin_peak_idx]
        austin_hosp_peak_val = austin_hosp_data[second_austin_peak_idx]
        cook_peaks, _ = find_peaks(df_cook_hosp["hospitalized"],distance=60)
        first_cook_peak_idx = cook_peaks[0]
        cook_date = df_cook_hosp["date"]
        cook_peak_date = cook_date[first_cook_peak_idx]
        cook_hosp_data = df_cook_hosp["hospitalized"]
        cook_hosp_peak_val = cook_hosp_data[first_cook_peak_idx]
        time_delta = (cook_peak_date - austin_peak_date).days
        print(time_delta)
        scale_factor = cook_hosp_peak_val / austin_hosp_peak_val
        print(scale_factor)
        # unshifted
        plt.clf()
        fig, ax = plt.subplots()
        if "death" in outfile_name: 
            ax.scatter(df_austin_hosp["date"], np.cumsum(df_austin_hosp["hospitalized"]), label="Austin")
            ax.scatter(df_cook_hosp["date"], np.cumsum(df_cook_hosp["hospitalized"]), label="Cook")
        else:
            ax.scatter(df_austin_hosp["date"], df_austin_hosp["hospitalized"], label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks], df_austin_hosp["hospitalized"][austin_peaks], "x", c="black")
            ax.scatter(df_cook_hosp["date"], df_cook_hosp["hospitalized"], label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], df_cook_hosp["hospitalized"][cook_peaks], "x", c="black")
        fig.autofmt_xdate()
        plt.legend(loc='best')
        plt.savefig("./plots/"+ "unshifted_unscaled_" + outfile_name + ".png" )
        
        # shifted
        plt.clf()
        fig, ax = plt.subplots()
        if "death" in outfile_name: 
            ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"]), label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"])[austin_peaks], "x", c="black")
            ax.scatter(df_cook_hosp["date"], np.cumsum(df_cook_hosp["hospitalized"]), label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], np.cumsum(df_cook_hosp["hospitalized"])[cook_peaks], "x", c="black")
        else:
            ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"], label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"][austin_peaks], "x", c="black")
            ax.scatter(df_cook_hosp["date"], df_cook_hosp["hospitalized"], label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], df_cook_hosp["hospitalized"][cook_peaks], "x", c="black")
        fig.autofmt_xdate()
        plt.legend(loc='best')
        plt.savefig("./plots/"+ "shifted_unscaled_" + outfile_name + ".png" )

        plt.clf()
        fig, ax = plt.subplots()
        # shifted and scaled
        if "death" in outfile_name: 
            ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"]) * scale_factor, label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"])[austin_peaks] * scale_factor, "x", c="black")
            ax.scatter(df_cook_hosp["date"], np.cumsum(df_cook_hosp["hospitalized"]), label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], np.cumsum(df_cook_hosp["hospitalized"])[cook_peaks], "x", c="black")
        else:
            ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"] * scale_factor, label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"][austin_peaks] * scale_factor, "x", c="black")
            ax.scatter(df_cook_hosp["date"], df_cook_hosp["hospitalized"], label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], df_cook_hosp["hospitalized"][cook_peaks], "x", c="black")
        fig.autofmt_xdate()
        plt.legend(loc='best')
        plt.savefig("./plots/"+ "shifted_scaled_" + outfile_name + ".png" )

    


# plot_austin_vs_cook()  

        

        

        
