import datetime
from sys import prefix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from InputOutputTools import SimReplication_IO_list_of_arrays_var_names
from scipy.signal import find_peaks

def plot_from_file(simulation_filename, out_file, instance):
    # plt.clf()
    # fig, ax = plt.subplots()
    if "cook" in simulation_filename:
        dir = "cook/"
    elif "austin" in simulation_filename:
        dir = "austin/"
    prefix = out_file[:-5] + "_test_"
    for var in SimReplication_IO_list_of_arrays_var_names:
        plt.clf()
        start_date = instance.start_date
        fig, ax = plt.subplots()
        # for i in range(1, 301):
        #     with open("2323_" + str(i) + "_sim.json") as file:
        #         data = json.load(file)
        #     y_data = data[var]
        #     start_date = instance.start_date
        #     num_days = len(y_data)
        #     x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
        #     ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1),color="gray")
        with open(simulation_filename) as file:
            data = json.load(file)
        if var == "ICU_history":
            y_data = data[var]
            num_days = len(y_data)
            x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted", zorder=2001)
            ax.scatter(x_axis, instance.real_hosp_icu[0:num_days], color="r", label="data", zorder=2000)
            fig.autofmt_xdate()
            # plt.title('ICU census vs real data', fontsize=20)
            plt.ylabel('COVID-19 ICU patients', fontsize=16)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix+  "ICU_history.png")
        elif var == "IH_history":
            y_data = data[var]
            num_days = len(y_data)
            x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
            real_hosp = [ai - bi for (ai, bi) in zip(instance.real_hosp, instance.real_hosp_icu)]
            
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted", zorder=2001)
            ax.scatter(x_axis, real_hosp[0:num_days], color="r", label="data", zorder=2000)
            fig.autofmt_xdate()
            # plt.title('General ward census vs real data', fontsize=20)
            plt.ylabel('COVID-19 general ward patients', fontsize=16)
            plt.legend() 
            plt.savefig("./plots/" + dir +prefix+  "IH_history.png")
        elif var == "ToIHT_history":
            y_data = data[var]
            num_days = len(y_data)
            x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
            # ax.scatter(x_axis, instance.real_hosp_ad[0:num_days], color="r", label="data")
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted", zorder=2001)
            ax.scatter(x_axis, instance.real_hosp_ad[0:num_days], color="r", label="data", zorder=2000)
            fig.autofmt_xdate()
            # plt.title('admission vs real data', fontsize=20)
            plt.ylabel('COVID-19 Hospital Admission', fontsize=16)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix+ "admission_history.png")
        elif var == "D_history":
            y_data = data[var]
            num_days = len(y_data)
            x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted", zorder=2001)
            ax.scatter(x_axis, np.cumsum(instance.real_death_total[0:num_days]), color="r", label="data", zorder=2000)
            fig.autofmt_xdate()
            plt.ylabel('COVID-19 cumulative death' , fontsize=20)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix + "death_history.png")
        plt.close()
    fig, ax = plt.subplots()
    ICUD = data['ToICUD_history']
    IHD = data['ToIHD_history']
    ToICU = data['ToICU_history']
    ToIH = data['ToIH_history']
    ToIHT = data['ToIHT_history']
    total_death = [np.array(a) + np.array(b) for (a, b) in zip(ICUD, IHD)]
    # print(total_death[0])
    total_admission = [np.array(a) + np.array(b) for (a, b) in zip(ToICU, ToIH)]
    # print(np.sum(total_admission))
    death_ratio = [np.array(a) / np.array(b) for (a, b) in zip(total_death, ToIHT)]
    death_ratio = np.array(death_ratio)
    # print(death_ratio)
    # for i in range(2):
    #     for j in range(5):
    #         risk = "low" if i == 0 else "high"
    #         ax.plot(x_axis, death_ratio[:,j,i], label= str(j)+ "_" + risk)
    # plt.legend()
    # plt.show()


    # print(total_admission[-1])
    # print(np.sum(total_admission[-1]))
    calculated_total_death = [np.sum(np.array(a).reshape(5, 2), axis=1) * np.array([0.05859157, 0.05859157, 0.05859157, 0.15620653, 0.30852592]) for a in ToIHT]
    # calculated_gw_death = [np.sum(np.array(a).reshape(5, 2), axis=1) * np.array([0.024, 0.024, 0.052, 0.093, 0.22159]) for a in ToIH]
    
    # calculated_total_gw_death = np.sum(np.sum(calculated_gw_death))
    # print(calculated_total_gw_death)

    calculated_total_death = np.sum(np.array(calculated_total_death), axis=0)
    print(calculated_total_death)

    calculated_IY_total_death = np.sum(np.sum(np.array(data['ToIYD_history']), axis=0), axis=1)
    print("calculated_IY_death", calculated_IY_total_death)

    print("total_admission", np.sum(np.array(ToIHT), axis=0))
    total_IY = np.sum(np.array(data['ToIY_history']), axis=0)
    print("total_IY", total_IY)
    print(np.sum(np.array(ToIHT), axis=0)/ total_IY)
    # for (a, b) in zip(total_death, total_admission):
    #     # print(a) 
    #     # print(b)
    #     print(np.array(a) / np.array(b))
    # # plot 7 day moving average IYD ratio and ICUD ratio
    # IYD = np.sum(np.sum(data['ToIYD_history'], axis=1), axis=1)
    # ICUD = np.sum(np.sum(data['ToICUD_history'], axis=1), axis=1)
    # IHD = np.sum(np.sum(data['ToIHD_history'], axis=1), axis=1)
    # # TD = ICUD + IHD
    # window_size = 7
    # IYD_numbers_series = pd.Series(IYD)
    # ICUD_numbers_series = pd.Series(ICUD)
    # IHD_numbers_series = pd.Series(IHD)
    # # Get the window of series
    # # of observations of specified window size
    # IYD_windows = IYD_numbers_series.rolling(window_size)
    # ICUD_windows = ICUD_numbers_series.rolling(window_size)
    # IHD_windows = IHD_numbers_series.rolling(window_size)
    # # Create a series of moving
    # # averages of each window
    # seven_day_ma_IYD = IYD_windows.mean()
    # seven_day_ma_ICUD = ICUD_windows.mean()
    # seven_day_ma_IHD = IHD_windows.mean()
    # # Convert pandas series back to list
    # seven_day_ma_IYD_list = seven_day_ma_IYD.tolist()
    # seven_day_ma_ICUD_list = seven_day_ma_ICUD.tolist()
    # seven_day_ma_IHD_list = seven_day_ma_IHD.tolist()
    # # Remove null entries from the list
    # seven_day_ma_IYD_list = seven_day_ma_IYD_list[window_size - 1:]
    # seven_day_ma_ICUD_list = seven_day_ma_ICUD_list[window_size - 1:]
    # seven_day_ma_IHD_list = seven_day_ma_IHD_list[window_size - 1:]
    # IYD_ratio = [a / (a + b + c) for (a, b, c) in zip(seven_day_ma_IYD_list, seven_day_ma_ICUD_list, seven_day_ma_IHD_list)]
    # plt.clf()
    # fig, ax = plt.subplots()
    # start_date = instance.start_date
    # num_days = len(data['ToIYD_history'])
    # x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)][window_size -1: ]
    # ax.scatter(x_axis, IYD_ratio, color="r")
    # # ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
    # fig.autofmt_xdate()
    # plt.title('IYD ratio', fontsize=20)
    # plt.ylabel('7d MA IYD / (7d MA IYD + ICUD + IHD)', fontsize=16)
    # # plt.legend() 
    # plt.savefig("./plots/" + dir + prefix+  "IYD_ICUD_ratio_history.png")
    '''
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
            ax.scatter(x_axis, instance.real_hosp_icu[0:num_days], color="r", label="data")
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            np.savetxt("icu_history.csv",np.sum(np.sum(y_data, axis=1), axis=1))
            
            fig.autofmt_xdate()
            plt.title('ICU census vs real data', fontsize=20)
            plt.ylabel('COVID-19 ICU patients', fontsize=16)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix+  "ICU_history.png")
        elif var == "IH_history":
            real_hosp = [ai - bi for (ai, bi) in zip(instance.real_hosp, instance.real_hosp_icu)]
            ax.scatter(x_axis, real_hosp[0:num_days], color="r", label="data")
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            fig.autofmt_xdate()
            plt.title('General ward census vs real data', fontsize=20)
            plt.ylabel('COVID-19 general ward patients', fontsize=16)
            plt.legend() 
            plt.savefig("./plots/" + dir +prefix+  "IH_history.png")
            np.savetxt("gw_history.csv",np.sum(np.sum(y_data, axis=1), axis=1),)
        elif var == "ToIHT_history":
            ax.scatter(x_axis, instance.real_hosp_ad[0:num_days], color="r", label="data")
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            fig.autofmt_xdate()
            plt.title('admission vs real data', fontsize=20)
            plt.ylabel('COVID-19 Hospital Admission\n(Seven-day Average)', fontsize=13)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix+ "admission_history.png")
            plt.close()
            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(x_axis,  np.cumsum(np.sum(np.sum(y_data, axis=1), axis=1)), label= "fitted")
            fig.autofmt_xdate()
            plt.title('cumulative admission', fontsize=20)
            plt.savefig("./plots/" + dir + prefix+ "cum_admission.png")
            plt.close()
        elif var == "D_history":
            ax.scatter(x_axis, np.cumsum(instance.real_death_total[0:num_days]), color="r", label="data")
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            fig.autofmt_xdate()
            plt.ylabel('COVID-19 cumulative death' , fontsize=20)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix + "death_history.png")

        elif var == "ToICU_history":
            ax.plot(x_axis,  np.cumsum(np.sum(np.sum(y_data, axis=1), axis=1)), label= "fitted")
            fig.autofmt_xdate()
            plt.title('cumulative ICU admission', fontsize=20)
            plt.savefig("./plots/" + dir + prefix+ "cum_ICU_admission.png")
        plt.close()
        plt.show()
    '''


def plot_debug(simulation_filename, simulation_filename1, out_file, instance):
    with open(simulation_filename) as file:
        data = json.load(file)

    with open(simulation_filename1) as file1:
        data1 = json.load(file1)
    
    if "cook" in simulation_filename:
        dir = "cook/"
    prefix = "debug_" + out_file[:-5] + "_"
    for var in ["ICU_history"]:
        y_data = data[var]
        y_data1 = data1[var]

        print(var)
        print(len(y_data))
        plt.clf()
        fig, ax = plt.subplots()
        start_date = instance.start_date
        num_days = len(y_data)
        x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
        if var == "ICU_history":
            ax.scatter(x_axis, instance.real_hosp_icu[0:num_days], color="r", label="data")
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            ax.plot(x_axis, np.sum(np.sum(y_data1, axis=1), axis=1), label= "fitted-2")
            fig.autofmt_xdate()
            
            total_hosp1 = np.sum(np.sum(y_data[100:num_days], axis=1), axis=1) +  np.sum(np.sum(data["IH_history"][100:num_days], axis=1), axis=1)
            total_hosp2 = np.sum(np.sum(y_data1[100:num_days], axis=1), axis=1) +  np.sum(np.sum(data1["IH_history"][100:num_days], axis=1), axis=1)
            print("fitted icu rsq", np.sum(abs(instance.real_hosp_icu[100:num_days] - np.sum(np.sum(y_data[100:num_days], axis=1), axis=1))))
            print("fitted-2 icu rsq", np.sum(abs(instance.real_hosp_icu[100:num_days] - np.sum(np.sum(y_data1[100:num_days], axis=1), axis=1))))
            real_hosp = [ai - bi for (ai, bi) in zip(instance.real_hosp, instance.real_hosp_icu)]
            print("fitted gw rsq", np.sum(abs(real_hosp[100:num_days] - np.sum(np.sum(data["IH_history"][100:num_days], axis=1), axis=1))))
            print("fitted-2 gw rsq", np.sum(abs(real_hosp[100:num_days] - np.sum(np.sum(data1["IH_history"][100:num_days], axis=1), axis=1))))
            print("fitted rsq", np.sum(abs(instance.real_hosp[100:num_days] - total_hosp1)))
            print("fitted-2 rsq", np.sum(abs(instance.real_hosp[100:num_days] - total_hosp2)))
            plt.title('ICU census vs real data', fontsize=20)
            plt.ylabel('COVID-19 ICU patients', fontsize=16)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix+  "ICU_history_debug.png")
        elif var == "IH_history":
            real_hosp = [ai - bi for (ai, bi) in zip(instance.real_hosp, instance.real_hosp_icu)]
            ax.scatter(x_axis, real_hosp[0:num_days], color="r", label="data")
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            fig.autofmt_xdate()
            plt.title('General ward census vs real data', fontsize=20)
            plt.ylabel('COVID-19 general ward patients', fontsize=16)
            plt.legend() 
            plt.savefig("./plots/" + dir +prefix+  "IH_history.png")
        elif var == "ToIHT_history":
            ax.scatter(x_axis, instance.real_hosp_ad[0:num_days], color="r", label="data")
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            fig.autofmt_xdate()
            plt.title('admission vs real data', fontsize=20)
            plt.ylabel('COVID-19 Hospital Admission\n(Seven-day Average)', fontsize=13)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix+ "admission_history.png")
        elif var == "D_history":
            ax.scatter(x_axis, np.cumsum(instance.real_death_total[0:num_days]), color="r", label="data")
            ax.plot(x_axis,  np.sum(np.sum(y_data, axis=1), axis=1), label= "fitted")
            fig.autofmt_xdate()
            plt.ylabel('COVID-19 cumulative death' , fontsize=20)
            plt.legend() 
            plt.savefig("./plots/" + dir + prefix + "death_history.png")

def plot_austin_vs_cook():
    austin_files = [
        "austin_real_hosp_no_may.csv",
              "austin_real_icu_no_may.csv",
              "austin_hosp_ad_no_may.csv",]
    cook_files = [
        "cook_hosp_region_sum_estimated.csv",
              "cook_icu_region_sum_estimated.csv",
              "cook_hosp_ad_region_sum_18_syn.csv",]

    outfile_names = ["region_sum" + i for i in ["hosp", "icu", "admission",]]

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
        plt.savefig("./plots/"+ "unshifted_unscaled_estimated_" + outfile_name + ".png" )
        plt.show()
        # # shifted
        # plt.clf()
        # fig, ax = plt.subplots()
        # if "death" in outfile_name: 
        #     ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"]), label="Austin")
        #     plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"])[austin_peaks], "x", c="black")
        #     ax.scatter(df_cook_hosp["date"], np.cumsum(df_cook_hosp["hospitalized"]), label="Cook")
        #     plt.plot(df_cook_hosp["date"][cook_peaks], np.cumsum(df_cook_hosp["hospitalized"])[cook_peaks], "x", c="black")
        # else:
        #     ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"], label="Austin")
        #     plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"][austin_peaks], "x", c="black")
        #     ax.scatter(df_cook_hosp["date"], df_cook_hosp["hospitalized"], label="Cook")
        #     plt.plot(df_cook_hosp["date"][cook_peaks], df_cook_hosp["hospitalized"][cook_peaks], "x", c="black")
        # fig.autofmt_xdate()
        # plt.legend(loc='best')
        # plt.savefig("./plots/"+ "shifted_unscaled_estimated_" + outfile_name + ".png" )
        # plt.show()
        plt.clf()
        fig, ax = plt.subplots()
        # shifted and scaled
        if "death" in outfile_name: 
            ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"]) * scale_factor, label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), np.cumsum(df_austin_hosp["hospitalized"])[austin_peaks] * scale_factor, "x", c="black")
            ax.scatter(df_cook_hosp["date"], np.cumsum(df_cook_hosp["hospitalized"]), label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], np.cumsum(df_cook_hosp["hospitalized"])[cook_peaks], "x", c="black")
        elif "admission" in outfile_name:
            ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta + 8, 'D'), df_austin_hosp["hospitalized"] * scale_factor, label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta+ 8, 'D'), df_austin_hosp["hospitalized"][austin_peaks] * scale_factor, "x", c="black")
            ax.scatter(df_cook_hosp["date"], df_cook_hosp["hospitalized"], label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], df_cook_hosp["hospitalized"][cook_peaks], "x", c="black")
        else:
            ax.scatter(df_austin_hosp["date"] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"] * scale_factor, label="Austin")
            plt.plot(df_austin_hosp["date"][austin_peaks] + np.timedelta64(time_delta, 'D'), df_austin_hosp["hospitalized"][austin_peaks] * scale_factor, "x", c="black")
            ax.scatter(df_cook_hosp["date"], df_cook_hosp["hospitalized"], label="Cook")
            plt.plot(df_cook_hosp["date"][cook_peaks], df_cook_hosp["hospitalized"][cook_peaks], "x", c="black")
        fig.autofmt_xdate()
        plt.legend(loc='best')
        plt.savefig("./plots/"+ "shifted_scaled_estimated_" + outfile_name + ".png" )

        # plt.show()

def plot_region_sum_vs_state():
    state_files = [
        "cook_hosp_estimated.csv",
              "cook_icu_estimated.csv",
              "cook_hosp_ad_17_syn.csv",]
    region_sum_files = [
        "cook_hosp_region_sum_estimated.csv",
              "cook_icu_region_sum_estimated.csv",
              "cook_hosp_ad_region_sum_18_syn.csv",]

    outfile_names = ["" + i for i in ["hosp", "icu", "admission",]]

    for (state_file, region_sum_file, outfile_name) in zip(state_files, region_sum_files, outfile_names):
        df_state_hosp = pd.read_csv(
                "./instances/cook/" + state_file,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        df_region_sum_hosp = pd.read_csv(
                "./instances/cook/" + region_sum_file,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        # unshifted
        plt.clf()
        fig, ax = plt.subplots()
        
        ax.scatter(df_state_hosp["date"], df_state_hosp["hospitalized"], label="state")
        
        ax.scatter(df_region_sum_hosp["date"], df_region_sum_hosp["hospitalized"], label="region_sum")
            
        fig.autofmt_xdate()
        plt.legend(loc='best')
        plt.savefig("./plots/"+ "state_vs_region_sum_" + outfile_name + ".png" )
        plt.show()
     

# plot_austin_vs_cook()  
# plot_region_sum_vs_state()
        

        

        
