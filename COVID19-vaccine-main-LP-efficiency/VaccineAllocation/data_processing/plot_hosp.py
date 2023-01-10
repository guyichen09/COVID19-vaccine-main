import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


def plot_total_beds(option = "cook"):
    if option == "cook":
        region_sum_total_beds = pd.read_csv("./instances/cook/cook_total_beds_region_sum.csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2020_idph = pd.read_csv("./instances/cook/cook_total_beds_combined_2020_idph.csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2021_idph = pd.read_csv("./instances/cook/cook_total_beds_2021_idph.csv", parse_dates=['date'], index_col=0)
        title = "Cook County hospitalization census"

    elif option == "IL":
        region_sum_total_beds = pd.read_csv("./instances/cook/IL_total_beds_region_sum.csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2020_idph = pd.read_csv("./instances/cook/IL_total_beds_combined_2020_idph.csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2021_idph = pd.read_csv("./instances/cook/IL_total_beds_2021_idph.csv", parse_dates=['date'], index_col=0)
        title = "IL hospitalization census"
    else:
        fig, ax = plt.subplots()
        for idx, option in enumerate([7, 9, 10, 11]): 
            region_sum_total_beds = pd.read_csv("./instances/cook/region_" + str(option) + "_total_beds_region_sum.csv", parse_dates=['date'], index_col=0)
            cook_total_beds_2020_idph = pd.read_csv("./instances/cook/region_" + str(option) + "_total_beds_combined_2020_idph.csv", parse_dates=['date'], index_col=0)
            cook_total_beds_2021_idph = pd.read_csv("./instances/cook/region_" + str(option) + "_total_beds_2021_idph.csv", parse_dates=['date'], index_col=0)
            title = "Region " + str(option)
            plt.subplot(2, 2, idx+1)
            plt.plot(region_sum_total_beds['date'], region_sum_total_beds['hospitalized'], label="IDPH website")
            plt.plot(cook_total_beds_2020_idph['date'], cook_total_beds_2020_idph['hospitalized'], label="WW Nov update")
            plt.plot(cook_total_beds_2021_idph['date'], cook_total_beds_2021_idph['hospitalized'], label="WW Oct update")
            plt.ylabel("Total beds (COVID patients)")
            plt.title(title)
            myFmt = mdates.DateFormatter('%y-%m')
            ax.xaxis.set_major_formatter(myFmt)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.legend()    
        plt.savefig("./" + "791011_different_hosp_source.png")


def plot_admission(infile_name= "", option = "cook"):
    if option == "cook":
        region_sum_total_beds = pd.read_csv("./instances/cook/cook_hosp_ad_region_sum.csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2020_idph = pd.read_csv("./instances/cook/cook_county_" + infile_name + ".csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2021_idph = pd.read_csv("./instances/cook/cook_ad_2021_idph.csv", parse_dates=['date'], index_col=0)
        title = "Cook County admission"

    elif option == "IL":
        region_sum_total_beds = pd.read_csv("./instances/cook/IL_hosp_ad_region_sum.csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2020_idph = pd.read_csv("./instances/cook/IL_" + infile_name + ".csv", parse_dates=['date'], index_col=0)
        cook_total_beds_2021_idph = pd.read_csv("./instances/cook/IL_ad_2021_idph.csv", parse_dates=['date'], index_col=0)
        title = "IL admission"
    else:
        fig, ax = plt.subplots()
        for idx, option in enumerate(["cook", "IL"]): 
            region_sum_total_beds = pd.read_csv("./instances/cook/" + option + "_hosp_ad_region_sum.csv", parse_dates=['date'], index_col=0)
            cook_total_beds_2020_idph = pd.read_csv("./instances/cook/" + option + "_" + infile_name + ".csv", parse_dates=['date'], index_col=0)
            cook_total_beds_2021_idph = pd.read_csv("./instances/cook/" + option + "_ad_2021_idph.csv", parse_dates=['date'], index_col=0)
            if option == "cook":
                title1 = "Cook County"
            elif option == "IL":
                title1 = "IL"
            plt.subplot(2, 1, idx+1)
            plt.plot(region_sum_total_beds['date'], region_sum_total_beds['hospitalized'], label="IDPH website")
            plt.plot(cook_total_beds_2020_idph['date'], cook_total_beds_2020_idph['cases'], label="Linelisted data")
            plt.plot(cook_total_beds_2021_idph['date'], cook_total_beds_2021_idph['hospitalized'], label="Admissions_WW_Surveilance")
            plt.ylabel("COVID Admission (7 day MA)")
            plt.title(title1)
            myFmt = mdates.DateFormatter('%y-%m')
            ax.xaxis.set_major_formatter(myFmt)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.legend()    
        plt.savefig("./" + "cook_IL_different_ad_source_h.png")
    # fig, ax = plt.subplots()
    # plt.plot(region_sum_total_beds['date'], region_sum_total_beds['hospitalized'], label="IDPH website")
    # plt.plot(cook_total_beds_2020_idph['date'], cook_total_beds_2020_idph['cases'], label="Linelisted data")
    # plt.plot(cook_total_beds_2021_idph['date'], cook_total_beds_2021_idph['hospitalized'], label="Admissions_WW_Surveilance")
    # plt.ylabel("COVID-19 Admission")
    # plt.title(title)
    # myFmt = mdates.DateFormatter('%y-%m')
    # ax.xaxis.set_major_formatter(myFmt)
    # plt.legend()       
    # plt.savefig("./" + str(option) + infile_name + "_adm_different_hosp_source.png")


# plot_total_beds("cook")
# plot_total_beds("IL")
# plot_total_beds(None)
# for i in range(1, 12):
#     plot_total_beds(i)

# plot_admission("sensitivezips_hosp_adm-comb-221205_7ma", "cook")
plot_admission("sensitivezips_hosp_adm-comb-221205_7ma", "ILd")
# plot_admission("sensitivezips_hosp_adm-old_7ma", "cook")
# plot_admission("sensitivezips_hosp_adm-old_7ma", "IL")