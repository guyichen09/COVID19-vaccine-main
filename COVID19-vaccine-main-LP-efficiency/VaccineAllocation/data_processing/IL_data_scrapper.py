from ast import parse
from cProfile import label
import urllib.request
import json
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

filepath = "./instances/cook/"
def scrape_admission_total_beds_regions(input_regions=[10,11], outfile_name="cook"):
    regions = input_regions

    json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetResurgenceData?regionID='+str(11)+'&daysIncluded=0')
    data = json.loads(json_url.read())

    total_admission_df = pd.DataFrame(0, index=np.arange(len(data)), columns=['date', 'hospitalized'])
    total_beds_df = pd.DataFrame(0, index=np.arange(len(data)), columns=['date', 'hospitalized'])

    for region in regions:
        # read this file - this is where they put the data.  It's a json file, so need to re-format into a dataframe...
        json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetResurgenceData?regionID='+str(region)+'&daysIncluded=0')
        data = json.loads(json_url.read())
        mydf = pd.json_normalize(data)
        admission_df = mydf[['ReportDate','CLIAdmissions_RollingAvg']]
        bed_df = mydf[['ReportDate','COVIDHospitalBedsInUse']]
        admission_df = admission_df.rename(columns = {"ReportDate": "date", "CLIAdmissions_RollingAvg": "hospitalized"})
        bed_df = bed_df.rename(columns = {"ReportDate": "date", "COVIDHospitalBedsInUse": "hospitalized"})
        
        admission_df["date"] = pd.to_datetime(admission_df.date, format='%Y-%m-%dT%H:%M:%S')
        admission_df["date"] = admission_df["date"].dt.strftime("%m/%d/%y")
        # print(data[0])

        bed_df["date"] = pd.to_datetime(bed_df.date, format='%Y-%m-%dT%H:%M:%S')
        bed_df["date"] = bed_df["date"].dt.strftime("%m/%d/%y")

        total_admission_df['hospitalized'] += admission_df['hospitalized']
        total_beds_df['hospitalized'] += bed_df['hospitalized']
        total_admission_df['date'] = admission_df['date']
        total_beds_df['date'] = bed_df['date']

    total_admission_df.to_csv(os.path.join(filepath, outfile_name + '_hosp_ad_region_sum.csv'))
    total_beds_df.to_csv(os.path.join(filepath, outfile_name + '_total_beds_region_sum.csv'))


def scrape_cook_death_date():
    cook_death_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetCountyTestResultsTimeSeries?CountyName=cook')
    chicago_death_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetCountyTestResultsTimeSeries?CountyName=chicago')
    cook_death = json.loads(cook_death_url.read())
    cook_death_df = pd.json_normalize(cook_death)
    cook_death_df = cook_death_df[['ReportDate','Deaths']]
    cook_death_df = cook_death_df.rename(columns = {"ReportDate": "date", "Deaths": "hospitalized"})

    cook_death_df["date"] = pd.to_datetime(cook_death_df.date, format='%Y-%m-%dT%H:%M:%S')
    cook_death_df["date"] = cook_death_df["date"].dt.strftime("%m/%d/%y")


    chicago_death = json.loads(chicago_death_url.read())
    chicago_death_df = pd.json_normalize(chicago_death)
    chicago_death_df = chicago_death_df[['ReportDate','Deaths']]
    chicago_death_df = chicago_death_df.rename(columns = {"ReportDate": "date", "Deaths": "hospitalized"})

    chicago_death_df["date"] = pd.to_datetime(chicago_death_df.date, format='%Y-%m-%dT%H:%M:%S')
    chicago_death_df["date"] = chicago_death_df["date"].dt.strftime("%m/%d/%y")


    total_death_df = pd.DataFrame(0, index=np.arange(len(chicago_death)), columns=['date', 'hospitalized'])
    total_death_df["hospitalized"] = chicago_death_df["hospitalized"] + cook_death_df["hospitalized"]
    total_death_df["date"] = chicago_death_df["date"]
    total_death_df.to_csv(os.path.join(filepath,'cook_deaths.csv'))


def scrape_state_hosp_data():
    json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetHospitalUtilizationResults')
    data = json.loads(json_url.read())
    mydf = pd.json_normalize(data)

    print(mydf.info())

    icu_df = mydf[["ReportDate", "ICUInUseBedsCOVID"]]
    icu_df = icu_df.rename(columns = {"ReportDate": "date", "ICUInUseBedsCOVID": "hospitalized"})

    icu_df["hospitalized"] = icu_df["hospitalized"]
    icu_df["date"] = pd.to_datetime(icu_df.date, format='%Y-%m-%dT%H:%M:%S')
    icu_df["date"] = icu_df["date"].dt.strftime("%m/%d/%y")
    icu_df.to_csv(os.path.join(filepath,'IL_hosp_icu.csv'))
    
    generalward_df = mydf[["ReportDate", "TotalInUseBedsCOVID"]]
    generalward_df = generalward_df.rename(columns = {"ReportDate": "date", "TotalInUseBedsCOVID": "hospitalized"})

    generalward_df["hospitalized"] = generalward_df["hospitalized"]
    generalward_df["date"] = pd.to_datetime(generalward_df.date, format='%Y-%m-%dT%H:%M:%S')
    generalward_df["date"] = generalward_df["date"].dt.strftime("%m/%d/%y")
    generalward_df.to_csv(os.path.join(filepath,'IL_hosp.csv'))

def get_current_hosp_utilization_region():
    hospital_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetHospitalizationResultsRegion')
    hosp_data = json.loads(hospital_url.read())

def calculate_cook_IL_ratio():
    cook_bed_data = pd.read_csv("./instances/cook/cook_total_beds_region_sum.csv", parse_dates=["date"], index_col=0)
    IL_bed_data = pd.read_csv("./instances/cook/IL_total_beds_region_sum.csv", parse_dates=["date"], index_col=0).reset_index()
    cook_bed_data["date"] = cook_bed_data["date"].dt.strftime("%m/%d/%y")
    chicago_bed_data = pd.read_csv("./instances/cook/chicago_total_beds_region_sum.csv", parse_dates=["date"], index_col=0)
    cook_ad_data = pd.read_csv("./instances/cook/cook_hosp_ad_region_sum.csv", parse_dates=["date"], index_col=0).reset_index()
    IL_ad_data = pd.read_csv("./instances/cook/IL_hosp_ad_region_sum.csv", parse_dates=["date"], index_col=0).reset_index()
    
    ratio_bed_cook_IL = cook_bed_data["hospitalized"] / IL_bed_data["hospitalized"]
    ratio_ad_cook_IL = cook_ad_data["hospitalized"] / IL_ad_data["hospitalized"]

    # ratio_bed_chicago_cook = chicago_bed_data["hospitalized"]/ cook_bed_data["hospitalized"]
    # print(ratio_bed_chicago_cook[:100].describe())
    # print(ratio_bed_cook_IL)
    # print(ratio_ad_cook_IL)

    # state data 
    state_bed_data = pd.read_csv("./instances/cook/IL_hosp.csv", parse_dates=["date"], index_col=0).reset_index(drop=True)
    state_icu_data = pd.read_csv("./instances/cook/IL_hosp_icu.csv", parse_dates=["date"], index_col=0).reset_index(drop=True)
    IL_total_beds = state_bed_data.query("date >= @ dt.datetime(2020, 6, 13) & date <= @ dt.datetime(2022, 4, 7)").reset_index()
    IL_icu_beds = state_icu_data.query("date >= @ dt.datetime(2020, 6, 13) & date <= @ dt.datetime(2022, 4, 7)").reset_index()
    first_IL_beds = state_bed_data.query("date < @ dt.datetime(2020, 6, 13)")
    first_icu_beds = state_icu_data.query("date < @ dt.datetime(2020, 6, 13)")

    total_beds_copy = cook_bed_data
    first_icu_beds["date"] = first_icu_beds["date"].dt.strftime("%m/%d/%y")
    first_IL_beds["date"] = first_IL_beds["date"].dt.strftime("%m/%d/%y")
    estimated_icu_cook = IL_icu_beds["hospitalized"] * ratio_bed_cook_IL * 1.11
    icu_copy = pd.DataFrame({"date":cook_bed_data["date"], "hospitalized": estimated_icu_cook})
   
    estimated_hosp_cook = IL_total_beds["hospitalized"] * ratio_bed_cook_IL
    total_beds_copy["hospitalized"] =  estimated_hosp_cook
    first_IL_beds["hospitalized"] = first_IL_beds["hospitalized"] * 0.7
    first_icu_beds["hospitalized"] = first_icu_beds["hospitalized"] * 0.7 * 1.11

    estimated_icu_cook_df = pd.concat([first_icu_beds, icu_copy]).reset_index(drop=True)
    estimated_beds_cook_df = pd.concat([first_IL_beds, total_beds_copy]).reset_index(drop=True)
    estimated_icu_cook_df.to_csv(os.path.join(filepath,'cook_icu_estimated_adjusted.csv'))
    # estimated_beds_cook_df.to_csv(os.path.join(filepath,'cook_hosp_estimated.csv'))

def icu_hosp_ratio():
    state_bed_data = pd.read_csv("./instances/cook/IL_hosp.csv", index_col=0).reset_index(drop=True)
    state_icu_data = pd.read_csv("./instances/cook/IL_hosp_icu.csv", index_col=0).reset_index(drop=True)
    icu_hosp_ratio = state_icu_data["hospitalized"] / state_bed_data["hospitalized"]
    cook_bed_data = pd.read_csv("./instances/cook/cook_hosp_region_sum_estimated.csv")
    cook_icu_data = cook_bed_data["hospitalized"] * icu_hosp_ratio
    cook_icu_data_df = pd.DataFrame({"date": state_bed_data["date"], "hospitalized": cook_icu_data})
    cook_icu_data_df.to_csv(os.path.join(filepath,'cook_icu_region_sum_estimated.csv'))

def hosp_region_sum_estimation():
    cook_bed_data = pd.read_csv("./instances/cook/cook_total_beds_region_sum.csv", index_col=0)
    state_bed_data = pd.read_csv("./instances/cook/IL_hosp.csv", parse_dates=["date"], index_col=0).reset_index(drop=True)
    first_IL_beds = state_bed_data.query("date < @ dt.datetime(2020, 6, 13)")
    first_IL_beds["date"] = first_IL_beds["date"].dt.strftime("%m/%d/%y")
    first_IL_beds["hospitalized"] = first_IL_beds["hospitalized"] * 1.05 * 0.7
    estimated_beds_cook_df = pd.concat([first_IL_beds, cook_bed_data]).reset_index(drop=True)
    estimated_beds_cook_df.to_csv(os.path.join(filepath,'cook_hosp_region_sum_estimated.csv'))



def add_shifted_scaled_austin_data_to_cook():
    austin_files = ["austin_real_hosp_no_may.csv",
                "austin_real_icu_no_may.csv",
                "austin_hosp_ad_no_may.csv",]
    cook_files = ["cook_hosp_region_sum_estimated.csv",
                "cook_icu_region_sum_estimated.csv",
                "cook_hosp_ad_region_sum_18_syn.csv",]

    outfile_names = ["hosp", "icu", "admission",]

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
        # print(time_delta)
        scale_factor = cook_hosp_peak_val / austin_hosp_peak_val
        print(outfile_name)
        print(scale_factor)

        first_cook_date = cook_date[0]
        if "admission" in outfile_name:
            shifted_first_date_austin = austin_date[0] + dt.timedelta(time_delta + 8)
            shifted_austin_dates = austin_date + np.timedelta64(time_delta + 8, 'D')
            # print(shifted_austin_dates[:-first_time_delta])
            first_time_delta = (shifted_first_date_austin - first_cook_date).days
            shifted_austin_dates = shifted_austin_dates[:-first_time_delta]
            austin_shifted_scaled_data = austin_hosp_data[:-first_time_delta] * scale_factor
        else:
            shifted_first_date_austin = austin_date[0] + dt.timedelta(time_delta)
            shifted_austin_dates = austin_date + np.timedelta64(time_delta, 'D')
            first_time_delta = (shifted_first_date_austin - first_cook_date).days
            # print(shifted_austin_dates[:-first_time_delta])
            shifted_austin_dates = shifted_austin_dates[:-first_time_delta]
            austin_shifted_scaled_data = austin_hosp_data[:-first_time_delta] * scale_factor
        print(shifted_first_date_austin)
        print(first_cook_date)
        # print(first_time_delta)
       
        dates = pd.concat([shifted_austin_dates, cook_date]).dt.strftime("%m/%d/%y")
        hosp_data = pd.concat([austin_shifted_scaled_data, cook_hosp_data])
        df = pd.DataFrame({'date': dates,
                    'hospitalized': hosp_data}).reset_index(drop=True)
        # print(df[20:70])
        df.to_csv("./instances/cook/" + "smoothed_region_sum_estimated_no_may_" + outfile_name + ".csv")


def line_list_admission(in_file_name="sensitivezips_hosp_adm-comb.csv", is_cook=True):
    Zipdata = pd.read_csv("./instances/cook/" + in_file_name)
    if is_cook:
        groupby_data = Zipdata.drop_duplicates(subset=['date','zipcode']).groupby(['date','county']).agg( {'cases':'sum','population':'sum','land_area':'sum'}).sort_values( by=['date','county']).reset_index()
        cook_county_data = groupby_data.loc[groupby_data['county'] == "Cook"]
        cook_county_data.to_csv("./instances/cook/cook_county_" + in_file_name)
    else:
        groupby_data = Zipdata.drop_duplicates(subset=['date','zipcode']).groupby(['date']).agg( {'cases':'sum','population':'sum','land_area':'sum'}).sort_values(by=['date']).reset_index()
        # cook_county_data = groupby_data.loc[groupby_data['county'] == "Cook"]
        groupby_data.to_csv("./instances/cook/IL_" + in_file_name)
        

def process_line_list_admission():
    cook_county_data = pd.read_csv("./instances/cook/senstive_line_list_admission.csv")
    idx = pd.date_range('01-02-2020', '06-13-2020')
    cook_county_data_cases = cook_county_data["cases"]
    cook_county_data_cases.index = cook_county_data['date']
    
    cook_county_data_cases.index = pd.DatetimeIndex(cook_county_data_cases.index) 
    cook_county_data_cases = cook_county_data_cases.reindex(idx, fill_value=0)
    cook_county_data_cases.to_csv("./instances/cook/sensitive_line_list_admission_processed.csv")


def ww_admission_data(is_cook=True):
    data = pd.read_csv("./instances/cook/Admissions_WW_surveilance.csv", parse_dates=["date"])
    if is_cook:
        x = data.loc[data['CovidRegionID'].isin(["10", "11"])]
        cook_data = x.groupby(["date"]).aggregate({"count_rolling_7day_mean":sum}).sort_values( by=['date']).reset_index()
        cook_data["date"] = cook_data["date"].dt.strftime("%m/%d/%y")
        print(cook_data)
        cook_ad_df = cook_data[["date", "count_rolling_7day_mean"]]
        cook_ad_df = cook_ad_df.rename(columns={"count_rolling_7day_mean": "hospitalized"})
        cook_ad_df.to_csv("./instances/cook/cook_ad_2021_idph.csv")
    else:
        cook_data = data.loc[data['CovidRegionID'].isin(["IL"])]
        # cook_data = x.groupby(["date"]).aggregate({"count_rolling_7day_mean":sum}).sort_values( by=['date']).reset_index()
        cook_data["date"] = cook_data["date"].dt.strftime("%m/%d/%y")
        print(cook_data)
        cook_ad_df = cook_data[["date", "count_rolling_7day_mean"]]
        cook_ad_df = cook_ad_df.rename(columns={"count_rolling_7day_mean": "hospitalized"})
        cook_ad_df.to_csv("./instances/cook/IL_ad_2021_idph.csv")

    # print(cook_data)

def process_2021_hosp_data(is_cook=True, region= "1"):
    data = pd.read_csv("./instances/cook/EMresourcesdata_WW_surveilance.csv", parse_dates=["reported_date"])
    if is_cook is True:
        data = data.loc[data['CovidRegionID'].isin(["10", "11"])]
        out_name = "cook"
    elif region is not None:
        data = data.loc[data['CovidRegionID'].isin([region])]
        out_name = "region_" + region
    else:
        out_name = "IL"
    data["reported_date"] = data["reported_date"].dt.strftime("%m/%d/%y")
    data['reported_date'] = pd.to_datetime(data['reported_date'])
    cook_data = data.groupby(["reported_date"]).aggregate({"TotalBedsByCOVID":sum, "NonICUCovidPatients":sum, "ICUBedsCovidPatients":sum}).sort_values( by=['reported_date']).reset_index()
    cook_data["reported_date"] = cook_data["reported_date"].dt.strftime("%m/%d/%y")
    print(cook_data)
    cook_icu_df = cook_data[["reported_date", "ICUBedsCovidPatients"]]
    cook_icu_df = cook_icu_df.rename(columns={"reported_date":"date", "ICUBedsCovidPatients": "hospitalized"})
    cook_icu_df.to_csv("./instances/cook/" + out_name + "_icu_2021_idph.csv")
    cook_total_beds_df = cook_data[["reported_date", "TotalBedsByCOVID"]]
    cook_total_beds_df = cook_total_beds_df.rename(columns={"reported_date":"date", "TotalBedsByCOVID": "hospitalized"})
    cook_total_beds_df.to_csv("./instances/cook/" + out_name + "_total_beds_2021_idph.csv")
    cook_general_ward_df = cook_data[["reported_date", "NonICUCovidPatients"]]
    cook_general_ward_df = cook_general_ward_df.rename(columns={"reported_date":"date", "NonICUCovidPatients": "hospitalized"})
    cook_general_ward_df.to_csv("./instances/cook/" + out_name + "_general_ward_2021_idph.csv")

def process_2020_hosp_data(is_cook=True, region=None):
    data = pd.read_csv("./instances/cook/EMresource_hosp_WW_byzip.csv", parse_dates=["Date"])
    if is_cook is True:
        data = data.loc[data['Region_st'].isin(["Region10", "Region11"])]
        out_name = "cook"
    elif region is not None:
        if region < 10:
            region_st = "Region0" + str(region)
        else:
            region_st = "Region" + str(region)
        data = data.loc[data['Region_st'].isin([region_st])]
        out_name = "region_" + str(region)
    else:
        out_name = "IL"
    data["Date"] = data["Date"].dt.strftime("%m/%d/%y")
    data['Date'] = pd.to_datetime(data['Date'])
    cook_data = data.groupby(["Date"]).aggregate({"Suspected COVID Pt's in ICU:": sum, "COVID or PUI in non-ICU beds":sum, "Confirmed COVID Pt's in ICU:":sum}).sort_values( by=['Date']).reset_index()
    print(cook_data)
    cook_icu_df = cook_data[["Date", "Confirmed COVID Pt's in ICU:"]]
    cook_icu_df = cook_icu_df.rename(columns={"Date":"date", "Confirmed COVID Pt's in ICU:": "hospitalized"})
    cook_icu_df["date"] = cook_icu_df["date"].dt.strftime("%m/%d/%y")
    cook_icu_df.to_csv("./instances/cook/" + out_name + "_icu_confirmed_2020_idph.csv")
    
    cook_icu_sus_df = cook_data[["Date", "Suspected COVID Pt's in ICU:"]]
    cook_icu_sus_df = cook_icu_sus_df.rename(columns={"Date":"date", "Suspected COVID Pt's in ICU:": "hospitalized"})
    cook_icu_sus_df["date"] = cook_icu_sus_df["date"].dt.strftime("%m/%d/%y")
    cook_icu_sus_df.to_csv("./instances/cook/" + out_name + "_icu_suspected_2020_idph.csv")

    cook_general_ward_df = cook_data[["Date", "COVID or PUI in non-ICU beds"]]
    cook_general_ward_df = cook_general_ward_df.rename(columns={"Date":"date", "COVID or PUI in non-ICU beds": "hospitalized"})
    cook_general_ward_df.to_csv("./instances/cook/" + out_name + "_general_ward_2020_idph.csv")

    cook_icu_combined = cook_icu_df['hospitalized'] + cook_icu_sus_df['hospitalized']
    cook_icu_combined_df = pd.DataFrame({'date': cook_icu_df['date'], 'hospitalized': cook_icu_combined})
    cook_icu_combined_df.to_csv("./instances/cook/" + out_name + "_icu_combined_2020_idph.csv")
    

    cook_total_beds = cook_icu_df['hospitalized'] + cook_general_ward_df['hospitalized']
    cook_total_beds_df = pd.DataFrame({'date': cook_icu_df['date'], 'hospitalized': cook_total_beds})
    cook_total_beds_df.to_csv("./instances/cook/" + out_name + "_total_beds_2020_idph.csv")

    cook_total_beds_combined = cook_icu_combined_df['hospitalized'] + cook_general_ward_df['hospitalized']
    cook_total_beds_combined_df = pd.DataFrame({'date': cook_icu_df['date'], 'hospitalized': cook_total_beds_combined})
    cook_total_beds_combined_df.to_csv("./instances/cook/" + out_name + "_total_beds_combined_2020_idph.csv")


def process_2020_admission_data(is_cook=True):
    data = pd.read_csv("./instances/cook/Cases_WW_2022_11_10.csv", parse_dates=["CASE_OPEN_DATE"])
    if is_cook is True:
        data = data.loc[data['COUNTY'].isin(["Cook"])]
        out_name = "cook"
    else:
        out_name = "IL"

    data = data.sort_values(by=['CASE_OPEN_DATE'])
    print(data)


def hosp_ratio():
    cook_hosp = pd.read_csv("./instances/cook/cook_total_beds_2021_idph.csv", parse_dates=["date"])
    state_hosp = pd.read_csv("./instances/cook/IL_total_beds_2021_idph.csv", parse_dates=["date"])
    hosp_ratio_cook_il = cook_hosp["hospitalized"] / state_hosp["hospitalized"]
    cook_icu = pd.read_csv("./instances/cook/cook_icu_2021_idph.csv", parse_dates=["date"])
    state_icu = pd.read_csv("./instances/cook/IL_icu_2021_idph.csv", parse_dates=["date"])
    icu_ratio_cook_il = cook_icu["hospitalized"] / state_icu["hospitalized"]

    ratio_diff = ((icu_ratio_cook_il - hosp_ratio_cook_il)/hosp_ratio_cook_il)[:100].describe()
    print(ratio_diff)
    fig, ax = plt.subplots()
    plt.plot(cook_hosp["date"], hosp_ratio_cook_il, label="hosp")
    plt.plot(cook_hosp["date"], icu_ratio_cook_il, label="icu")
    myFmt = mdates.DateFormatter('%y-%m')
    ax.xaxis.set_major_formatter(myFmt)
    plt.ylabel("cook / IL ratio")
    plt.legend()
    plt.show()
    cook_region_sum_hosp = pd.read_csv("./instances/cook/cook_total_beds_region_sum.csv")
    
def admission_hosp_ratio():
    cook_hosp_2020 = pd.read_csv("./instances/cook/smoothed_region_sum_estimated_no_may_hosp.csv", parse_dates=["date"])
    cook_icu_2020 = pd.read_csv("./instances/cook/smoothed_region_sum_estimated_no_may_icu.csv", parse_dates=["date"])
    cook_ad_2020 = pd.read_csv("./instances/cook/sensitive_line_list_admission_processed_7_day.csv", parse_dates=["date"])
    
    cook_hosp_icu_ratio_2020 = cook_icu_2020["hospitalized"] / cook_hosp_2020["hospitalized"]
    cook_ad_hosp_ratio_2020 = (cook_hosp_2020["hospitalized"] - cook_icu_2020["hospitalized"]) / cook_ad_2020["hospitalized"]
    cook_ad_hosp_total_ratio_2020 = cook_hosp_2020["hospitalized"] / cook_ad_2020["hospitalized"]
    # print(cook_ad_hosp_ratio_2020)
    cook_hosp = pd.read_csv("./instances/cook/cook_general_ward_2021_idph.csv", parse_dates=["date"])
    cook_icu = pd.read_csv("./instances/cook/cook_icu_2021_idph.csv",parse_dates=["date"])
    cook_ad = pd.read_csv("./instances/cook/cook_ad_2021_idph.csv",parse_dates=["date"])
    cook_icu_hosp_ratio_idph = cook_icu["hospitalized"] / (cook_hosp["hospitalized"] + cook_icu["hospitalized"]) 
    austin_hosp = pd.read_csv("./instances/austin/austin_real_hosp_updated.csv", parse_dates=["date"])
    austin_icu = pd.read_csv("./instances/austin/austin_real_icu_updated.csv" , parse_dates=["date"])
    austin_ad = pd.read_csv("./instances/austin/austin_hosp_ad_updated.csv", parse_dates=["date"])
    austin_hosp_icu_ratio =  austin_icu["hospitalized"] / austin_hosp["hospitalized"]
    cook_ad_hosp_ratio = cook_hosp["hospitalized"] / cook_ad["hospitalized"]
    austin_ad_hosp_ratio = (austin_hosp["hospitalized"] - austin_icu["hospitalized"]) / austin_ad["hospitalized"]
    austin_ad_hosp_total_ratio = austin_hosp["hospitalized"]/ austin_ad["hospitalized"]
    
    fig, ax = plt.subplots()
    plt.plot(austin_hosp["date"][100:], austin_hosp_icu_ratio[100:], label="austin")
    plt.plot(cook_hosp_2020["date"][100:], cook_hosp_icu_ratio_2020[100:], label="cook")
    plt.plot(cook_icu["date"], cook_icu_hosp_ratio_idph,label="cook-idph" )
    myFmt = mdates.DateFormatter('%y-%m')
    ax.xaxis.set_major_formatter(myFmt)
    plt.legend()
    plt.ylabel("icu / total_beds ")
    plt.savefig("./plots/austin_vs_cook/icu_total_beds_ratio.png")
    plt.show()

    # plt.plot(cook_hosp["date"], cook_ad_hosp_ratio[:-1])
    plt.plot(austin_hosp["date"][:164], austin_ad_hosp_ratio[:164], label="austin")
    plt.plot(cook_ad_2020["date"][:], cook_ad_hosp_ratio_2020[:164], label="cook")
    plt.legend()
    plt.ylabel("general ward census / admission")
    myFmt = mdates.DateFormatter('%y-%m')
    ax.xaxis.set_major_formatter(myFmt)
    plt.savefig("./plots/austin_vs_cook/general_ward_ad_ratio.png")
    plt.show()


    plt.plot(austin_hosp["date"][:164], austin_ad_hosp_total_ratio[:164], label="austin")
    plt.plot(cook_ad_2020["date"][:], cook_ad_hosp_total_ratio_2020[:164], label="cook")
    plt.legend()
    plt.ylabel("hospital census / admission")
    plt.savefig("./plots/austin_vs_cook/hosp_ad_ratio.png")
    myFmt = mdates.DateFormatter('%y-%m')
    ax.xaxis.set_major_formatter(myFmt)
    plt.show()

    # plt.plot(cook_region_sum_hosp["date"], cook_region_sum_hosp["hospitalized"])
    # plt.plot(cook_hosp["date"], cook_hosp["hospitalized"])
    # plt.show()

# process_2020_hosp_data(True)
# process_2020_hosp_data(False)
# process_2020_admission_data()
# admission_hosp_ratio()

# ww_admission_data(is_cook=False)
# hosp_ratio()
# process_2021_hosp_data(True)


# for i in range(11,12):
#     regions = [i]
#     out_name = "region_" + str(i)
#     scrape_admission_total_beds_regions(regions, out_name)


# for i in range(11,12):
#     region = str(i)
#     process_2021_hosp_data(is_cook=False, region=region)

# process_2021_hosp_data(is_cook=True)


# for i in range(11,12):
#     region = i
#     process_2020_hosp_data(is_cook=False, region=region)

# process_2020_hosp_data(is_cook=True)


# scrape_admission_total_beds_regions([11], "chicago")
# scrape_admission_total_beds_regions([10], "subcook")
# calculate_cook_IL_ratio()
# scrape_state_hosp_data()
# add_shifted_scaled_austin_data_to_cook()
# hosp_region_sum_estimation()
# icu_hosp_ratio()

line_list_admission(in_file_name="sensitivezips_hosp_adm-comb-221205.csv", is_cook=False)
# line_list_admission(in_file_name="sensitivezips_hosp_adm-old.csv", is_cook=True)

# line_list_admission(in_file_name="sensitivezips_hosp_adm-comb-221109.csv", is_cook=False)
# line_list_admission(in_file_name="sensitivezips_hosp_adm-old.csv", is_cook=False)
# process_line_list_admission()

# ww_admission_data()