import urllib.request
import json
import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

filepath = "./instances/cook/"
regions = [10,11]

# json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetResurgenceData?regionID='+str(11)+'&daysIncluded=0')
# data = json.loads(json_url.read())

# total_admission_df = pd.DataFrame(0, index=np.arange(len(data)), columns=['date', 'hospitalized'])



# def sum_admission(df1, df2):
#     if df1['date'] == df2['date']:
#         return df1['admission'] + df2['admission']
# admission_dict = {}
# icu_dict = {}
# general_ward = {}
# for region in regions:
#     # read this file - this is where they put the data.  It's a json file, so need to re-format into a dataframe...
#     json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetResurgenceData?regionID='+str(region)+'&daysIncluded=0')
#     data = json.loads(json_url.read())
#     mydf = pd.json_normalize(data)
#     admission_df = mydf[['ReportDate','CLIAdmissions_RollingAvg']]
#     admission_df = admission_df.rename(columns = {"ReportDate": "date", "CLIAdmissions_RollingAvg": "hospitalized"})
    
#     admission_df["date"] = pd.to_datetime(admission_df.date, format='%Y-%m-%dT%H:%M:%S')
#     admission_df["date"] = admission_df["date"].dt.strftime("%m/%d/%y")
#     # print(data[0])
#     total_admission_df['hospitalized'] += admission_df['hospitalized']
#     total_admission_df['date'] = admission_df['date']

# total_admission_df.to_csv(os.path.join(filepath,'cook_hosp_ad.csv'))



# cook_death_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetCountyTestResultsTimeSeries?CountyName=cook')
# chicago_death_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetCountyTestResultsTimeSeries?CountyName=chicago')
# cook_death = json.loads(cook_death_url.read())
# cook_death_df = pd.json_normalize(cook_death)
# cook_death_df = cook_death_df[['ReportDate','Deaths']]
# cook_death_df = cook_death_df.rename(columns = {"ReportDate": "date", "Deaths": "hospitalized"})

# cook_death_df["date"] = pd.to_datetime(cook_death_df.date, format='%Y-%m-%dT%H:%M:%S')
# cook_death_df["date"] = cook_death_df["date"].dt.strftime("%m/%d/%y")


# chicago_death = json.loads(chicago_death_url.read())
# chicago_death_df = pd.json_normalize(chicago_death)
# chicago_death_df = chicago_death_df[['ReportDate','Deaths']]
# chicago_death_df = chicago_death_df.rename(columns = {"ReportDate": "date", "Deaths": "hospitalized"})

# chicago_death_df["date"] = pd.to_datetime(chicago_death_df.date, format='%Y-%m-%dT%H:%M:%S')
# chicago_death_df["date"] = chicago_death_df["date"].dt.strftime("%m/%d/%y")


# total_death_df = pd.DataFrame(0, index=np.arange(len(chicago_death)), columns=['date', 'hospitalized'])
# total_death_df["hospitalized"] = chicago_death_df["hospitalized"] + cook_death_df["hospitalized"]
# total_death_df["date"] = chicago_death_df["date"]
# total_death_df.to_csv(os.path.join(filepath,'cook_deaths.csv'))



# json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetHospitalUtilizationResults')
# data = json.loads(json_url.read())
# mydf = pd.json_normalize(data)

# print(mydf.info())

# icu_df = mydf[["ReportDate", "ICUInUseBedsCOVID"]]
# icu_df = icu_df.rename(columns = {"ReportDate": "date", "ICUInUseBedsCOVID": "hospitalized"})

# icu_df["hospitalized"] = icu_df["hospitalized"]
# icu_df["date"] = pd.to_datetime(icu_df.date, format='%Y-%m-%dT%H:%M:%S')
# icu_df["date"] = icu_df["date"].dt.strftime("%m/%d/%y")
# icu_df.to_csv(os.path.join(filepath,'IL_hosp_icu.csv'))


# generalward_df = mydf[["ReportDate", "TotalInUseBedsCOVID"]]
# generalward_df = generalward_df.rename(columns = {"ReportDate": "date", "TotalInUseBedsCOVID": "hospitalized"})

# generalward_df["hospitalized"] = generalward_df["hospitalized"]
# generalward_df["date"] = pd.to_datetime(generalward_df.date, format='%Y-%m-%dT%H:%M:%S')
# generalward_df["date"] = generalward_df["date"].dt.strftime("%m/%d/%y")
# generalward_df.to_csv(os.path.join(filepath,'IL_hosp.csv'))

# print(icu_df)





#     for d in data:
#         # convert report date to 
#         date = d["ReportDate"]
#         formated_date = datetime.strptime(date,'%Y-%m-%dT%H:%M:%S')
#         formated_date_str = datetime.strftime(formated_date, "%m/%d/%y")
#         if formated_date_str not in admission_dict:
#             admission_dict[formated_date_str] = 0
#         admission_dict[formated_date_str] += d["CLIAdmissions_RollingAvg"]

# admission_df = pd.DataFrame.from_dict(admission_dict)
# admission_df = admission_df.sort_values(by=["date"])

# print(admission_dict)

# hospital_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetHospitalizationResultsRegion')
# hosp_data = json.loads(hospital_url.read())
# print(hosp_data)


        
    # mydf = pd.json_normalize(data)
    # admission_df = mydf[['ReportDate','CLIAdmissions_RollingAvg']]
    # ICU_df = []
    # general_ward_df = []
    # admission_df = admission_df.rename(columns = {"ReportDate": "date", "CLIAdmissions_RollingAvg": "admission"})
    
    # admission_df["date"] = pd.to_datetime(admission_df.date, format='%Y-%m-%dT%H:%M:%S')
    # admission_df["date"] = admission_df["date"].dt.strftime("%m/%d/%y")
    # print(admission_df)
    # total_admission = df
    # mydf.to_csv(os.path.join(filepath,'regionmetrics'+str(region)+'_220415.csv'),index=False)


austin_files = ["austin_real_hosp_updated.csv",
              "austin_real_icu_updated.csv",
              "austin_hosp_ad_updated.csv",
              "austin_real_death_from_hosp_updated.csv",
              "austin_real_total_death.csv"]
cook_files = ["cook_hosp.csv",
            "cook_hosp_icu.csv",
            "cook_hosp_ad_syn.csv",
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
    # print(time_delta)
    scale_factor = cook_hosp_peak_val / austin_hosp_peak_val
    print(scale_factor)

    shifted_first_date_austin = austin_date[0] + dt.timedelta(time_delta)
    print(shifted_first_date_austin)
    first_cook_date = cook_date[0]
    print(first_cook_date)
    first_time_delta = (shifted_first_date_austin - first_cook_date).days
    # print(first_time_delta)
    austin_shifted_scaled_data = austin_hosp_data[:-first_time_delta] * scale_factor
    shifted_austin_dates = austin_date + np.timedelta64(time_delta, 'D')
    # print(shifted_austin_dates[:-first_time_delta])
    shifted_austin_dates = shifted_austin_dates[:-first_time_delta]
    dates = pd.concat([shifted_austin_dates, cook_date])
    hosp_data = pd.concat([austin_shifted_scaled_data, cook_hosp_data])
    df = pd.DataFrame({'date': dates,
                   'hospitalized': hosp_data})

    df.to_csv("./instances/cook/" + outfile_name + ".csv")

# add scaled Austin data to the cook data
# -78
# 6.142682926829268
# -84
# 4.8375
# -73
# 1.6703296703296704
# -64
# 8.071428571428571
# -64
# 7.368421052631579