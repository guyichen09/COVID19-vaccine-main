"""
Read and clean COVID-19 hospital data
"""

import pandas as pd
import numpy as np
import csv


hosp_adm = pd.read_excel('CV19Hospital_ICU_DeID_20211001.xlsx',
                             date_parser=pd.to_datetime)

hosp_adm = hosp_adm.sort_values(by=['admission_date'])

date_input = pd.read_excel('date.xlsx',
                             date_parser=pd.to_datetime)

date = [d for d in date_input['date']]
hosp_adm_unknown =  hosp_adm[hosp_adm['covid_vaccination_status'] == "Unknown"]
hosp_adm_v_first = hosp_adm[hosp_adm['covid_vaccination_status'] == "Partially Vaccinated"]
hosp_adm_v_full =  hosp_adm[hosp_adm['covid_vaccination_status'] == "Fully Vaccinated"]
hosp_adm_unv =  hosp_adm[hosp_adm['covid_vaccination_status'] == "Not Vaccinated"]

def countX(lst, x):
    return lst.count(x)

def process_data(data, file_name):
    adm_updated = []
    adm_dt = [d for d in data['admission_date']]
    discharge_dt = [d for d in data['discharge_date']]
    #breakpoint()
    cum_hosp = 0
    for d in date:
        adm = countX(adm_dt, d)
        discharge = countX(discharge_dt, d)
        cum_hosp += adm - discharge 
        adm_updated.append([d, adm,discharge , cum_hosp])
    
    header = ['date', 'adm', 'discharge', 'hosp']    
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(adm_updated)    
    

process_data(hosp_adm_unknown, 'hosp_adm_unknown.csv')
process_data(hosp_adm_v_first, 'hosp_adm_v_first.csv')
process_data(hosp_adm_v_full, 'hosp_adm_v_full.csv')
process_data(hosp_adm_unv, 'hosp_adm_unv.csv')
process_data(hosp_adm, 'hosp_adm.csv')

    

    
    
    
    