'''
Alternative Care Site (ACS)
'''

import pickle
import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from VaccineAllocation import load_config_file,config_path
from reporting.plotting import plot_multi_tier_sims
from reporting.report_pdf import generate_report
from reporting.output_processors import build_report
from pipelinemultitier import read_hosp, multi_tier_pipeline
import csv

def getACS_util(reg_hosp, profiles, T):
    # output the expected ACS utilization number
    useList = []
    maxUseList = []
    maxDayList = []
    capList = []
    overList = []
    noTriggered = 0
    for p in profiles:
        # time series of over regular capacity
        overCap_reg = np.maximum(np.sum(p['IHT'], axis = (1,2)) - reg_hosp, 0)
        # time series of over ACS capacity
        overCap_ACS = np.maximum(np.sum(p['IHT'], axis = (1,2)) - p["capacity"],0)
        # time series of ACS usage
        acs_usage = overCap_reg - overCap_ACS
        # time series of ACS capacity
        acs_cap = np.array(p["capacity"]) - reg_hosp
        # number of paths with ACS triggered
        if p['acs_triggered'] and len(np.unique(p['capacity'][:T])) > 1:
            noTriggered += 1
        
        # ACS required for this path
        maxUseList.append(np.max(overCap_reg[:T]))
        # number of days requiring ACS for this path
        maxDayList.append(np.sum(overCap_reg[:T] > 0))
        # total number of ACS usage for this path
        useList.append(np.sum(acs_usage[:T]))
        # total capacity of ACS usage for this path
        capList.append(np.sum(acs_cap[:T]))
        # total number of ACS unsatisfaction for this path
        overList.append(np.sum(overCap_ACS[:T]))
    meanUse = np.mean(useList)
    meanUtil = np.nanmean(np.array(useList)/np.array(capList))
    #breakpoint()
    return useList,maxUseList,maxDayList,capList,overList,meanUse,meanUtil,noTriggered

def getACS_util_ICU(reg_hosp, profiles, T, t_start = 0):
    # output the expected ACS utilization number
    useList = []
    maxUseList = []
    maxDayList = []
    capList = []
    overList = []
    noTriggered = 0
    for p in profiles:
        # time series of over regular capacity
        overCap_reg = np.maximum(np.sum(p['ICU'], axis = (1,2))[t_start:] - reg_hosp, 0)
        # time series of over ACS capacity
        overCap_ACS = np.maximum(np.sum(p['ICU'], axis = (1,2))[t_start:] - p["capacity"][t_start:],0)
        # time series of ACS usage
        acs_usage = overCap_reg - overCap_ACS
        # time series of ACS capacity
        acs_cap = np.array(p["capacity"]) - reg_hosp
        # number of paths with ACS triggered
        if p['acs_triggered'] and len(np.unique(p['capacity'][:T])) > 1:
            noTriggered += 1
        
        # ACS required for this path
        maxUseList.append(np.max(overCap_reg[:T]))
        # number of days requiring ACS for this path
        maxDayList.append(np.sum(overCap_reg[:T] > 0))
        # total number of ACS usage for this path
        useList.append(np.sum(acs_usage[:T]))
        # total capacity of ACS usage for this path
        capList.append(np.sum(acs_cap[:T]))
        # total number of ACS unsatisfaction for this path
        overList.append(np.sum(overCap_ACS[:T]))
    meanUse = np.mean(useList)
    meanUtil = np.nanmean(np.array(useList)/np.array(capList))
   # breakpoint()
    return useList,maxUseList,maxDayList,capList,overList,meanUse,meanUtil,noTriggered

def getACS_reppath(profiles,reg_cap):
    dateList = []
    for i in range(len(profiles)):
        if len(np.where(np.array(profiles[i]['capacity']) > reg_cap)[0]) > 0:
            dateList.append([i,np.where(np.array(profiles[i]['capacity']) > reg_cap)[0][0]])
        else:
            dateList.append([i,10000])
    dateList.sort(key = lambda x: x[1]) 
    return dateList
    
def getACS_gap(profiles, reg_cap):
    outList = []
    for i in range(300):
        IHTList = np.sum(profiles[i]['IHT'], axis = (1,2))
        capList = np.array(profiles[i]['capacity'])
        profList = [i]
        if len(np.where(IHTList > reg_cap)[0]) > 0:
            profList.append(np.where(IHTList > reg_cap)[0][0])
        else:
            profList.append(10000)
        if len(np.where(capList > reg_cap)[0]) > 0:
            profList.append(np.where(capList > reg_cap)[0][0])
        else:
            profList.append(10000)
        outList.append(profList)
    return outList

def getACS_gap_ICU(profiles, reg_cap):
    outList = []
    for i in range(len(profiles)):
        ICUList = np.sum(profiles[i]['ICU'], axis = (1,2))
        capList = np.array(profiles[i]['capacity'])
        profList = [i]
        if len(np.where(ICUList > reg_cap)[0]) > 0:
            profList.append(np.where(ICUList > reg_cap)[0][0])
        else:
            profList.append(10000)
        if len(np.where(capList > reg_cap)[0]) > 0:
            profList.append(np.where(capList > reg_cap)[0][0])
        else:
            profList.append(10000)
        outList.append(profList)
    return outList


fileList = os.listdir("output")

# load Austin real hospitalization
file_path = "instances/austin/austin_real_hosp_updated.csv"
start_date = dt(2020,2,28)
real_hosp = read_hosp(file_path, start_date)
hosp_beds_list = None
file_path = "instances/austin/austin_hosp_ad_updated.csv"
hosp_ad = read_hosp(file_path, start_date, "admits")
file_path = "instances/austin/austin_real_icu_updated.csv"
real_icu = read_hosp(file_path, start_date)
hosp_beds_list = None

# newly defined color
add_tiers = {0.62379925: '#ffE000',
              0.6465315: '#ffC000',
              0.66926375: '#ffA000',
              0.71472825: '#ff6000',
              0.7374605: '#ff4000',
              0.76019275: '#ff2000'
    }


fi = open("/Users/nazlicanarslan/Desktop/github_clones/COVID19-vaccine/VaccineAllocation/output/v-acs.csv","w",newline="")
csvWriter = csv.writer(fi,dialect='excel')
csvWriter.writerow(['Case_Name','ACS_Quantity','ACS_Trigger','Scenario with ACS Triggered',
                    'Infeasible Scenarios','Mean ACS Usage', 'Mean ACS Util Rate', 
                    'Max No of Days Requiring ACS','95% Days Requiring ACS',
                    'Max ACS Required', '95% ACS Required', 'Original Unmet Mean', 'Original Unmet Median', 'Original Unmet Std', 'Original Unmet 5%', 'Original Unmet 95%'])

trend_comp = True

for instance_raw in fileList:
    if ".p" in instance_raw:
        try:
            instance_name = instance_raw[:-2]
            file_path = f'output/austin_acs_071321/{instance_name}.p'
            #breakpoint()  
            with open(file_path, 'rb') as outfile:
                read_output = pickle.load(outfile)
               
            instance, interventions, best_params, best_policy, vaccines, profiles, sim_output, expected_cost, config, seeds_info = read_output
      
            if "11_13" in instance_name:
                end_history = dt(2020,11,13)
            else:
                end_history = dt(2021,7,12)
            t_start = (end_history - start_date).days
            
            acs_results = getACS_util(instance.hosp_beds,profiles,601)
            print("====================================================")
            print(instance_name)
            case_name = str(instance.transmission_file)[:-4]
            print(case_name)
            #instance_name = "austin_{}_{}".format(case_name,best_policy.acs_Q)
            #os.rename(file_path, r"/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ACS_Analysis_11_13/" + instance_name + ".p")
            
            print("ACS Trigger: ", best_policy.acs_thrs)
            print("ACS Quantity: ", best_policy.acs_Q)
            
            infeas_scen = np.sum(np.array(acs_results[4]) > 0)
            print("Infeasible Scenarios Testing: ", infeas_scen)
            print("Mean ACS Usage: ", acs_results[5])
            
            mean_util_rate = np.round(acs_results[6]*100,2)
            print("Mean ACS Utilization Rate: ", mean_util_rate)
            print("Number of paths hitting the trigger: ", acs_results[7])
            
            print("Maximum number of days requiring ACS", np.max(acs_results[2]))
            print("95 Percentile of days requiring ACS", np.percentile(acs_results[2],95))
            print("Maximum ACS required", np.max(acs_results[1]))
            print("95 Percentile of ACS required", np.percentile(acs_results[1],95))
            
            n_replicas = len(profiles)
            unmet_IHT = [np.sum(np.maximum(np.sum(profiles[i]['IHT'],axis = (1,2)) - 1500,0)) for i in range(300)]
            over_mean = np.mean(unmet_IHT)
            over_median = np.median(unmet_IHT)
            over_std = np.std(unmet_IHT)
            over_5P = np.percentile(unmet_IHT,5)
            over_95P = np.percentile(unmet_IHT,95)
            
            data = [case_name, best_policy.acs_Q, best_policy.acs_thrs, acs_results[7],
                    infeas_scen, acs_results[5], mean_util_rate, 
                    np.max(acs_results[2]), np.percentile(acs_results[2],95),
                    np.max(acs_results[1]), np.percentile(acs_results[1],95),over_mean,over_median,over_std,over_5P,over_95P]

            csvWriter.writerow(data)
            dateList = getACS_reppath(profiles,1500)
            if not trend_comp:
                cpid = dateList[14][0]
            else:
                cpid = 0
            IHD_plot = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['IHT'],
                            plot_right_axis=[],
                            T=601,
                            interventions=interventions,
                            show=False,
                            align_axes=True,
                            plot_triggers=False,
                            plot_ACS_triggers=True,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=best_policy.acs_Q + 2000,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            add_tiers=add_tiers,
                            real_new_admission=real_hosp,
                            real_hosp_or_icu=real_hosp,
                            t_start = t_start,
                            is_representative_path=False,
                            central_path_id = cpid,
                            cap_path_id = cpid,
                            history_white = True,
                            acs_fill = True,
                            )
            IYIH_plot = plot_multi_tier_sims(instance_name,
                                  instance,
                                  best_policy,
                                  profiles, ['sim'] * len(profiles),
                                  real_hosp,
                                  plot_left_axis=['ToIHT'],
                                  plot_right_axis=[],
                                  T=601,
                                  interventions=interventions,
                                  show=False,
                                  align_axes=False,
                                  plot_triggers=False,
                                  plot_ACS_triggers=True,
                                  plot_trigger_annotations=False,
                                  plot_legend=False,
                                  y_lim=300,
                                  policy_params=best_params,
                                  n_replicas=n_replicas,
                                  config=config,
                                  hosp_beds_list=hosp_beds_list,
                                  real_new_admission=hosp_ad,
                                  add_tiers=add_tiers,
                                  t_start = t_start,
                                  central_path_id = cpid,
                                  cap_path_id = cpid,
                                  history_white = True,
                                  no_fill = True,
                                  #no_center = True
                                  )
        except:
            pass
        
fi.close()



# ICU expansion analysis
fileList = os.listdir("output")

# load Austin real hospitalization
file_path = "instances/austin/austin_real_hosp_updated.csv"
start_date = dt(2020,2,28)
real_hosp = read_hosp(file_path, start_date)
hosp_beds_list = None
file_path = "instances/austin/austin_hosp_ad_updated.csv"
hosp_ad = read_hosp(file_path, start_date, "admits")
file_path = "instances/austin/austin_real_icu_updated.csv"
real_icu = read_hosp(file_path, start_date)
hosp_beds_list = None


fi = open("/Users/nazlicanarslan/Desktop/github_clones/COVID19-vaccine/VaccineAllocation/output/v-acs_ICU.csv","w",newline="")
csvWriter = csv.writer(fi,dialect='excel')
csvWriter.writerow(['Case_Name','ACS_Quantity','ACS_Trigger','Scenario with ACS Triggered',
                    'Infeasible Scenarios','Mean ACS Usage', 'Mean ACS Util Rate', 
                    'Max No of Days Requiring ACS','95% Days Requiring ACS',
                    'Max ACS Required', '95% ACS Required', 'Original Unmet Mean', 'Original Unmet Median', 'Original Unmet Std', 'Original Unmet 5%', 'Original Unmet 95%'])

trend_comp = True

for instance_raw in fileList:
    if ".p" in instance_raw:
       # try:
            instance_name = instance_raw[:-2]
            file_path = f'output/{instance_name}.p'
            with open(file_path, 'rb') as outfile:
                read_output = pickle.load(outfile)  
            instance, interventions, best_params, best_policy, vaccines, profiles, sim_output, expected_cost, config, seeds_info = read_output
            
            # if "11_13" in instance_name:
            #     end_history = dt(2020,11,13)
            # else:
            #     end_history = dt(2020,7,12)
            end_history = dt(2021,11,24)
            t_start = (end_history - start_date).days
            
            acs_results = getACS_util_ICU(instance.icu,profiles,601,t_start)
            print("====================================================")
            print(instance_name)
            case_name = str(instance.transmission_file)[str(instance.transmission_file).find("/instances/austin")+len("/instances/austin")+1:-4]
            print(case_name)
            instance_name = "austin_{}_{}_{}_{}".format(case_name,best_policy.acs_Q,best_policy.acs_type, best_policy.acs_thrs)
            #os.rename(file_path, r"/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ACS_Analysis_11_13/" + instance_name + ".p")
            
            print("ACS Trigger: ", best_policy.acs_thrs)
            print("ACS Quantity: ", best_policy.acs_Q)
            
            infeas_scen = np.sum(np.array(acs_results[4]) > 0)
            print("Infeasible Scenarios Testing: ", infeas_scen)
            print("Mean ACS Usage: ", acs_results[5])
            
            mean_util_rate = np.round(acs_results[6]*100,2)
            print("Mean ACS Utilization Rate: ", mean_util_rate)
            print("Number of paths hitting the trigger: ", acs_results[7])
            
            print("Maximum number of days requiring ACS", np.max(acs_results[2]))  
            print("95 Percentile of days requiring ACS", np.percentile(acs_results[2],95))
            print("90 Percentile of days requiring ACS", np.percentile(acs_results[2],90))
            print("80 Percentile of days requiring ACS", np.percentile(acs_results[2],80))
            print("50 Percentile of days requiring ACS", np.percentile(acs_results[2],50))
            print("Maximum ACS required", np.max(acs_results[1]))
            print("95 Percentile of ACS required", np.percentile(acs_results[1],95))
            print("90 Percentile of ACS required", np.percentile(acs_results[1],90))
            print("80 Percentile of ACS required", np.percentile(acs_results[1],80))
            print("50 Percentile of ACS required", np.percentile(acs_results[1],50))
            
            n_replicas = len(profiles)
            unmet_ICU = [np.sum(np.maximum(np.sum(profiles[i]['ICU'],axis = (1,2))[t_start:] - 200,0)) for i in range(len(profiles))]
            over_mean = np.mean(unmet_ICU)
            over_median = np.median(unmet_ICU)
            over_std = np.std(unmet_ICU)
            over_5P = np.percentile(unmet_ICU,5)
            over_95P = np.percentile(unmet_ICU,95)
            
            data = [case_name, best_policy.acs_Q, best_policy.acs_thrs, acs_results[7],
                    infeas_scen, acs_results[5], mean_util_rate, 
                    np.max(acs_results[2]), np.percentile(acs_results[2],95),
                    np.max(acs_results[1]), np.percentile(acs_results[1],95),over_mean,over_median,over_std,over_5P,over_95P,
                    np.percentile(acs_results[1],50),np.percentile(acs_results[1],75)]
            csvWriter.writerow(data)
            dateList = getACS_reppath(profiles,1500)
            if not trend_comp:
                cpid = dateList[14][0]
            else:
                cpid = 0
            # IHD_plot = plot_multi_tier_sims(instance_name,
            #                 instance,
            #                 best_policy,
            #                 profiles, ['sim'] * len(profiles),
            #                 real_hosp,
            #                 plot_left_axis=['ICU'],
            #                 plot_right_axis=[],
            #                 T=601,
            #                 interventions=interventions,
            #                 show=True,
            #                 align_axes=True,
            #                 plot_triggers=False,
            #                 plot_trigger_annotations=False,
            #                 plot_legend=False,
            #                 y_lim=best_policy.acs_Q + 500,
            #                 policy_params=best_params,
            #                 n_replicas=n_replicas,
            #                 config=config,
            #                 real_new_admission=real_hosp,
            #                 real_hosp_or_icu=real_icu,
            #                 t_start = t_start,
            #                 is_representative_path=False,
            #                 central_path_id = cpid,
            #                 cap_path_id = cpid,
            #                 history_white = True,
            #                 acs_type = 'ICU'
            #                 )
            # IYIH_plot = plot_multi_tier_sims(instance_name,
            #                       instance,
            #                       best_policy,
            #                       profiles, ['sim'] * len(profiles),
            #                       real_hosp,
            #                       plot_left_axis=['ToIHT'],
            #                       plot_right_axis=[],
            #                       T=601,
            #                       interventions=interventions,
            #                       show=False,
            #                       align_axes=False,
            #                       plot_triggers=False,
            #                       plot_ACS_triggers=True,
            #                       plot_trigger_annotations=False,
            #                       plot_legend=False,
            #                       y_lim=250,
            #                       policy_params=best_params,
            #                       n_replicas=n_replicas,
            #                       config=config,
            #                       hosp_beds_list=hosp_beds_list,
            #                       real_new_admission=hosp_ad,
            #                       t_start = t_start,
            #                       central_path_id = cpid,
            #                       cap_path_id = cpid,
            #                       history_white = True
            #                       )
        # except:
        #     pass
        
fi.close()
