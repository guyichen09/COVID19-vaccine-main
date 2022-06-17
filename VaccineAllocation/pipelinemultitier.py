import pickle
import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from VaccineAllocation import load_config_file,config_path, plots_path
from reporting.plotting import plot_multi_tier_sims
from reporting.report_pdf import generate_report
from reporting.output_processors import build_report_tiers

def read_hosp(file_path, start_date, typeInput="hospitalized"):
    with open(file_path, 'r') as hosp_file:
        df_hosp = pd.read_csv(
            file_path,
            parse_dates=['date'],
            date_parser=pd.to_datetime,
        )
    # if hospitalization data starts before start_date 
    if df_hosp['date'][0] <= start_date:
        df_hosp = df_hosp[df_hosp['date'] >= start_date]
        real_hosp = list(df_hosp[typeInput])
    else:
        real_hosp = [0] * (df_hosp['date'][0] - start_date).days + list(df_hosp[typeInput])
    
    return real_hosp
    
def icu_pipeline(file_path, 
                 instance_name, 
                 real_hosp=None, 
                 real_admit=None,
                 real_icu=None,
                 is_CDC=False,
                 **kwargs):

    plot_end = kwargs['plot_end']
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance, interventions, best_params, best_policy, vaccines, profiles, sim_output, expected_cost, config, seeds_info = read_output

    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
    profiles = [p for p in profiles]

    n_replicas = len(profiles)
    T = instance.cal.calendar_ix[plot_end]
    plot_trigger_ToIHT = False
    plot_trigger_ToIHT = True
     
    moving_avg_len = config['moving_avg_len']
    N = instance.N
    #real_icu_ratio = [real_icu[i]/(real_hosp[i]) for i in range(len(real_icu))  if real_hosp[i] != 0]
    real_ToIHT_total = [np.array(real_admit)[i: min(i + moving_avg_len, T)].sum()* 100000/np.sum(N, axis=(0,1))  for i in range(T-moving_avg_len)]
    real_ToIHT_average = [np.array(real_admit)[i: min(i + moving_avg_len, T)].mean() for i in range(T-moving_avg_len)]
    real_percent_IH = [np.array(real_hosp)[i: min(i + moving_avg_len, T)].mean()/instance.hosp_beds for i in range(T-moving_avg_len)]
    real_case_total = [np.array(case_ad)[i: min(i + moving_avg_len, T)].sum()* 100000/np.sum(N, axis=(0,1)) for i in range(T-moving_avg_len)]
    
    y_lim = {
        'ToIY_moving': 1500,
        'ToIHT_total': 60,
        'ToIHT_moving': 200,
        'ICU': 500,
        'IHT_moving': 1,
        'D': 1500}
 
    cap = {
        'ToIY_moving': 200,
        'ToIHT_total': None,
        'ToIHT_moving': None,
        'ICU': 150,
        'IHT_moving': None,
        'D': None}
    
    real_data = {'ToIY_moving': real_case_total,
                 'ToIHT_total': real_ToIHT_total,
                 'ToIHT_moving': real_ToIHT_average,
                 'ICU': real_icu,
                 'IHT_moving': real_percent_IH,
                 'D': real_death}
    
    if is_CDC:
        comp_to_plot = {'ToIY_moving', 'ToIHT_total', 'IHT_moving', 'ToIHT_moving', 'ICU', 'D'}
    else:
        comp_to_plot = {'ToIHT_moving', 'ICU', 'D'}
        
    for comp in comp_to_plot:
        plot = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            plot_left_axis=[comp],
                            plot_right_axis=[],
                            interventions=interventions,
                            show=True,
                            align_axes=True,
                            plot_triggers=plot_trigger_ToIHT,
                            plot_trigger_annotations=True if comp in {'ToIHT_total', 'ToIHT_moving', 'IHT_moving'} else False,
                            plot_legend=False,
                            y_lim=y_lim[comp],
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            real_data=real_data[comp],
                            vertical_fill=  True if comp in {'ToIY_moving', 'D'} else False,
                            dali_plot=True if comp == 'ICU' else False,
                            capacity=cap[comp],
                            **kwargs
                            )  
 
def report(file_path, 
           instance_name, 
           real_hosp=None, 
           real_admit=None, 
           real_icu=None,
           **kwargs):
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance, interventions, best_params, best_policy, vaccines, profiles, sim_output, expected_cost, config, seeds_info = read_output
    icu_beds_list = [instance.icu]
    build_report_tiers(instance_name,
                        instance,
                        best_policy,
                        profiles,
                        config=config,
                        interventions=interventions,
                        policy_params=best_params,
                        stat_start=instance.cal.calendar[t_start+1],
                        stat_end=dt(2020,9,1),
                        central_id_path=kwargs['central_id_path'])
    
if __name__ == "__main__":
    # list all .p files from the output folder
    fileList = os.listdir("output")
    for instance_raw in fileList:
        if ".p" in instance_raw:
            if "austin" in instance_raw:
                file_path = "instances/austin/austin_real_hosp_updated.csv"
                start_date = dt(2020,2,28)
                end_history = dt(2020,4,30)
                t_start =(end_history - start_date).days
                real_hosp = read_hosp(file_path, start_date)
                file_path = "instances/austin/austin_hosp_ad_updated.csv"
                hosp_ad = read_hosp(file_path, start_date, "admits")
                file_path = "instances/austin/austin_real_case.csv"
                case_ad = read_hosp(file_path, start_date, "admits")
                file_path = "instances/austin/austin_real_icu_updated.csv"
                real_icu = read_hosp(file_path, start_date)
                file_path = "instances/austin/austin_real_cum_total_death.csv"
                real_death = read_hosp(file_path, start_date)
                
            kwargs_plotting = {
                'central_id_path': 0,
                'acs_type': 'ICU',
                'plot_end': dt(2020,9,1),
                'is_representative_path': False,
                'history_white': True,
                't_start': t_start,
                'x_axis_tick': 2}
            
            instance_name = instance_raw[:-2]
            path_file = f'output/{instance_name}.p'
            if 'CDC' in path_file:
                is_CDC = True
            else:
                is_CDC = False
            icu_pipeline(path_file, 
                          instance_name, 
                          real_hosp, 
                          hosp_ad, 
                          real_icu,
                          is_CDC=is_CDC,
                          **kwargs_plotting)
            

            kwargs_reporting = {'central_id_path': 0}
            report(path_file, 
                    instance_name, 
                    real_hosp, 
                    hosp_ad, 
                    real_icu,
                    **kwargs_reporting
                    )
   
            
     
            