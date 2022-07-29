import numpy as np
from reporting.report_pdf import generate_report,generate_report_tier
from collections import defaultdict
import csv
import datetime as dt
from matplotlib import pyplot as plt
from VaccineAllocation import plots_path

def plot_tier_bar(instance_name, tiers, tier_hist_mean, colors):
    fig, ax = plt.subplots()
    ax.bar(tiers[1:], tier_hist_mean[1:], color=colors[1:])
    ax.set_ylabel("Proportion of days")
    ax.set_ylim(0, 1)
    plot_filename = plots_path / f'bar_{instance_name}.pdf'
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(plot_filename)
    
    plt.show()
    plt.close()
    return plot_filename
    
def build_report_tiers(instance_name,
                 instance,
                 policy,
                 profiles,
                 n_replicas=300,
                 to_email=None,
                 config=None,
                 stat_start=None,
                 stat_end=None,
                 central_id_path=0,
                 template_file = "report_template_tier.tex",
                 **kwargs):
    '''
        Gathers data to build a city report.
    '''
  
    report_data = {'instance_name': instance_name}
    report_data['CITY'] = config['city']
    
    T = instance.T
    cal = instance.cal
    population = instance.N.sum()
    interventions = kwargs['interventions']
    hosp_beds = instance.hosp_beds
    policy_params = kwargs['policy_params']
    
    report_data['START-DATE'] = cal.calendar[0].strftime("%Y-%m-%d")
    if stat_start == None:
        report_data['STATISTICS-START-DATE'] = report_data['START-DATE']
        T_start = 0
    else:
        T_start = instance.cal.calendar_ix[stat_start]
        report_data['STATISTICS-START-DATE'] = cal.calendar[T_start].strftime("%Y-%m-%d")

    T_end =  instance.cal.calendar_ix[stat_end] 
    report_data['STATISTICS-END-DATE'] = cal.calendar[T_end - 1].strftime("%Y-%m-%d")
    report_data['END-DATE'] = cal.calendar[T - 1].strftime("%Y-%m-%d")
    report_data['policy_params'] = policy_params

 
    # Transform data of interest
    states_to_report = ['S', 'E', 'IH', 'IA', 'IY', 'R', 'D', 'IYIH', 'IHT', 'ICU', 'ToIHT', 'ToICU']
    all_states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[T_start:T_end] for p in profiles)) for v in states_to_report}
    all_states_ts['z'] = np.vstack(list(p['z'][T_start:T_end] for p in profiles))
    all_states_ts['tier_history'] = np.vstack(list(p['tier_history'][T_start:T_end] for p in profiles))
    
    central_path = central_id_path
    mean_st = {v:  all_states_ts[v][central_path] if v not in ['z', 'tier_history'] else  all_states_ts[v] for v in  all_states_ts}

    assert len(all_states_ts['IHT']) >= n_replicas
    for v in all_states_ts:
        all_states_ts[v] = all_states_ts[v][:n_replicas]
    assert len(all_states_ts['IHT']) == n_replicas
    # Hospitalizations Report
    print('Hospitalization Peaks')
    hosp_peaks_vals = {}
    hosp_peaks_dates = {}
    icu_peaks_vals = {}
    icu_peaks_dates = {}
    hosp_peak_days = np.argmax(all_states_ts['IHT'], axis=1)
    hosp_peak_vals = np.take_along_axis(all_states_ts['IHT'], hosp_peak_days[:, None], axis=1)
    icu_peak_days = np.argmax(all_states_ts['ICU'], axis=1)
    icu_peak_vals = np.take_along_axis(all_states_ts['ICU'], icu_peak_days[:, None], axis=1)
    print(f'{"Percentile (%)":<15s} {"Peak IHT":<15s}  {"Date":15}')
    
    hosp_peak_mean = np.mean(hosp_peak_vals)
    report_data['MEAN-HOSP-PEAK'] = np.round(hosp_peak_mean)
    icu_peak_mean = np.mean(icu_peak_vals)
    report_data['MEAN-ICU-PEAK'] = np.round(icu_peak_mean)

    for q in [5, 50, 95, 100]:
        hosp_peak_day_percentile = int(np.round(np.percentile(hosp_peak_days, q)))
        hosp_peak_percentile = np.percentile(hosp_peak_vals, q)
        icu_peak_day_percentile = int(np.round(np.percentile(icu_peak_days, q)))
        icu_peak_percentile = np.percentile(icu_peak_vals, q)

        hosp_peaks_vals[f'HOSP-PEAK-P{q}'] = np.round(hosp_peak_percentile)
        hosp_peaks_dates[f'HOSP-PEAK-DATE-P{q}'] = cal.calendar[hosp_peak_day_percentile +  T_start].strftime("%Y-%m-%d")
        icu_peaks_vals[f'ICU-PEAK-P{q}'] = np.round(icu_peak_percentile)
        icu_peaks_dates[f'ICU-PEAK-DATE-P{q}'] = cal.calendar[icu_peak_day_percentile + T_start].strftime("%Y-%m-%d")

        print(f'{q:<15} {hosp_peak_percentile:<15.0f}  {str(cal.calendar[hosp_peak_day_percentile])}')
        print(f'{q:<15} {icu_peak_percentile:<15.0f}  {str(cal.calendar[icu_peak_day_percentile])}')
  
    report_data.update(hosp_peaks_vals)
    report_data.update(hosp_peaks_dates)
    report_data.update(icu_peaks_vals)
    report_data.update(icu_peaks_dates)
    
    # Patients after capacity
    patients_excess = np.sum(np.maximum(all_states_ts['IHT'][:, :-1] - hosp_beds,0),axis = 1)
    report_data['PATHS-HOSP-UNMET'] = 100*np.round(np.sum(patients_excess > 0)/n_replicas,3)
    report_data['MEAN-HOSP-UNSERVED'] = np.round(patients_excess.mean())
    report_data['SD-HOSP-UNSERVED'] = np.round(patients_excess.std())
    for q in [5, 50, 95, 99, 100]:
        report_data[f'HOSP-UNSERVED-P{q}'] = np.round(np.percentile(patients_excess, q))
  
    for cap in [350, 300, 250, 200, 150]:
        icu_patients_excess = np.sum(np.maximum(all_states_ts['ICU'][:, :-1] - cap,0),axis = 1)
        report_data[f'{cap}-PATHS-ICU-UNMET'] = 100*np.round(np.sum(icu_patients_excess > 0)/n_replicas,3)
        report_data[f'{cap}-MEAN-ICU-UNSERVED'] = np.round(icu_patients_excess.mean())
        report_data[f'{cap}-SD-ICU-UNSERVED'] = np.round(icu_patients_excess.std())
     
        for q in [5, 50, 95, 99, 100]:
            report_data[f'{cap}-ICU-UNSERVED-P{q}'] = np.round(np.percentile(icu_patients_excess, q))
    

    # Deaths data
    avg_deaths = np.round(np.mean(all_states_ts['D'][:, -1] - all_states_ts['D'][:, 0]), 0)
    P50_deaths = np.round(np.percentile(all_states_ts['D'][:, -1] - all_states_ts['D'][:, 0], 50))
    P5_deaths = np.round(np.percentile(all_states_ts['D'][:, -1] - all_states_ts['D'][:, 0], 5))
    P95_deaths = np.round(np.percentile(all_states_ts['D'][:, -1] - all_states_ts['D'][:, 0], 95))

    deaths_report = {
        'MEAN-DEATHS': int(avg_deaths),
        'P5-DEATHS': int(P5_deaths),
        'P50-DEATHS': int(P50_deaths),
        'P95-DEATHS': int(P95_deaths)
    }
    print('Deaths End Horizon')
    print(f'Point forecast {all_states_ts["D"][0][-1]}')
    print('Fraction by Age and Risk Group (1-5, L-H)')
 
    report_data.update(deaths_report)

    R_mean = np.mean(all_states_ts['R'][:, -1] - all_states_ts['R'][:, 0] / population)
    print(f'R End Horizon {R_mean}')
    print('Lockdown Threshold:')
    print('policy')
    report_data['policy'] = policy
    
    # Plot school closure and cocooning
    z_ts = mean_st['z'][central_path][:]
    sc_co = [interventions[k].school_closure for k in z_ts]
    unique_policies = set(sc_co)
    sd_levels = [interventions[k].social_distance for k in z_ts]
    unique_sd_policies = set(sd_levels)
    
    intervals = {u: [False for t in range(len(z_ts) + 1)] for u in unique_policies}
    intervals_sd = {u: [False for t in range(len(z_ts) + 1)] for u in unique_sd_policies}
    for t in range(len(z_ts)):
        sc_co_t = interventions[z_ts[t]].school_closure
        for u in unique_policies:
            if u == sc_co_t:
                intervals[u][t] = True
                intervals[u][t + 1] = True
        for u_sd in unique_sd_policies:
            if u_sd == interventions[z_ts[t]].social_distance:
                intervals_sd[u_sd][t] = True
                intervals_sd[u_sd][t + 1] = True
    
    report_data['INITIAL-LOCKDOWN-KAPPA'] = 'XX\\%'
    report_data['LOCKDOWN-KAPPA'] = 'XX\\%'
    report_data['RELAXATION-KAPPA'] = 'XX\\%'
    
    # Plot social distance
    social_distance = [interventions[k].social_distance for k in z_ts]
    #policy_ax.plot(social_distance, c='k', alpha=0.6 * hide)  # marker='_', linestyle='None',
    hsd = np.sum(np.array(social_distance[:]) == policy.tiers[-1]['transmission_reduction'])
    print(f'HIGH SOCIAL DISTANCE')
    print(f'Point Forecast: {hsd}')
    tier_hist_list = []
    tier_hist_mean = []
    colors = []
    tiers = np.arange(len(policy.tiers))
    
    for i, tier in enumerate(policy.tiers):
        tier_hist_list.append(np.array([
            np.sum(
                np.array([interventions[k].social_distance for k in z_ts]) == tier['transmission_reduction'])
            for z_ts in all_states_ts['z']
        ]))
        tier_hist_mean.append(np.mean(tier_hist_list[i])/ (T_end - T_start))
        colors.append(tier['color'])
    report_data['BAR_PLOT'] = plot_tier_bar(instance_name, tiers, tier_hist_mean, colors)    

    lockdown_report = []
    for i, tier in enumerate(policy.tiers):
        if i != 0:
            #breakpoint()
            lockdown_report.append({
                f'MEAN-{tier["color"].upper()}': f'{np.mean(tier_hist_list[i]):.2f}',
                f'P50-{tier["color"].upper()}': f'{np.percentile(tier_hist_list[i],q=50)}',
                f'P5-{tier["color"].upper()}': f'{np.percentile(tier_hist_list[i],q=5)}',
                f'P95-{tier["color"].upper()}': f'{np.percentile(tier_hist_list[i],q=95)}'
            })
            report_data.update(lockdown_report[i-1])
            report_data[f'PATHS-IN-{tier["color"].upper()}'] = 100*round(sum(tier_hist_list[i] > 0)/len(tier_hist_list[i]), 4)
    
    count_lockdowns = defaultdict(int)
    for z_ts in all_states_ts['z']:
        n_lockdowns = 0
        for ix_k in range(1, len(z_ts)):
            if interventions[z_ts[ix_k]].social_distance - interventions[z_ts[ix_k - 1]].social_distance > 0:
                n_lockdowns += 1
        count_lockdowns[n_lockdowns] += 1

    for nlock in np.sort(list(count_lockdowns.keys())):
        print(f'Prob of having exactly {nlock} stricter tier change: {count_lockdowns[nlock]/len(all_states_ts["z"]):4f}')
    unique_social_distance = np.unique(social_distance)

    generate_report_tier(report_data,template_file = template_file, to_email = to_email)
