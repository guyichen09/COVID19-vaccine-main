'''
Module for plotting function
'''
import os
import sys
import numpy as np
import pandas as pd
import time
import argparse
import calendar as py_cal
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib import rc
import matplotlib.patches as patches
import matplotlib.colors as pltcolors
from collections import defaultdict
from utils import round_closest, roundup
from VaccineAllocation import plots_path, instances_path
import copy
import csv

plt.rcParams['hatch.linewidth'] = 3.0

colors = {'S': 'b', 'S0': 'b', 'S1': 'b', 'S2': 'b',  'S3': 'b', 'E': 'y', 'IA': 'c', 'IY': 'm', 'IH': 'k','IHT_moving': 'k', 'R': 'g', 'D': 'k', 'ToIHT': 'teal', 'ToIHT_moving': 'teal', 'ToIHT_total': 'teal','ToIA': 'teal', 'ToIY_moving': 'teal','ToIHT_unvac': 'teal', 'ToIHT_vac': 'teal', 'ICU': 'k', 'ICU_ratio': 'k','ToICU': 'teal', 'IHT': 'k', 'ITot': 'k'}
light_colors = {'IH':'silver', 'IHT_moving':'silver', 'ToIHT':'paleturquoise',  'ToIHT_moving':'paleturquoise', 'ToIHT_total':'paleturquoise', 'D': 'teal', 'ToIHT_unvac':'paleturquoise', 'ToIHT_unvac':'paleturquoise','ToIA':'paleturquoise', 'ToIY_moving': 'paleturquoise', 'ICU':'silver', 'ICU_ratio':'silver','ToICU': 'paleturquoise', 'IHT': 'silver', 'ITot': 'silver', 'S': 'blue', 'S0': 'blue', 'S1': 'blue', 'S2': 'blue', 'S3': 'purple'}
l_styles = {'sim': '-', 'opt': '--'}
compartment_names = {
    'ITot': 'Total Infectious',
    'IY': 'Symptomatic',
    'IH': 'General Beds',
    'IHT_moving': 'Percent of staffed inpatient beds \n(7-day average)',
    'ToIHT_moving': 'COVID-19 Hospital Admissions\n(Seven-day Average)',
    'ToIHT_total': 'COVID-19 Hospital Admissions per 100k \n(Seven-day Total)',
    'D': 'Deaths',
    'R': 'Recovered',
    'S': 'Susceptible',
    'S0': 'S Group 0',
    'S1': 'S Group 1',
    'S2': 'S Group 2',
    'S3': "Susceptible with Waned Immunity",
    'ICU': 'COVID-19 ICU Patients',
    'IHT': 'COVID-19 Hospitalizations',
    'ToICU': 'Daily COVID-19 ICU Admissions',
    'ToIA': 'Daily COVID-19 IA Admissions',
    'ToIY_moving': 'COVID-19 New Symptomatic Cases per 100k \n(Seven-day Sum)',
    'ToIHT_unvac': 'COVID-19 Hospitalizations (Unvax)',
    'ToIHT_vac': 'COVID-19 Hospitalizations (Vax)',
    'ICU_ratio': 'ICU to hospitalization census ratio'
}

def colorDecide(u,tier_by_tr):
    preCoded_color = ["green","blue","yellow","orange","red"]
    colorDict = {}
    for tKey in tier_by_tr.keys():
        colorDict[tier_by_tr[tKey]["color"]] = tKey
    belowTier = -1
    aboveTier = 2
    for item in preCoded_color:
        if (u > colorDict[item])and(colorDict[item] >= belowTier):
            belowTier = colorDict[item]
        if (u < colorDict[item])and(colorDict[item] <= aboveTier):
            aboveTier = colorDict[item]
    aboveColor = pltcolors.to_rgb(tier_by_tr[aboveTier]["color"])
    belowColor = pltcolors.to_rgb(tier_by_tr[belowTier]["color"])
    ratio = (u - belowTier)/(aboveTier - belowTier)
    setcolor = ratio*np.array(aboveColor) + (1-ratio)*np.array(belowColor)
    return setcolor,tier_by_tr[aboveTier]["color"]+\
                            "_"+tier_by_tr[belowTier]["color"]+\
                            "_"+str(ratio)    
                            
def change_avg(all_st, min_st ,max_st, mean_st, nday_avg):
    # obtain the n-day average of the statistics
    all_st_copy = copy.deepcopy(all_st)
    min_st_copy = copy.deepcopy(min_st)
    max_st_copy = copy.deepcopy(max_st)
    mean_st_copy = copy.deepcopy(mean_st)
    
    # change all statistics to n-day average
    for v in all_st_copy.keys():
        if v not in ['z', 'tier_history'] and v != "S3":
            for i in range(len(all_st_copy[v])):
                for t in range(len(all_st_copy[v][i])):
                    all_st_copy[v][i][t] = np.mean(all_st[v][i][np.maximum(t-nday_avg,0):t+1])
            for t in range(len(min_st_copy[v])):
                min_st_copy[v][t] = np.mean(min_st[v][np.maximum(t-nday_avg,0):t+1])
            for t in range(len(max_st_copy[v])):
                max_st_copy[v][t] = np.mean(max_st[v][np.maximum(t-nday_avg,0):t+1])
            for t in range(len(mean_st_copy[v])):
                mean_st_copy[v][t] = np.mean(mean_st[v][np.maximum(t-nday_avg,0):t+1])
          
    return all_st_copy,min_st_copy,max_st_copy,mean_st_copy

def convert_CDC_threshold(policy, N):
    '''
    CDC hospital admission thresholds are for 7-day total, 
    convert to 7-day average for plotting.        
    '''
    nonsurge_hosp_adm_avg = policy.nonsurge_thresholds['hosp_adm'].copy()
    surge_hosp_adm_avg = policy.surge_thresholds['hosp_adm'].copy()
    
    nonsurge_staffed_bed_avg = policy.nonsurge_thresholds['staffed_bed'].copy()
    surge_staffed_bed_avg = policy.surge_thresholds['staffed_bed'].copy()
    
    nonsurge_hosp_adm_avg_ub = policy.nonsurge_thresholds_ub['hosp_adm'].copy()
    surge_hosp_adm_avg_ub = policy.surge_thresholds_ub['hosp_adm'].copy()
    
    nonsurge_staffed_bed_avg_ub = policy.nonsurge_thresholds_ub['staffed_bed'].copy()
    surge_staffed_bed_avg_ub = policy.surge_thresholds_ub['staffed_bed'].copy()
    for tier in range(len(policy.tiers)):
        nonsurge_hosp_adm_avg[tier] = max(nonsurge_hosp_adm_avg[tier]*N/700000, -1)
        surge_hosp_adm_avg[tier] = max(surge_hosp_adm_avg[tier]*N/700000, -1)
        nonsurge_staffed_bed_avg[tier] =  max(nonsurge_staffed_bed_avg[tier]*N/700000, -1)
        surge_staffed_bed_avg[tier] =  max(surge_staffed_bed_avg[tier]*N/700000, -1)
        
        if tier != len(policy.tiers)-1:
            nonsurge_hosp_adm_avg_ub[tier] = max(nonsurge_hosp_adm_avg_ub[tier]*N/700000, -1)
            surge_hosp_adm_avg_ub[tier] = max(surge_hosp_adm_avg_ub[tier]*N/700000, -1)
            nonsurge_staffed_bed_avg_ub[tier] =  max(nonsurge_staffed_bed_avg_ub[tier]*N/700000, -1)
            surge_staffed_bed_avg_ub[tier] =  max(surge_staffed_bed_avg_ub[tier]*N/700000, -1)
    
    nonsurge_thresholds_avg = {"hosp_adm":nonsurge_hosp_adm_avg, "staffed_bed":nonsurge_staffed_bed_avg}
    surge_thresholds_avg = {"hosp_adm":surge_hosp_adm_avg, "staffed_bed":surge_staffed_bed_avg}
    
    nonsurge_thresholds_avg_ub = {"hosp_adm":nonsurge_hosp_adm_avg_ub, "staffed_bed":nonsurge_staffed_bed_avg_ub}
    surge_thresholds_avg_ub = {"hosp_adm":surge_hosp_adm_avg_ub, "staffed_bed":surge_staffed_bed_avg_ub}
    lockdown_threshold_avg = {0: nonsurge_thresholds_avg, 1: surge_thresholds_avg} 
    lockdown_threshold_avg_ub = {0: nonsurge_thresholds_avg_ub, 1: surge_thresholds_avg_ub} 

    return lockdown_threshold_avg, lockdown_threshold_avg_ub 
    
def plot_multi_tier_sims(instance_name,
                         instance,
                         policy,
                         profiles,
                         profile_labels,
                         plot_left_axis=['IH'],
                         plot_right_axis=[],
                         scale_plot=False,
                         align_axes=True,
                         show=True,
                         plot_triggers=False,
                         plot_trigger_annotations=False,
                         plot_legend=False,
                         y_lim=1000,
                         n_replicas=300,
                         config=None,
                         capacity=None,
                         real_data=None,
                         bed_scale=1,
                         vertical_fill=True,
                         dali_plot=False,
                         **kwargs):
    '''
    Plots a list of profiles in the same figure. Each profile corresponds
    to a stochastic replica for the given instance.

    Args:
        profiles (list of dict): a list of dictionaries that contain epi vars profiles
        profile_labels (list of str): name of each profile
    '''
    plt.rcParams["font.size"] = "18"
    T = instance.cal.calendar_ix[kwargs['plot_end']] 
    t_start = kwargs['t_start']
  
    if "add_tiers" in kwargs.keys():
        add_tiers = kwargs["add_tiers"]
    cal = instance.cal
    interventions = kwargs['interventions']
  
    lb_band = 5
    ub_band = 95
    
    text_size = 28
    fig, (ax1, actions_ax) = plt.subplots(2, 1, figsize=(17, 9), gridspec_kw={'height_ratios': [10, 1.1]})
    # Main axis
    ax2 = None
    policy_ax = ax1.twinx()
    if len(plot_right_axis) > 0:
        # Create second axis
        ax2 = ax1.twinx()
        # Fix policy axis
        policy_ax.spines["right"].set_position(("axes", 1.1))
        make_patch_spines_invisible(policy_ax)
        policy_ax.spines["right"].set_visible(True)
        
    plotted_lines = []
  
    # Add IHT field
    for p in profiles:
        p['IHT'] = p['IH'] + p['ICU']

    # Transform data of interest
    states_to_plot = plot_left_axis + plot_right_axis
    if 'ToIHT_moving' in states_to_plot or 'ToIY_moving' in states_to_plot or 'IHT_moving' in states_to_plot  or 'ToIHT_total' in states_to_plot  or 'ICU_ratio' in states_to_plot:
        states_ts = {v: np.vstack(list(p[v][:T] for p in profiles)) for v in states_to_plot}
    else:
        states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot}

    states_ts['z'] = np.vstack(list(p['z'][:T] for p in profiles))
    states_ts['tier_history'] = np.vstack(list(p['tier_history'][:T] for p in profiles))
    
    
    central_path = kwargs['central_id_path']
    mean_st = {v: states_ts[v][central_path] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}

    all_st = {v: states_ts[v][:] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    min_st = {
        v: np.percentile(states_ts[v], q=lb_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    max_st = {
        v: np.percentile(states_ts[v], q=ub_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    # if nday_avg is not None:
    #     all_st, min_st ,max_st, mean_st = change_avg(all_st, min_st ,max_st, mean_st, nday_avg)

    # Plot school closure and cocooning
    tiers = policy.tiers
    z_ts = profiles[central_path]['z'][:T]
    if profiles[central_path]['surge_history'] is not None:
        surge_hist = profiles[central_path]['surge_history'][:T]
        surge_states = [0, 1]
        surge_colors = {0: 'moccasin', 1: 'pink'}
        intervals_surge = {u: [False for t in range(len(z_ts) + 1)] for u in surge_states}

    sc_co = [interventions[k].school_closure for k in z_ts]
    unique_policies = set(sc_co)
    sd_lvl = [interventions[k].social_distance for k in z_ts]
    sd_levels = [tier['transmission_reduction'] for tier in tiers] + [0, 0.95] + sd_lvl
    unique_sd_policies = list(set(sd_levels))
    unique_sd_policies.sort()
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
        if profiles[central_path]['surge_history'] is not None:
            for u in surge_states:
                if surge_hist[t] == u:
                    intervals_surge[u][t] = True
                    intervals_surge[u][t + 1] = True
    # Policy
    if profiles[central_path]['surge_history'] is not None:
        lockdown_threshold  = {0: policy.nonsurge_thresholds, 1: policy.surge_thresholds}
        lockdown_threshold_ub = {0: policy.nonsurge_thresholds_ub, 1: policy.surge_thresholds_ub}
        lockdown_threshold_avg, lockdown_threshold_avg_ub = convert_CDC_threshold(policy, instance.N.sum(axis=(0,1)))
    else:
        lockdown_threshold = policy.lockdown_thresholds[0]
    hide = 1
    l_style = l_styles['sim']
    for v in plot_left_axis:
        label_v = compartment_names[v]           
        v_a = ax1.plot(mean_st[v].T * bed_scale, c=colors[v], linestyle=l_style, linewidth=2, label=label_v, alpha=1 * hide, zorder = 50)
        plotted_lines.append(v_a[0])
        v_aa = ax1.plot(all_st[v].T * bed_scale, c=light_colors[v], linestyle=l_style, linewidth=1, label=label_v, alpha=0.8 * hide)
        plotted_lines.append(v_aa[0])

        if real_data is not None:
            real_h_plot = ax1.scatter(range(len(real_data)), real_data, color='maroon', label='Actual hospitalizations',zorder=100,s=15)

        if capacity is not None:
            ax1.hlines(capacity, 0, T, color='k', linestyle='-', linewidth=3)

    interval_color = {0: 'orange', 1: 'purple', 0.5: 'green'}
    interval_labels = {0: 'Schools Open', 1: 'Schools Closed', 0.5: 'Schools P. Open'}
    for u in unique_policies:
        u_color = interval_color[u]
        u_label = interval_labels[u]
        
        actions_ax.fill_between(
            range(len(z_ts) + 1),
            0,
            1,
            where=intervals[u],
            color='white',  #u_color,
            alpha=0,  #interval_alpha[u],
            label=u_label,
            linewidth=0,
            hatch = '/',
            step='pre')

    sd_labels = {
        0: '',
        0.95: 'Initial lock-down',
    }
    sd_labels.update({tier['transmission_reduction']: tier['name'] for tier in tiers})
    tier_by_tr = {tier['transmission_reduction']: tier for tier in tiers}
    tier_by_tr[0.746873309820472] = {
        "name": 'Ini Lockdown',
        "transmission_reduction": 0.95,
        "cocooning": 0.95,
        "school_closure": 1,
        "min_enforcing_time": 0,
        "daily_cost": 0,
        "color": 'darkgrey'
    }
    if "add_tiers" in kwargs.keys():
        for add_t in add_tiers.keys():
            tier_by_tr[add_t] = {"color": add_tiers[add_t],
                                 "name": "added stage"}
    

    if vertical_fill:
        if v == 'ToIY_moving':
            ax1.hlines(capacity, 0, T, linestyle='-')
            for u in surge_states:
                fill_1 = intervals_surge[u].copy()
                fill_2 = intervals_surge[u].copy()
                u_alpha1 = 0.6
                u_alpha2 = 0.6
                u_color = surge_colors[u] 
                for i in range(len(intervals_surge[u])):
                    if 'history_white' in kwargs.keys() and kwargs['history_white']:
                        if i <= t_start:
                            fill_2[i] = False
                            fill_1[i] = False
                    policy_ax.fill_between(range(len(z_ts) + 1),
                                      0,
                                      1,
                                      where=fill_1,
                                      color=u_color,
                                      linewidth=0.0,
                                      step='pre')
                policy_ax.fill_between(range(t_start+1),
                                      0,
                                      1,
                                      color='white',
                                      linewidth=0.0,
                                      step='pre')
     
        else:
            for u in unique_sd_policies:
                try:
                    if u in tier_by_tr.keys():
                        u_color = tier_by_tr[u]['color']
                        u_label = f'{tier_by_tr[u]["name"]}' if u > 0 else ""
                    else:
                        u_color,u_label = colorDecide(u,tier_by_tr)
                    u_alpha1 = 0.6
                    u_alpha2 = 0.6
                    fill_1 = intervals_sd[u].copy()
                    fill_2 = intervals_sd[u].copy()
                    for i in range(len(intervals_sd[u])):
                        if 'history_white' in kwargs.keys() and kwargs['history_white']:
                            if i <= t_start:
                                fill_2[i] = False
                            fill_1[i] = False
                        else:
                            if i <= t_start:
                                fill_2[i] = False
                            else:
                                fill_1[i] = False
                        
                        policy_ax.fill_between(range(len(z_ts) + 1),
                                       0,
                                       1,
                                       where=fill_1,
                                       color=u_color,
                                       alpha=u_alpha1,
                                       label=u_label,
                                       linewidth=0.0,
                                       step='pre')
                        policy_ax.fill_between(range(len(z_ts) + 1),
                                       0,
                                       1,
                                       where=fill_2,
                                       color=u_color,
                                       alpha=u_alpha2,
                                       label=u_label,
                                       linewidth=0.0,
                                       step='pre')
                    policy_ax.fill_between(range(t_start+1),
                                      0,
                                      1,
                                      color='white',
                                      linewidth=0.0,
                                      step='pre')
                except Exception:
                    print(f'WARNING: TR value {u} was not plotted')
    elif dali_plot:
        # plot the stacked part of the stage proportion
        ax3 = ax1.twinx()
        ax3.set_ylim(0, y_lim)
        data = states_ts['tier_history'].T
    
        tierColor = {}
        for tierInd in range(len(policy.tiers)):
            tierColor[tierInd] = (np.sum(data[(t_start+1):T,:] == tierInd, axis = 1)/len(data[0]))* y_lim   

        r = range((t_start+1), T)
        bottomTier = 0
        #breakpoint()
        for tierInd in range(len(policy.tiers)):
            #breakpoint()
            ax3.bar(r, tierColor[tierInd], color = policy.tiers[tierInd]['color'], bottom = bottomTier, label = 'tier{}'.format(tierInd), width = 1, alpha = 0.6, linewidth = 0)
           # breakpoint()
            bottomTier += np.array(tierColor[tierInd])
           # breakpoint()
        ax3.set_yticks([])
    else:
        # fill the horizontal policy color
        if profiles[central_path]['surge_history'] is not None:
            for u in surge_states:
                fill = intervals_surge[u].copy()
                u_alpha = 0.6
                for i in range(len(intervals_surge[u])):
                    if 'history_white' in kwargs.keys() and kwargs['history_white']:
                        if i <= t_start:
                            fill[i] = False
                    for tier_ix, tier in enumerate(policy.tiers):
                        u_color = tier['color']
                        if v == 'ToIHT_total':
                            u_lb = lockdown_threshold[u]["hosp_adm"][tier_ix]
                            u_ub = lockdown_threshold_ub[u]["hosp_adm"][tier_ix]
                        elif v == 'IHT_moving':
                            u_lb = lockdown_threshold[u]["staffed_bed"][tier_ix]
                            u_ub = lockdown_threshold_ub[u]["staffed_bed"][tier_ix]
                        elif v == 'ToIHT_moving':
                            u_lb = lockdown_threshold_avg[u]["hosp_adm"][tier_ix]
                            u_ub = lockdown_threshold_avg_ub[u]["hosp_adm"][tier_ix]
                        if u_ub == np.inf:
                            u_ub = y_lim
                        if u_lb >= -1 and u_ub >= 0:
                            policy_ax.fill_between(range(len(z_ts) + 1),
                                                   u_lb/y_lim,
                                                   u_ub/y_lim,
                                                   color=u_color,
                                                   alpha=u_alpha,
                                                   where=fill,
                                                   linewidth=0.0,
                                                   step='pre')
        else:
            for ti in range(len(tiers)):
                u = tiers[ti]['transmission_reduction']
                if u in tier_by_tr.keys():
                    u_color = tier_by_tr[u]['color']
                    u_label = f'{tier_by_tr[u]["name"]}' if u > 0 else ""
                else:
                    u_color,u_label = colorDecide(u,tier_by_tr)
                u_alpha = 0.6 
                u_lb = policy.lockdown_thresholds[ti][0]
                u_ub = policy.lockdown_thresholds_ub[ti][0]
                if u_ub == np.inf:
                    u_ub = y_lim
                if u_lb >= -1 and u_ub >= 0:
                    policy_ax.fill_between(range(len(z_ts) + 1),
                                       u_lb/y_lim,
                                       u_ub/y_lim,
                                       color=u_color,
                                       alpha=u_alpha,
                                       label=u_label,
                                       linewidth=0.0,
                                       step='pre')
        policy_ax.fill_between(range(t_start+1),
                                   0,
                                   1,
                                   color='white',
                                   linewidth=0.0,
                                   step='pre')

    # START PLOT STYLING
    # Axis limits
    ax1.set_ylim(0, y_lim)
    if ax2 is not None:
        ax2.set_ylim(0, 0.5)
    policy_ax.set_ylim(0, 1)
    
    # plot a vertical line for the t_start
    plt.vlines(t_start, 0, y_lim, colors='k',linewidth = 3)
    
    # Axis format and names
    ax1.set_ylabel(" / ".join((compartment_names[v] for v in plot_left_axis)), fontsize=text_size)
    if ax2 is not None:
        ax2.set_ylabel(" / ".join((compartment_names[v] for v in plot_right_axis)), fontsize=text_size)
    
    # Axis ticks
    if kwargs['x_axis_tick'] == 1:
        ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1)])
        ax1.xaxis.set_ticklabels(
            [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1)],
            rotation=0,
            fontsize=22)
    elif kwargs['x_axis_tick'] == 2:
        ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and d.month % 2 == 1)])
        ax1.xaxis.set_ticklabels(
            [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and d.month % 2 == 1)],
            rotation=0,
            fontsize=22)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('left')
    ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
    ax1.tick_params(axis='x', length=5, width=2)
    
    # Policy axis span 0 - 1
    if v == 'IHT_moving':
        ax1.yaxis.set_ticks(np.arange(0, 1.001, 0.2))
        ax1.yaxis.set_ticklabels(
        [f' {np.round(t*100)}%' for t in np.arange(0, 1.001, 0.2)],
        rotation=0,
        fontsize=22)
    policy_ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelbottom=False,
        labelright=False)  # labels along the bottom edge are off
    
    actions_ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off
    actions_ax.spines['top'].set_visible(False)
    actions_ax.spines['bottom'].set_visible(False)
    actions_ax.spines['left'].set_visible(False)
    actions_ax.spines['right'].set_visible(False)

    if 100 <= T:
        actions_ax.annotate('2020',
                            xy=(120, 0),
                            xycoords='data',
                            color='k',
                            annotation_clip=True,
                            fontsize=text_size - 2)
    if 480 <= T:
        actions_ax.annotate('2021',
                            xy=(480, 0),
                            xycoords='data',
                            color='k',
                            annotation_clip=True,
                            fontsize=text_size - 2)
    
    if 700 <= T:
        actions_ax.annotate('2022',
                            xy=(700, 0),
                            xycoords='data',
                            color='k',
                            annotation_clip=True,
                            fontsize=text_size - 2) 
        
    # Order of layers
    ax1.set_zorder(policy_ax.get_zorder() + 10)  # put ax in front of policy_ax
    ax1.patch.set_visible(False)  # hide the 'canvas'
    if ax2 is not None:
        ax2.set_zorder(policy_ax.get_zorder() + 5)  # put ax in front of policy_ax
        ax2.patch.set_visible(False)  # hide the 'canvas'
    
    # Plot margins
    ax1.margins(0)
    actions_ax.margins(0)
    if ax2 is not None:
        ax2.margins(0)
    policy_ax.margins(0.)

    if plot_legend:
        handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], [])
        handles_action_ax, labels_action_ax = actions_ax.get_legend_handles_labels()
        handles_policy_ax, labels_policy_ax = policy_ax.get_legend_handles_labels()
        plotted_labels = [pl.get_label() for pl in plotted_lines]
        if 'ToIHT' in plot_left_axis or True:
            fig_legend = ax1.legend(
                plotted_lines + handles_policy_ax + handles_action_ax,
                plotted_labels + labels_policy_ax + labels_action_ax,
                loc='upper right',
                fontsize=text_size + 2,
                #bbox_to_anchor=(0.90, 0.9),
                prop={'size': text_size},
                framealpha=1)
        elif 'IH' in plot_left_axis:
            fig_legend = ax1.legend(
                handles_ax1,
                labels_ax1,
                loc='upper right',
                fontsize=text_size + 2,
                #bbox_to_anchor=(0.90, 0.9),
                prop={'size': text_size},
                framealpha=1)
        fig_legend.set_zorder(4)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plots_left_right = plot_left_axis + plot_right_axis
    plot_filename = plots_path / f'scratch_{instance_name}_{"".join(plots_left_right)}.pdf'
    plt.savefig(plot_filename)
    if show:
        plt.show()
    plt.close()
    return plot_filename

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)