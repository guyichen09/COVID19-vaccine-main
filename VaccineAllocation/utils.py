import os
import sys
import numpy as np
import time
import argparse
import shlex
import calendar as py_cal
from pathlib import Path
from collections import defaultdict
import datetime

plots_path = Path(__file__).parent / 'plots'

profile_log = {}


def roundup(x, l):
    return int(np.ceil(x / l)) * l


def round_closest(x, l):
    return int(np.around(x / l)) * l


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        name = kw.get('log_name', method.__name__)
        call_time = (te - ts)
        if name not in profile_log:
            profile_log[name] = {'calls': 1, 'max_time': call_time, 'avg_time': call_time}
        else:
            avg = profile_log[name]['avg_time']
            calls = profile_log[name]['calls']
            profile_log[name]['avg_time'] = (avg * calls + call_time) / (calls + 1)
            profile_log[name]['calls'] = calls + 1
            profile_log[name]['max_time'] = np.maximum(profile_log[name]['max_time'], call_time)
        
        # else:
        #     print(f'{method.__name__}: {(te - ts): 10.5f} s')
        return result
    
    return timed


def print_profiling_log(logger):
    for (k, v) in profile_log.items():
        logger.info(f'{str(k):15s}: {str(v)}')


def parse_arguments():
    parser = argparse.ArgumentParser("Parameters for trigger optimization")
    parser.add_argument("city", metavar='c', type=str, help='City to simulate')
    parser.add_argument("-det",
                        default=False,
                        type=bool,
                        help='True: it will run the deterministic path if False it will run stochastic path')
    parser.add_argument("-f",
                        metavar='FILENAME',
                        default='setup_data.json',
                        type=str,
                        help='File name where setup data is (stem only).')
    parser.add_argument("-t",
                        metavar='FILENAME',
                        default='tiers.json',
                        type=str,
                        help='File name where tier data is (stem only).')
    parser.add_argument("-initial",
                        metavar='OBJ',
                        default=0,
                        type=int,
                        help='Which initial solutio is used')
    parser.add_argument("-v",
                        metavar='FILENAME',
                        default='vaccines.json',
                        type=str,
                        help='File name where vaccine data is.')   
    parser.add_argument("-v_allocation", 
                        metavar='FILENAME',
                        default='vaccine_allocation_fixed.csv',
                        type=str,
                        help='File name where vaccine supply and allocation data is.')
    parser.add_argument("-v_boost", 
                        metavar = 'FILENAME',
                        default = 'booster_allocation_fixed.csv',
                        type = str,
                        help = 'File name where booster dose supply and allocation data is.')
    # parser.add_argument("-v_time_increment",
    #                     metavar='Vaccine',
    #                     default=5,
    #                     type=int,
    #                     help='Time increment for greedy vaccine allocation.')
    # parser.add_argument("-percentage",
    #                     metavar='Vaccine',
    #                     default=1,
    #                     type=float,
    #                     help='Percentage for the phase 1b 65+ allocation.')
    parser.add_argument("-tr",
                        metavar='FILENAME',
                        default='transmission.csv',
                        type=str,
                        help='File name where transmission reduction data is (stem only).')
    parser.add_argument("-train_reps",
                        metavar='N1',
                        default=300,
                        type=int,
                        help='Number of replicas when training the policy.')
    parser.add_argument("-test_reps",
                        metavar='N2',
                        default=350,
                        type=int,
                        help='Number of replicas when testing the policy.')
    parser.add_argument("-f_config",
                        metavar='FILENAME',
                        default='trigger_config.json',
                        type=str,
                        help='File name with opt configurations.')
    parser.add_argument("-seed",
                        metavar='FILENAME',
                        default='seeds.p',
                        type=str,
                        help='File name with opt configurations.')
    parser.add_argument("-hos",
                        metavar='FILENAME',
                        default='hospitalizations.csv',
                        type=str,
                        help='File name with real hospitalizations.')
    parser.add_argument("-n_proc", metavar='NP', default=4, type=int, help='Number of processor to use')
    # parser.add_argument("-task", metavar='TASK', default='optimize', type=str, help='Task to be excecuted')
    parser.add_argument("-n_policy",
                        metavar='POLICIES',
                        default=None,
                        type=int,
                        help='Vaccine policy no.')

    parser.add_argument("-agg", action="store_true", help='Whether to use agg for matplotlib')
    parser.add_argument("-plot", action="store_true", help='Whether to use agg for matplotlib')
    parser.add_argument("-machine", default='local', type=str, help='Whether to use agg for matplotlib')
    parser.add_argument("-gt", default=None, help='Given tier levels, in the format of list or list of lists, without space in between')
    parser.add_argument("-gv", default=None, help='Given vaccine allocation, in the format of True or False')
    parser.add_argument("-gd", default=None, help='Given step date, in the format of year=yyyy,month=mm,day=dd, without space in between')
    parser.add_argument("-gs", default=0.1, type=float, help='Given slope for the linear threshold policy') 
    parser.add_argument("-field", default='ToIHT', type=str, help='The field selected for criterion')
    parser.add_argument("-fo", default='[]', type=str, help='Forced out tiers before first red')
    parser.add_argument("-rl", default=1000, type=int, help='How long is the maximum length of red')
    parser.add_argument("-aftert", default='[0,1,2,3,4]', type=str, help='Tiers after first red')
    parser.add_argument("-pub", default=None, help='Policy upper bound')
    parser.add_argument("-df_obj", default='mean_death', type=str, help='The field selected for dfo objective')
    
    args = parser.parse_args()
    
    if args.agg:
        import matplotlib
        matplotlib.use('agg')
        print('Agg in use')
    return args

def parse_arguments_acs():
    parser = argparse.ArgumentParser("Parameters for trigger optimization")
    parser.add_argument("city", metavar='c', type=str, help='City to simulate')
    parser.add_argument("-f",
                        metavar='FILENAME',
                        default='setup_data.json',
                        type=str,
                        help='File name where setup data is (stem only).')
    parser.add_argument("-t",
                        metavar='FILENAME',
                        default='tiers.json',
                        type=str,
                        help='File name where tier data is (stem only).')
    parser.add_argument("-initial",
                        metavar='OBJ',
                        default=0,
                        type=int,
                        help='Which initial solutio is used')
    parser.add_argument("-v",
                        metavar='FILENAME',
                        default='vaccines.json',
                        type=str,
                        help='File name where vaccine data is.')   
    parser.add_argument("-v_allocation", 
                        metavar='FILENAME',
                        default=None,
                        type=str,
                        help='File name where fixed vaccine allocation data is.')
    parser.add_argument("-v_time_increment",
                        metavar='Vaccine',
                        default=5,
                        type=int,
                        help='Time increment for greedy vaccine allocation.')
    parser.add_argument("-percentage",
                        metavar='Vaccine',
                        default=1,
                        type=float,
                        help='Percentage for the phase 1b 65+ allocation.')
    parser.add_argument("-tr",
                        metavar='FILENAME',
                        default='transmission.csv',
                        type=str,
                        help='File name where transmission reduction data is (stem only).')
    parser.add_argument("-train_reps",
                        metavar='N1',
                        default=300,
                        type=int,
                        help='Number of replicas when training the policy.')
    parser.add_argument("-test_reps",
                        metavar='N2',
                        default=350,
                        type=int,
                        help='Number of replicas when testing the policy.')
    parser.add_argument("-f_config",
                        metavar='FILENAME',
                        default='trigger_config.json',
                        type=str,
                        help='File name with opt configurations.')
    parser.add_argument("-seed",
                        metavar='FILENAME',
                        default='seeds_acs.p',
                        type=str,
                        help='File name with opt configurations.')
    parser.add_argument("-hos",
                        metavar='FILENAME',
                        default='hospitalizations.csv',
                        type=str,
                        help='File name with real hospitalizations.')
    parser.add_argument("-n_proc", metavar='NP', default=4, type=int, help='Number of processor to use')
    # parser.add_argument("-task", metavar='TASK', default='optimize', type=str, help='Task to be excecuted')
    # parser.add_argument("-n_tiers",
    #                     metavar='TIERS',
    #                     default=5,
    #                     type=int,
    #                     help='Number of tiers to use, options are 2 and 5.')
    parser.add_argument("-n_policy",
                        metavar='POLICIES',
                        default=None,
                        type=int,
                        help='Vaccine policy no.')
    parser.add_argument("-v_boost", 
                        metavar = 'FILENAME',
                        default = None,
                        type = str,
                        help = 'File name where booster dose supply and allocation data is.')
  
    parser.add_argument("-agg", action="store_true", help='Whether to use agg for matplotlib')
    parser.add_argument("-plot", action="store_true", help='Whether to use agg for matplotlib')
    parser.add_argument("-machine", default='local', type=str, help='Whether to use agg for matplotlib')
    parser.add_argument("-gt", default=None, help='Given tier levels, in the format of list or list of lists, without space in between')
    parser.add_argument("-gv", default=None, help='Given vaccine allocation, in the format of True or False')
    parser.add_argument("-gd", default=None, help='Given step date, in the format of year=yyyy,month=mm,day=dd, without space in between')
    parser.add_argument("-gs", default=0.1, type=float, help='Given slope for the linear threshold policy') 
    parser.add_argument("-field", default='ToIHT', type=str, help='The field selected for criterion')
    parser.add_argument("-fo", default='[]', type=str, help='Forced out tiers before first red')
    parser.add_argument("-rl", default=1000, type=int, help='How long is the maximum length of red')
    parser.add_argument("-aftert", default='[0,1,2,3,4]', type=str, help='Tiers after first red')
    parser.add_argument("-pub", default=None, help='Policy upper bound')
    parser.add_argument("-acs_bounds", default=None, help='ACS trigger searching range')
    parser.add_argument("-acs_times", default=None, help='ACS effective times searching range')
    parser.add_argument("-acs_leadT", default=21, type=int, help='ACS construction lead time')
    parser.add_argument("-acs_Q", default=500, type=int, help='ACS capacity')
    parser.add_argument("-acs_type", default='IHT', type=str, help='ACS type: ICU or IHT')
    parser.add_argument("-df_obj", default='mean_death', type=str, help='The field selected for dfo objective')
    
    args = parser.parse_args()
    
    if args.agg:
        import matplotlib
        matplotlib.use('agg')
        print('Agg in use')
    return args
