import argparse
from pathlib import Path

plots_path = Path(__file__).parent / 'plots'

profile_log = {}

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
    parser.add_argument("-agg", action="store_true", help='Whether to use agg for matplotlib')
    parser.add_argument("-gt", default=None, help='Given tier levels, in the format of list or list of lists, without space in between')
    parser.add_argument("-field", default='ToIHT', type=str, help='The field selected for criterion')
    parser.add_argument("-fo", default='[]', type=str, help='Forced out tiers before first red')
    parser.add_argument("-rl", default=1000, type=int, help='How long is the maximum length of red')
    parser.add_argument("-pub", default=None, help='Policy upper bound')
    
    args = parser.parse_args()
    
    if args.agg:
        import matplotlib
        matplotlib.use('agg')
        print('Agg in use')
    return args