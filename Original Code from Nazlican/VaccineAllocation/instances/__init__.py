import json
import numpy as np
import pickle
import pandas as pd
import datetime as dt
from pathlib import Path
from epi_params import EpiSetup
from SEIYAHRD import SimCalendar
from scipy.optimize import root_scalar
from vaccine_params import Vaccine

instances_path = Path(__file__).parent

datetime_formater = '%Y-%m-%d %H:%M:%S'
date_formater = '%Y-%m-%d'


class Instance():
    def __init__(self, city, setup_file_name, transmission_file_name, hospitalization_file_name):
        self.city = city
        self.path_to_data = instances_path / f"{city}"
        self.setup_file = self.path_to_data / setup_file_name
        self.transmission_file = self.path_to_data / transmission_file_name
        self.hospitalization_file = self.path_to_data / hospitalization_file_name
        self.delta_prev_file = self.path_to_data / "delta_prevelance.csv"
        self.omicron_prev_file = self.path_to_data / "omicron_prevelance.csv"
        self.variant_prev_file = self.path_to_data / "variant_prevelance.csv"
    
    def load_data(self):
        '''
            Load setup file of the instance.
        '''
        filename = str(self.setup_file)
        with open(filename, 'r') as input_file:
            data = json.load(input_file)
            assert self.city == data['city'], "Data file does not match city."
            
            for (k, v) in data.items():
                setattr(self, k, v)
            
            # Load demographics
            self.N = np.array(data['population'])
            del self.population
            self.I0 = np.array(data['IY_ini'])
            del self.IY_ini
            
            # Load simulation dates
            self.start_date = dt.datetime.strptime(data['start_date'], datetime_formater)
            self.end_date = dt.datetime.strptime(data['end_date'], datetime_formater)
            self.last_date_interventions = dt.datetime.strptime(data['last_date_interventions'], datetime_formater)
            self.school_closure_period = []
            for blSc in range(len(data['school_closure'])):
                self.school_closure_period.append([
                    dt.datetime.strptime(data['school_closure'][blSc][0], datetime_formater),
                    dt.datetime.strptime(data['school_closure'][blSc][1], datetime_formater)
                ])
            del self.school_closure
            
            # Load epi parameters
            self.epi = EpiSetup.load_file(data['epi_params'])
            del self.epi_params
            # check if qInt is in epi, if not, add a placeholder
            try:
                self.epi.qInt['testStart'] = dt.datetime.strptime(self.epi.qInt['testStart'], datetime_formater)
            except:
                setattr(self.epi, "qInt", {'testStart': self.end_date, 
                                           'qRate': {'IY': 0, 'IA': 0, 'PY': 0, 'PA': 0}, 
                                           'randTest': 0})
        
        cal_filename = str(self.path_to_data / 'calendar.csv')
        with open(cal_filename, 'r') as cal_file:
            cal_df = pd.read_csv(cal_file, parse_dates=['Date'], date_parser=pd.to_datetime)
            self.weekday_holidays = list(cal_df['Date'][cal_df['Calendar'] == 3])
            self.weekday_longholidays = list(cal_df['Date'][cal_df['Calendar'] == 4])
        
        filename = str(self.hospitalization_file)
        with open(filename, 'r') as hosp_file:
            df_hosp = pd.read_csv(
                filename,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        # if hospitalization data starts before self.start_date 
        if df_hosp['date'][0] <= self.start_date:
            df_hosp = df_hosp[df_hosp['date'] >= self.start_date]
            df_hosp = df_hosp[df_hosp['date'] <= self.end_date]
            self.real_hosp = list(df_hosp['hospitalized'])
        else:
            df_hosp = df_hosp[df_hosp['date'] <= self.end_date]
            self.real_hosp = [0] * (df_hosp['date'][0] - self.start_date).days + list(df_hosp['hospitalized'])
    
    
        filename = str(self.delta_prev_file)
        with open(filename, 'r') as hosp_file:
            df_delta = pd.read_csv(
                filename,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        self.delta_prev = list(df_delta['delta_prev'])    
        self.delta_start = df_delta['date'][0]
    
    
        filename = str(self.omicron_prev_file)
        with open(filename, 'r') as hosp_file:
            df_omicron = pd.read_csv(
                filename,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        self.omicron_prev = list(df_omicron['prev'])    
        self.omicron_start = df_omicron['date'][0]
        
        filename = str(self.variant_prev_file)
        with open(filename, 'r') as hosp_file:
            df_variant = pd.read_csv(
                filename,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        self.variant_prev = list(df_variant['prev'])    
        self.variant_start = df_variant['date'][0]
        
    def process_data(self):
        '''
            Compute couple parameters (i.e., parameters that depend on the input)
            and build th simulation calendar.
        '''
        hosp_beds = self.hosp_beds
        k = self.staffing_rule_safety
        root_sol = root_scalar(lambda x: x + k * np.sqrt(x) - hosp_beds, x0=hosp_beds, x1=hosp_beds * 0.5)
        assert root_sol.converged, 'Staffing rule failed'
        self.lambda_star = root_sol.root
        
        # Dimension variables
        self.A = len(self.N)
        self.L = len(self.N[0])
        self.T = 1 + (self.end_date - self.start_date).days
        self.otherInfo = {}
        
        cal = SimCalendar(self.start_date, self.T)
        try:
            df_transmission = pd.read_csv(
                str(self.transmission_file),
                parse_dates=['date'],
                date_parser=pd.to_datetime,
                float_precision='round_trip'
            )
            transmission_reduction = [
                (d, tr) for (d, tr) in zip(df_transmission['date'], df_transmission['transmission_reduction'])
            ]
            try:
                cocooning = [
                    (d, co) for (d, co) in zip(df_transmission['date'], df_transmission['cocooning'])
                ]
            except:
                cocooning = [(d, 0.0) for d in df_transmission['date']]
            lockdown_end = df_transmission['date'].iloc[-1]
            cal.load_fixed_transmission_reduction(transmission_reduction, present_date=lockdown_end)
            cal.load_fixed_cocooning(cocooning, present_date=lockdown_end)
            for dfk in df_transmission.keys():
                if dfk != 'date' and dfk != 'transmission_reduction' and dfk != 'cocooning':
                    self.otherInfo[dfk] = {}
                    for (d, dfv) in zip(df_transmission['date'], df_transmission[dfk]):
                        if d in cal.calendar_ix:
                            d_ix = cal.calendar_ix[d]
                            self.otherInfo[dfk][d_ix] = dfv
        except FileNotFoundError:
            # Initialize empty if no file avalabe
            cal.load_fixed_transmission_reduction([])
        
        # School closures and school calendar
        cal.load_school_closure(self.school_closure_period)
        
        # Holidays
        try:
            cal.load_holidays(self.weekday_holidays, self.weekday_longholidays)
        except Exception:
            print('No calendar was provided')
        
        # Save real_hosp in calendar
        cal.real_hosp = self.real_hosp
        
        #Save calendar
        self.cal = cal
    
    @property
    def summary(self):
        return (
            self.epi,
            self.T,
            self.A,
            self.L,
            self.N,
            self.I0,
            self.hosp_beds,
            self.lambda_star,
            self.cal,
        )

class Tier:
    def __init__(self, city, tier_file_name):
        self.path_to_data = instances_path / f"{city}"
        self.tier_file = self.path_to_data / tier_file_name
    
    def load_data(self):
        tier_filename = str(self.tier_file)
        with open(tier_filename, 'r') as tier_input:
            tier_data = json.load(tier_input)
            self.community_transmision = tier_data['community_tranmission']
            assert tier_data['type'] in ['constant','step', 'linear', 'CDC'], "Tier type unknown."
            self.tier_type = tier_data['type']
            if self.tier_type == "CDC":
                self.case_threshold = tier_data["case_threshold"]
            else:
                 self.case_threshold = None     
            self.tier = tier_data['tiers']


def load_instance(city, setup_file_name, transmission_file_name, hospitalization_file_name):
    # TODO: Add the option of providing a different setup file
    instance = Instance(city, setup_file_name, transmission_file_name, hospitalization_file_name)
    instance.load_data()
    instance.process_data()
    
    return instance

def load_tiers(city, tier_file_name='tiers_data.json'):
    # read in the tier file
    tiers = Tier(city, tier_file_name)
    tiers.load_data()
    return tiers

def load_vaccines(city, instance, vaccine_file_name = 'vaccines.json', booster_file_name = 'booster_allocation_fixed.csv', vaccine_allocation_file_name = None):
    '''
        read in the vaccine files
 
        Parameters:
            path_to_data: string.
            vaccine_file_name: string (json file)
                name of vaccine file that store epidemiological characteristics for different vaccine type.
            vaccine_allocation_file_name:string (csv file)
                name of vaccine supply file. If there is a fixed vaccine allocation, input it here.
                First dose vaccine supply and allocation.               
    '''
    path_to_data = instances_path / f"{city}"
    vaccine_file = path_to_data / vaccine_file_name
    vaccine_allocation_file_name = path_to_data / vaccine_allocation_file_name
    
  
    with open(vaccine_file, 'r') as vaccine_input:
        vaccine_data = json.load(vaccine_input)
     
    with open(vaccine_allocation_file_name, 'r') as vaccine_allocation_input:   
        vaccine_allocation_data = pd.read_csv(vaccine_allocation_input, parse_dates=['vaccine_time'], date_parser=pd.to_datetime)
    
    if booster_file_name is not None:
        booster_file_name = path_to_data / booster_file_name
        with open(booster_file_name, 'r') as booster_allocation_input:   
            booster_allocation_data = pd.read_csv(booster_allocation_input, parse_dates=['vaccine_time'], date_parser=pd.to_datetime)
    else:
         booster_allocation_data = None
                  
    vaccines = Vaccine(vaccine_data, vaccine_allocation_data, booster_allocation_data, instance)
    #breakpoint()
    return vaccines
   

def load_seeds(city, seeds_file_name='seeds.p'):
    # read in the seeds file
    seedsinput = instances_path / f"{city}"
    try:
        with open(seedsinput / seeds_file_name, 'rb') as infile:
            seeds_data = pickle.load(infile)
        return seeds_data[0], seeds_data[1]    
        #return seeds_data['training'], seeds_data['testing']
    except:
        return [],[]