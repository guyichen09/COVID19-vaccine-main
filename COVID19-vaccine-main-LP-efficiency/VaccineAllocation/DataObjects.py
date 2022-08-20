import json
import numpy as np
import pandas as pd
import datetime as dt

from pathlib import Path

from SimObjects import EpiSetup

instances_path = Path(__file__).parent

datetime_formater = '%Y-%m-%d %H:%M:%S'

WEEKDAY = 1
WEEKEND = 2
HOLIDAY = 3
LONG_HOLIDAY = 4

class SimCalendar:
    '''
        A simulation calendar to map time steps to days. This class helps
        to determine whether a time step t is a weekday or a weekend, as well
        as school calendars.

        Attrs:
        start (datetime): start date of the simulation
        calendar (list): list of datetime for every time step
    '''

    def __init__(self, start_date, sim_length):
        '''
            Arg
        '''
        self.start = start_date
        self.calendar = [self.start + dt.timedelta(days=t) for t in range(sim_length)]
        self.calendar_ix = {d: d_ix for (d_ix, d) in enumerate(self.calendar)}
        self._is_weekday = [d.weekday() not in [5, 6] for d in self.calendar]
        self._day_type = [WEEKDAY if iw else WEEKEND for iw in self._is_weekday]
        self.lockdown = None
        self.schools_closed = None
        self.fixed_transmission_reduction = None
        self.fixed_cocooning = None
        self.month_starts = self.get_month_starts()

    def is_weekday(self, t):
        '''
            True if t is a weekday, False otherwise
        '''
        return self._is_weekday[t]

    def get_day_type(self, t):
        '''
            Returns the date type with the convention of the class
        '''
        return self._day_type[t]

    def load_predefined_lockdown(self, lockdown_blocks):
        '''
            Loads fixed decisions on predefined lock-downs and saves
            it on attribute lockdown.
            Args:
                lockdown_blocks (list of tuples): a list with blocks in which predefined lockdown is enacted
                (e.g. [(datetime.date(2020,3,24),datetime.date(2020,8,28))])

        '''
        self.lockdown = []
        for d in self.calendar:
            closedDay = False
            for blNo in range(len(lockdown_blocks)):
                if d >= lockdown_blocks[blNo][0] and d <= lockdown_blocks[blNo][1]:
                    closedDay = True
            self.lockdown.append(closedDay)

    def load_school_closure(self, school_closure_blocks):
        '''
            Load fixed decisions on school closures and saves
            it on attribute schools_closed
            Args:
                school_closure_blocks (list of tuples): a list with blocks in which schools are closed
                (e.g. [(datetime.date(2020,3,24),datetime.date(2020,8,28))])
        '''
        self.schools_closed = []
        for d in self.calendar:
            closedDay = False
            for blNo in range(len(school_closure_blocks)):
                if d >= school_closure_blocks[blNo][0] and d <= school_closure_blocks[blNo][1]:
                    closedDay = True
            self.schools_closed.append(closedDay)

    def load_fixed_transmission_reduction(self, ts_transmission_reduction, present_date=dt.datetime.today()):
        '''
            Load fixed decisions on transmission reduction and saves it on attribute fixed_transmission_reduction.
            If a value is not given, the transmission reduction is None.
            Args:
                ts_transmission_reduction (list of tuple): a list with the time series of
                    transmission reduction (datetime, float).
                present_date (datetime): reference date so that all dates before must have a
                    transmission reduction defined
        '''
        self.fixed_transmission_reduction = [None for d in self.calendar]
        for (d, tr) in ts_transmission_reduction:
            if d in self.calendar_ix:
                d_ix = self.calendar_ix[d]
                self.fixed_transmission_reduction[d_ix] = tr

    def load_fixed_cocooning(self, ts_cocooning, present_date=dt.datetime.today()):
        '''
            Load fixed decisions on transmission reduction and saves it on attribute fixed_transmission_reduction.
            If a value is not given, the transmission reduction is None.
            Args:
                ts_cocooning (list of tuple): a list with the time series of
                    transmission reduction (datetime, float).
                present_date (datetime): reference date so that all dates before must have a
                    transmission reduction defined
        '''
        self.fixed_cocooning = [None for d in self.calendar]
        for (d, tr) in ts_cocooning:
            if d in self.calendar_ix:
                d_ix = self.calendar_ix[d]
                self.fixed_cocooning[d_ix] = tr

    def load_holidays(self, holidays=[], long_holidays=[]):
        '''
            Change the day_type for holidays
        '''
        for hd in holidays:
            dt_hd = dt.datetime(hd.year, hd.month, hd.day)
            if dt_hd in self.calendar:
                self._day_type[self.calendar_ix[dt_hd]] = HOLIDAY

        for hd in long_holidays:
            dt_hd = dt.datetime(hd.year, hd.month, hd.day)
            if dt_hd in self.calendar:
                self._day_type[self.calendar_ix[dt_hd]] = LONG_HOLIDAY

    def get_month_starts(self):
        '''
            Get a list of first days of months
        '''
        month_starts = []

        currentTemp = self.get_next_month(self.start)
        while currentTemp <= self.calendar[-1]:
            month_starts.append(self.calendar_ix[currentTemp])
            currentTemp = self.get_next_month(currentTemp)

        return month_starts

    def __len__(self):
        return len(self.calendar)

    def get_next_month(self, dateG):
        if dateG.month == 12:
            startMonth = 1
            startYear = dateG.year + 1
        else:
            startMonth = dateG.month + 1
            startYear = dateG.year
        return dt.datetime(startYear, startMonth, 1)

class ProblemInstance:
    def __init__(self, city,
                 config_filename,
                 calendar_filename,
                 setup_filename,
                 transmission_filename,
                 hospitalization_filename,
                 delta_prevalence_filename,
                 omicron_prevalence_filename,
                 variant_prevalence_filename):
        self.city = city
        self.path_to_data = instances_path / "instances" / f"{city}"

        with open(str(self.path_to_data / config_filename), 'r') as input_file:
            self.config = json.load(input_file)

        self.load_data(setup_filename,
                       calendar_filename,
                       hospitalization_filename,
                       delta_prevalence_filename,
                       omicron_prevalence_filename,
                       variant_prevalence_filename)
        self.process_data(transmission_filename)

    def load_data(self, setup_filename,
                  calendar_filename,
                  hospitalization_filename,
                  delta_prevalence_filename,
                  omicron_prevalence_filename,
                  variant_prevalence_filename):
        '''
            Load setup file of the instance.
        '''
        filename = str(self.path_to_data / setup_filename)
        with open(filename, 'r') as input_file:
            data = json.load(input_file)
            assert self.city == data['city'], "Data file does not match city."

            for (k, v) in data.items():
                setattr(self, k, v)

            # Load demographics
            self.N = np.array(data['population'])
            self.I0 = np.array(data['IY_ini'])

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

            # Load epi parameters
            self.epi = EpiSetup.load_file(data['epi_params'])
            # check if qInt is in epi, if not, add a placeholder
            try:
                self.epi.qInt['testStart'] = dt.datetime.strptime(self.epi.qInt['testStart'], datetime_formater)
            except:
                setattr(self.epi, "qInt", {'testStart': self.end_date,
                                           'qRate': {'IY': 0, 'IA': 0, 'PY': 0, 'PA': 0},
                                           'randTest': 0})

        cal_df = pd.read_csv(str(self.path_to_data / calendar_filename),
                             parse_dates=['Date'], date_parser=pd.to_datetime)
        self.weekday_holidays = list(cal_df['Date'][cal_df['Calendar'] == 3])
        self.weekday_longholidays = list(cal_df['Date'][cal_df['Calendar'] == 4])

        df_hosp = pd.read_csv(
            str(self.path_to_data / hospitalization_filename),
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

        df_delta = pd.read_csv(
            str(self.path_to_data / delta_prevalence_filename),
            parse_dates=['date'],
            date_parser=pd.to_datetime,
        )
        self.delta_prev = list(df_delta['delta_prev'])
        self.delta_start = df_delta['date'][0]

        df_omicron = pd.read_csv(
            str(self.path_to_data / omicron_prevalence_filename),
            parse_dates=['date'],
            date_parser=pd.to_datetime,
        )
        self.omicron_prev = list(df_omicron['prev'])
        self.omicron_start = df_omicron['date'][0]

        df_variant = pd.read_csv(
            str(self.path_to_data / variant_prevalence_filename),
            parse_dates=['date'],
            date_parser=pd.to_datetime,
        )
        self.variant_prev = list(df_variant['prev'])
        self.variant_start = df_variant['date'][0]

    def process_data(self, transmission_filename):
        '''
            Compute couple parameters (i.e., parameters that depend on the input)
            and build th simulation calendar.
        '''

        # Dimension variables
        self.A = len(self.N)
        self.L = len(self.N[0])
        self.T = 1 + (self.end_date - self.start_date).days
        self.otherInfo = {}

        cal = SimCalendar(self.start_date, self.T)
        try:
            df_transmission = pd.read_csv(
                str(self.path_to_data / transmission_filename),
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
            # Initialize empty if no file available
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

        # Save calendar
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

class TierInformation:
    def __init__(self, city, tier_filename):
        self.path_to_data = instances_path / "instances" / f"{city}"
        with open(str(self.path_to_data / tier_filename), 'r') as tier_input:
            tier_data = json.load(tier_input)
            self.tier = tier_data['tiers']

class VaccineInstance:
    '''
        Vaccine class to define epidemiological characteristics, supply and fixed allocation schedule of vaccine.
        Parameters:
            vaccine_data: (dict) dict of vaccine characteristics.
            vaccine_allocation_data: (dict) contains vaccine schedule, supply and allocation data.
            booster_allocation_data: (dict) contains booster schedule, supply and allocation data.
            instance: data instance

    '''
    def __init__(self, instance, city,
                 vaccine_filename,
                 booster_filename,
                 vaccine_allocation_filename):

        self.path_to_data = instances_path / "instances" / f"{city}"

        with open(str(self.path_to_data / vaccine_filename), 'r') as vaccine_input:
            vaccine_data = json.load(vaccine_input)

        vaccine_allocation_data = pd.read_csv(str(self.path_to_data / vaccine_allocation_filename),
                                              parse_dates=['vaccine_time'],
                                              date_parser=pd.to_datetime)

        if booster_filename is not None:
            booster_allocation_data = pd.read_csv(str(self.path_to_data / booster_filename),
                                                  parse_dates=['vaccine_time'],
                                                  date_parser=pd.to_datetime)
        else:
            booster_allocation_data = None

        self.effect_time = vaccine_data['effect_time']
        self.waning_time = vaccine_data['waning_time']
        self.second_dose_time = vaccine_data['second_dose_time']
        self.beta_reduct = vaccine_data['beta_reduct']
        self.tau_reduct = vaccine_data['tau_reduct']
        self.beta_reduct_delta = vaccine_data['beta_reduct_delta']
        self.tau_reduct_delta = vaccine_data['tau_reduct_delta']
        self.tau_reduct_omicron = vaccine_data['tau_reduct_omicron']
        self.instance = instance

        self.vaccine_allocation = self.define_supply(instance, vaccine_allocation_data, booster_allocation_data)
        self.event_lookup_dict = self.build_event_lookup_dict()

    def build_event_lookup_dict(self):
        '''
        Must be called after self.vaccine_allocation is updated using self.define_supply

        This method creates a mapping between date and "vaccine events" in historical data
            corresponding to that date -- so that we can look up whether or not a vaccine group event occurs,
            rather than iterating through all items in self.vaccine_allocation

        Creates event_lookup_dict, a dictionary of dictionaries, with the same keys as self.vaccine_allocation,
            where each key corresponds to a vaccine group ("v_first", "v_second", "v_booster", "v_wane")
        self.event_lookup_dict[vaccine_type] is a dictionary
            the same length as self.vaccine_allocation[vaccine_ID]
        Each key in event_lookup_dict[vaccine_type] is a datetime object and the corresponding value is the
            i (index) of self.vaccine_allocation[vaccine_type] such that
            self.vaccine_allocation[vaccine_type][i]["supply"]["time"] matches the datetime object
        '''

        event_lookup_dict = {}
        for key in self.vaccine_allocation.keys():
            d = {}
            counter = 0
            for allocation_item in self.vaccine_allocation[key]:
                d[allocation_item["supply"]["time"]] = counter
                counter += 1
            event_lookup_dict[key] = d
        return event_lookup_dict

    def event_lookup(self, vaccine_type, date):
        '''
        Must be called after self.build_event_lookup_dict builds the event lookup dictionary

        vaccine_type is one of the keys of self.vaccine_allocation ("v_first", "v_second", "v_booster", "v_wane")
        date is a datetime object

        Returns the index i such that self.vaccine_allocation[vaccine_type][i]["supply"]["time"] == date
        Otherwise, returns None
        '''

        if date in self.event_lookup_dict[vaccine_type].keys():
            return self.event_lookup_dict[vaccine_type][date]

    def get_num_eligible(self, total_population, total_risk_gr, vaccine_group_name, v_in, v_out, date):

        '''

        :param total_population: integer, usually N parameter such as instance.N
        :param total_risk_gr: instance.A x instance.L
        :param vaccine_group_name: string of vaccine_group_name (see Vaccine.define_groups()) ("v_0", "v_1", "v_2", "v_3")
        :param v_in: tuple with strings of vaccine_types going "in" to that vaccine group
        :param v_out: tuple with strings of vaccine_types going "out" of that vaccine group
        :param date: datetime object
        :return: list of number eligible at that date, where each element corresponds to age/risk group
            (list is length A * L)
        '''

        # I don't know what dimension instance.N is, so need to check...

        N_in = np.zeros((total_risk_gr, 1))
        N_out = np.zeros((total_risk_gr, 1))

        for vaccine_type in v_in:
            event = self.event_lookup(vaccine_type, date)
            if event is not None:
                for i in range(event):
                    N_in += self.vaccine_allocation[vaccine_type][i]["assignment"].reshape((total_risk_gr,1))
            else:
                if date > self.vaccine_allocation[vaccine_type][0]["supply"]["time"]:
                    i = 0
                    event_date = self.vaccine_allocation[vaccine_type][i]["supply"]["time"]
                    while event_date < date:
                        N_in += self.vaccine_allocation[vaccine_type][i]["assignment"].reshape((total_risk_gr,1))
                        if i + 1 == len(self.vaccine_allocation[vaccine_type]):
                            break
                        i += 1
                        event_date = self.vaccine_allocation[vaccine_type][i]["supply"]["time"]

        for vaccine_type in v_out:
            event = self.event_lookup(vaccine_type, date)
            if event is not None:
                for i in range(event):
                    N_out += self.vaccine_allocation[vaccine_type][i]["assignment"].reshape((total_risk_gr,1))
            else:
                if date > self.vaccine_allocation[vaccine_type][0]["supply"]["time"]:
                    i = 0
                    event_date = self.vaccine_allocation[vaccine_type][i]["supply"]["time"]
                    while event_date < date:
                        N_out += self.vaccine_allocation[vaccine_type][i]["assignment"].reshape((total_risk_gr,1))
                        if i + 1 == len(self.vaccine_allocation[vaccine_type]):
                            break
                        i += 1
                        event_date = self.vaccine_allocation[vaccine_type][i]["supply"]["time"]

        if vaccine_group_name == 'v_0':
            N_eligible = total_population.reshape((total_risk_gr, 1)) - N_out
        else:
            N_eligible = N_in - N_out

        return N_eligible

    def define_supply(self, instance, vaccine_allocation_data, booster_allocation_data):
        '''
        Load vaccine supply and allocation data, and process them.
        Shift vaccine schedule for waiting vaccine to be effective, second dose and vaccine waning effect and also for booster dose.
        '''
        N = instance.N
        self.actual_vaccine_time = [time for time in vaccine_allocation_data['vaccine_time']]
        self.first_dose_time = [time + dt.timedelta(days = self.effect_time) for time in vaccine_allocation_data['vaccine_time']]
        self.second_dose_time = [time + dt.timedelta(days = self.second_dose_time + self.effect_time) for time in self.first_dose_time if time + dt.timedelta(days = self.second_dose_time + self.effect_time) <= instance.end_date]

        self.waning_time = [time + dt.timedelta(days = self.waning_time)  for time in vaccine_allocation_data['vaccine_time'] if time + dt.timedelta(days = self.waning_time) <= instance.end_date]
        self.vaccine_proportion = [amount for amount in vaccine_allocation_data['vaccine_amount']]

        self.vaccine_start_time = np.where(np.array(instance.cal.calendar) == self.actual_vaccine_time[0])[0]

        v_first_allocation = []
        v_second_allocation = []
        v_booster_allocation = []
        v_wane_allocation = []

        # Fixed vaccine allocation:
        for i in range(len(vaccine_allocation_data['A1-R1'])):
            vac_assignment = np.zeros((5, 2))
            vac_assignment[0, 0] = vaccine_allocation_data['A1-R1'][i]
            vac_assignment[0, 1] = vaccine_allocation_data['A1-R2'][i]
            vac_assignment[1, 0] = vaccine_allocation_data['A2-R1'][i]
            vac_assignment[1, 1] = vaccine_allocation_data['A2-R2'][i]
            vac_assignment[2, 0] = vaccine_allocation_data['A3-R1'][i]
            vac_assignment[2, 1] = vaccine_allocation_data['A3-R2'][i]
            vac_assignment[3, 0] = vaccine_allocation_data['A4-R1'][i]
            vac_assignment[3, 1] = vaccine_allocation_data['A4-R2'][i]
            vac_assignment[4, 0] = vaccine_allocation_data['A5-R1'][i]
            vac_assignment[4, 1] = vaccine_allocation_data['A5-R2'][i]

            if np.sum(vac_assignment) > 0:
                pro_round = vac_assignment/np.sum(vac_assignment)
            else:
                pro_round = np.zeros((5, 2))
            within_proportion = vac_assignment/N

            # First dose vaccine allocation:
            supply_first_dose =  {'time': self.first_dose_time[i], 'amount': self.vaccine_proportion[i], 'type': "first_dose"}
            allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'within_proportion': within_proportion,  'supply': supply_first_dose, 'type': 'first_dose', 'from': 'v_0', 'to': 'v_1'}
            v_first_allocation.append(allocation_item)

            # Second dose vaccine allocation:
            if i < len(self.second_dose_time):
                supply_second_dose =  {'time': self.second_dose_time[i], 'amount': self.vaccine_proportion[i], 'type': "second_dose"}
                allocation_item = {'assignment': vac_assignment, 'proportion': pro_round,'within_proportion': within_proportion,  'supply': supply_second_dose, 'type': 'second_dose', 'from': 'v_1', 'to': 'v_2'}
                v_second_allocation.append(allocation_item)

            # Waning vaccine efficacy:
            if i < len(self.waning_time):
                supply_waning =  {'time': self.waning_time[i], 'amount': self.vaccine_proportion[i], 'type': "waning"}
                allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'within_proportion': within_proportion, 'supply': supply_waning, 'type': 'waning', 'from': 'v_2', 'to': 'v_3'}
                v_wane_allocation.append(allocation_item)

        # Fixed booster vaccine allocation:
        if booster_allocation_data is not None:
            self.booster_time = [time  for time in booster_allocation_data['vaccine_time']]
            self.booster_proportion = [amount for amount in booster_allocation_data['vaccine_amount']]
            for i in range(len(booster_allocation_data['A1-R1'])):
                vac_assignment = np.zeros((5, 2))
                vac_assignment[0, 0] = booster_allocation_data['A1-R1'][i]
                vac_assignment[0, 1] = booster_allocation_data['A1-R2'][i]
                vac_assignment[1, 0] = booster_allocation_data['A2-R1'][i]
                vac_assignment[1, 1] = booster_allocation_data['A2-R2'][i]
                vac_assignment[2, 0] = booster_allocation_data['A3-R1'][i]
                vac_assignment[2, 1] = booster_allocation_data['A3-R2'][i]
                vac_assignment[3, 0] = booster_allocation_data['A4-R1'][i]
                vac_assignment[3, 1] = booster_allocation_data['A4-R2'][i]
                vac_assignment[4, 0] = booster_allocation_data['A5-R1'][i]
                vac_assignment[4, 1] = booster_allocation_data['A5-R2'][i]

                if np.sum(vac_assignment) > 0:
                    pro_round = vac_assignment/np.sum(vac_assignment)
                else:
                    pro_round = np.zeros((5, 2))
                within_proportion = vac_assignment/N

                # Booster dose vaccine allocation:
                supply_booster_dose =  {'time': self.booster_time[i], 'amount': self.booster_proportion[i], 'type': "booster_dose"}
                allocation_item = {'assignment': vac_assignment, 'proportion': pro_round, 'within_proportion': within_proportion,  'supply': supply_booster_dose, 'type': 'booster_dose', 'from': 'v_3', 'to': 'v_2'}
                v_booster_allocation.append(allocation_item)

        return {'v_first': v_first_allocation, 'v_second': v_second_allocation, 'v_booster': v_booster_allocation, 'v_wane': v_wane_allocation}
