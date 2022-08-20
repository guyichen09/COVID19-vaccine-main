import numpy as np
from itertools import product

datetime_formater = '%Y-%m-%d %H:%M:%S'

class MultiTierPolicy:
    '''
        A multi-tier policy allows for multiple tiers of lock-downs.
        Attrs:
            tiers (list of dict): a list of the tiers characterized by a dictionary
                with the following entries:
                    {
                        "transmission_reduction": float [0,1)
                        "cocooning": float [0,1)
                        "school_closure": int {0,1}
                    }

            lockdown_thresholds (list of list): a list with the thresholds for every
                tier. The list must have n-1 elements if there are n tiers. Each threshold
                is a list of values for evert time step of simulation.
            tier_type: functional form of the threshold (options are in THRESHOLD_TYPES)
            community_tranmission: (deprecated) CDC's old community tranmission threshold for staging.
                                    Not in use anymore.
    '''

    def __init__(self, instance, tiers, lockdown_thresholds, tier_type, community_tranmission):
        assert len(tiers) == len(lockdown_thresholds)
        self.tiers = tiers.tier
        self.tier_type = tier_type
        self.community_tranmission = community_tranmission
        self.lockdown_thresholds = lockdown_thresholds

        self.tier_history = None
        self._instance = instance

    def reset_tier_history(self):
        self.tier_history = None

    def compute_cost(self):
        return sum(
            self.tiers[i]['daily_cost'] for i in self.tier_history if i is not None and i in range(len(self.tiers)))

    def __repr__(self):
        return str(self.lockdown_thresholds)

    def __call__(self, t, ToIHT, IH, ToIY, ICU):
        '''
            Function that makes an instance of a policy a callable.
            Args:
                t (int): time period in the simulation
                z (object): deprecated, but maintained to avoid changes in the simulate function
                criStat (ndarray): the trigger statistics, previously daily admission, passed by the simulator
                IH (ndarray): hospitalizations admissions, passed by the simulator
                ** kwargs: additional parameters that are passed and are used elsewhere
        '''
        N = self._instance.N

        if self.tier_history is None:
            self.tier_history = [None for i in range(t)]

        if len(self.tier_history) > t:
            return

        ToIHT = np.array(ToIHT)
        IH = np.array(IH)
        ToIY = np.array(ToIY)
        ICU = np.array(ICU)

        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - self._instance.config['moving_avg_len'])
        criStat_total = ToIHT.sum((1, 2))
        criStat_avg = criStat_total[moving_avg_start:].mean()

        # print(criStat_avg)

        # Compute new cases per 100k:
        if len(ToIY) > 0:
            ToIY_avg = ToIY.sum((1, 2))[moving_avg_start:].sum() * 100000 / np.sum(N, axis=(0, 1))
        else:
            ToIY_avg = 0

        current_tier = self.tier_history[t - 1]
        T = self._instance.T

        # find new tier
        counter = 0
        lb_threshold = 0
        for lt in self.lockdown_thresholds:
            if criStat_avg >= lt:
                lb_threshold = counter
                counter += 1
                if counter == len(self.lockdown_thresholds):
                    break

        new_tier = lb_threshold

        # Check if community tranmission rate is included:
        if self.community_tranmission == "blue":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else:
                        new_tier = 2
            elif new_tier == 1:
                if ToIY_avg > 10:
                    new_tier = 2
        elif self.community_tranmission == "green":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else:
                        new_tier = 2

        if current_tier is None:  # bump to the next tier
            t_end = t + self.tiers[new_tier]['min_enforcing_time']

        elif new_tier > current_tier:
            t_end = t + self.tiers[new_tier]['min_enforcing_time']

        elif new_tier < current_tier:  # relax one tier, if safety trigger allows
            IH_total = IH[-1].sum()
            assert_safety_trigger = IH_total < self._instance.hosp_beds * self._instance.config['safety_threshold_frac']
            new_tier = new_tier if assert_safety_trigger else current_tier
            t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
            t_end = t + t_delta

        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = t + 1

        self.tier_history += [new_tier for i in range(t_end - t)]

class EpiSetup:
    '''
        A setup for the epidemiological parameters.
        Scenarios 6 corresponds to best guess parameters for UT group.
    '''

    def __init__(self):
        '''
            Initialize an instance of epidemiological parameters. If the
            parameter is random, is not initialize and is queried as a
            property
        '''

    @classmethod
    def load_file(cls, params):
        epi_params = cls()
        for (k, v) in params.items():
            if isinstance(v, list):
                if v[0] == "rnd_inverse" or v[0] == "rnd":
                    setattr(epi_params, k, ParamDistribution(*v))
                else:
                    setattr(epi_params, k, np.array(v))
            else:
                setattr(epi_params, k, v)
        return epi_params

    def update_rnd_stream(self, rnd_stream):
        '''
            Generates random parametes from a given random stream.
            Coupled parameters are updated as well.
            Args:
                rnd_stream (RandomState): a RandomState instance from numpy.
        '''

        # rnd_stream = None  #rnd_stream
        tempRecord = {}
        for k in vars(self):
            v = getattr(self, k)
            # if the attribute is random variable, generate a deterministic version
            if isinstance(v, ParamDistribution):
                tempRecord[v.param_name] = v.sample(rnd_stream)
            elif isinstance(v, np.ndarray):
                listDistrn = True
                # if it is a list of random variable, generate a list of deterministic values
                vList = []
                outList = []
                outName = None
                for vItem in v:
                    try:
                        vValue = ParamDistribution(*vItem)
                        outList.append(vValue.sample(rnd_stream))
                        outName = vValue.param_name
                    except:
                        vValue = 0
                    vList.append(vValue)
                    listDistrn = listDistrn and isinstance(vValue, ParamDistribution)
                if listDistrn:
                    tempRecord[outName] = np.array(outList)

        for k in tempRecord.keys():
            setattr(self, k, tempRecord[k])

        self.omega_P = np.array([(self.tau * self.omega_IY * (self.YHR_overall[a] / self.Eta[a] +
                                                              (1 - self.YHR_overall[a]) / self.gamma_IY) +
                                  (1 - self.tau) * self.omega_IA / self.gamma_IA) /
                                 (self.tau * self.omega_IY +
                                  (1 - self.tau) * self.omega_IA) * self.rho_Y * self.pp / (1 - self.pp)
                                 for a in range(len(self.YHR_overall))])
        self.omega_PA = self.omega_IA * self.omega_P
        self.omega_PY = self.omega_IY * self.omega_P
        # pi is computed using risk based hosp rate
        self.pi = np.array([
            self.YHR[a] * self.gamma_IY / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
            for a in range(len(self.YHR))
        ])
        self.YFR = self.IFR / self.tau
        self.HFR = self.YFR / self.YHR
        self.rIH0 = self.rIH
        self.YHR0 = self.YHR
        self.YHR_overall0 = self.YHR_overall
        # if gamma_IH and mu are lists, reshape them for right dimension
        if isinstance(self.gamma_IH, np.ndarray):
            self.gamma_IH = self.gamma_IH.reshape(self.gamma_IH.size, 1)
            self.gamma_IH0 = self.gamma_IH.copy()
        if isinstance(self.mu, np.ndarray):
            self.mu = self.mu.reshape(self.mu.size, 1)
            self.mu0 = self.mu.copy()
        try:
            self.HICUR0 = self.HICUR
            self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH - self.mu) * self.HICUR)
            if isinstance(self.gamma_ICU, np.ndarray):
                self.gamma_ICU = self.gamma_ICU.reshape(self.gamma_ICU.size, 1)
                self.gamma_ICU0 = self.gamma_ICU.copy()
            if isinstance(self.mu_ICU, np.ndarray):
                self.mu_ICU = self.mu_ICU.reshape(self.mu_ICU.size, 1)
                self.mu_ICU0 = self.mu_ICU.copy()
            self.nu_ICU = self.gamma_ICU * self.ICUFR / (self.mu_ICU + (self.gamma_ICU - self.mu_ICU) * self.ICUFR)
        except:
            self.nu = self.gamma_IH * self.HFR / (self.mu + (self.gamma_IH - self.mu) * self.HFR)

        self.beta = self.beta0

    def delta_update_param(self, prev):
        '''
            Update parameters according to delta variant prevelance.
        '''

        self.beta = self.beta0 * (1 - prev) + self.beta0 * (1.65) * prev  # increased transmission

        E_new = 1 / self.sigma_E - 1.5
        self.sigma_E = self.sigma_E * (1 - prev) + (1 / E_new) * prev  # decreased incubation period.
        # print(self.sigma_E)

        self.YHR = self.YHR * (1 - prev) + self.YHR * (1.8) * prev  # increased hospitalization rate.
        self.YHR_overall = self.YHR_overall * (1 - prev) + self.YHR_overall * (1.8) * prev

        # Update parameters where YHR is used:
        self.omega_P = np.array([(self.tau * self.omega_IY * (self.YHR_overall[a] / self.Eta[a] +
                                                              (1 - self.YHR_overall[a]) / self.gamma_IY) +
                                  (1 - self.tau) * self.omega_IA / self.gamma_IA) /
                                 (self.tau * self.omega_IY +
                                  (1 - self.tau) * self.omega_IA) * self.rho_Y * self.pp / (1 - self.pp)
                                 for a in range(len(self.YHR_overall))])
        self.omega_PA = self.omega_IA * self.omega_P
        self.omega_PY = self.omega_IY * self.omega_P

        self.pi = np.array([
            self.YHR[a] * self.gamma_IY / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
            for a in range(len(self.YHR))
        ])
        self.HFR = self.YFR / self.YHR

        try:
            self.HICUR0 = self.HICUR
            self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH - self.mu) * self.HICUR)
            if isinstance(self.gamma_ICU, np.ndarray):
                self.gamma_ICU = self.gamma_ICU.reshape(self.gamma_ICU.size, 1)
                self.gamma_ICU0 = self.gamma_ICU.copy()
            if isinstance(self.mu_ICU, np.ndarray):
                self.mu_ICU = self.mu_ICU.reshape(self.mu_ICU.size, 1)
                self.mu_ICU0 = self.mu_ICU.copy()
            self.nu_ICU = self.gamma_ICU * self.ICUFR / (self.mu_ICU + (self.gamma_ICU - self.mu_ICU) * self.ICUFR)
        except:
            self.nu = self.gamma_IH * self.HFR / (self.mu + (self.gamma_IH - self.mu) * self.HFR)

        # Update hospital dynamic parameters:
        self.gamma_ICU = (self.gamma_ICU0 * (1 + self.alpha1)) * (1 - prev) + (
                    self.gamma_ICU0 * 0.65 * (1 + self.alpha1_delta)) * prev
        self.mu_ICU = (self.mu_ICU0 * (1 + self.alpha3)) * (1 - prev) + (
                    self.mu_ICU0 * 0.65 * (1 + self.alpha3_delta)) * prev
        self.gamma_IH = (self.gamma_IH0 * (1 - self.alpha2)) * (1 - prev) + (
                    self.gamma_IH0 * (1 - self.alpha2_delta)) * prev

        self.alpha4 = self.alpha4_delta * prev + self.alpha4 * (1 - prev)

    def omicron_update_param(self, prev):
        '''
            Update parameters according omicron.
            Assume increase in the tranmission.
            The changes in hosp dynamic in Austin right before omicron emerged.
        '''
        self.beta = self.beta * (1 - prev) + self.beta * (self.omicron_beta) * prev  # increased transmission

        self.YHR = self.YHR0 * (1 - prev) + self.YHR0 * 0.9 * prev
        self.YHR_overall = self.YHR_overall * (1 - prev) + self.YHR_overall * 0.9 * prev

        # Update parameters where YHR is used:
        self.omega_P = np.array([(self.tau * self.omega_IY * (self.YHR_overall[a] / self.Eta[a] +
                                                              (1 - self.YHR_overall[a]) / self.gamma_IY) +
                                  (1 - self.tau) * self.omega_IA / self.gamma_IA) /
                                 (self.tau * self.omega_IY +
                                  (1 - self.tau) * self.omega_IA) * self.rho_Y * self.pp / (1 - self.pp)
                                 for a in range(len(self.YHR_overall))])
        self.omega_PA = self.omega_IA * self.omega_P
        self.omega_PY = self.omega_IY * self.omega_P

        self.pi = np.array([
            self.YHR[a] * self.gamma_IY / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
            for a in range(len(self.YHR))
        ])
        self.HFR = self.YFR / self.YHR

        try:
            self.HICUR0 = self.HICUR
            self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH - self.mu) * self.HICUR)
            if isinstance(self.gamma_ICU, np.ndarray):
                self.gamma_ICU = self.gamma_ICU.reshape(self.gamma_ICU.size, 1)
                self.gamma_ICU0 = self.gamma_ICU.copy()
            if isinstance(self.mu_ICU, np.ndarray):
                self.mu_ICU = self.mu_ICU.reshape(self.mu_ICU.size, 1)
                self.mu_ICU0 = self.mu_ICU.copy()
            self.nu_ICU = self.gamma_ICU * self.ICUFR / (self.mu_ICU + (self.gamma_ICU - self.mu_ICU) * self.ICUFR)
        except:
            self.nu = self.gamma_IH * self.HFR / (self.mu + (self.gamma_IH - self.mu) * self.HFR)

        # Update hospital dynamic parameters:
        self.gamma_ICU = self.gamma_ICU0 * (1 + self.alpha1_omic) * 1.1 * prev + (
                    self.gamma_ICU0 * 0.65 * (1 + self.alpha1_delta)) * (1 - prev)
        self.mu_ICU = self.mu_ICU0 * (1 + self.alpha3_omic) * prev + (self.mu_ICU0 * 0.65 * (1 + self.alpha3_delta)) * (
                    1 - prev)
        self.gamma_IH = self.gamma_IH0 * (1 - self.alpha2_omic) * prev + (self.gamma_IH0 * (1 - self.alpha2_delta)) * (
                    1 - prev)

        self.alpha4 = self.alpha4_omic * prev + self.alpha4_delta * (1 - prev)

    def variant_update_param(self, prev):
        '''
            Assume an imaginary new variant that is more transmissible.
        '''
        self.beta = self.beta * (1 - prev) + self.beta * (self.new_variant_beta) * prev  # increased transmission

    def effective_phi(self, school, cocooning, social_distance, demographics, day_type):
        '''
            school (int): yes (1) / no (0) schools are closed
            cocooning (float): percentage of transmition reduction [0,1]
            social_distance (int): percentage of social distance (0,1)
            demographics (ndarray): demographics by age and risk group
            day_type (int): 1 Weekday, 2 Weekend, 3 Holiday, 4 Long Holiday
        '''

        A = len(demographics)  # number of age groups
        L = len(demographics[0])  # number of risk groups
        d = demographics  # A x L demographic data
        phi_all_extended = np.zeros((A, L, A, L))
        phi_school_extended = np.zeros((A, L, A, L))
        phi_work_extended = np.zeros((A, L, A, L))
        for a, b in product(range(A), range(A)):
            phi_ab_split = np.array([
                [d[b, 0], d[b, 1]],
                [d[b, 0], d[b, 1]],
            ])
            phi_ab_split = phi_ab_split / phi_ab_split.sum(1)
            phi_ab_split = 1 + 0 * phi_ab_split / phi_ab_split.sum(1)
            phi_all_extended[a, :, b, :] = self.phi_all[a, b] * phi_ab_split
            phi_school_extended[a, :, b, :] = self.phi_school[a, b] * phi_ab_split
            phi_work_extended[a, :, b, :] = self.phi_work[a, b] * phi_ab_split

        # Apply school closure and social distance
        if day_type == 1:  # Weekday
            phi_age_risk = (1 - social_distance) * (phi_all_extended - school * phi_school_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = phi_all_extended - school * phi_school_extended
                phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk
        elif day_type == 2 or day_type == 3:  # is a weekend or holiday
            phi_age_risk = (1 - social_distance) * (phi_all_extended - phi_school_extended - phi_work_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = (phi_all_extended - phi_school_extended - phi_work_extended)
                phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk
        else:
            phi_age_risk = (1 - social_distance) * (phi_all_extended - phi_school_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = (phi_all_extended - phi_school_extended)
                phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk

    def update_hosp_duration(self):
        self.gamma_ICU = self.gamma_ICU0 * (1 + self.alpha1)
        self.mu_ICU = self.mu_ICU0 * (1 + self.alpha3)
        self.gamma_IH = self.gamma_IH0 * (1 - self.alpha2)

    def update_icu_params(self, rdrate):
        # update the ICU admission parameter HICUR and update nu
        self.HICUR = self.HICUR * rdrate
        self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH - self.mu) * self.HICUR)
        self.rIH = 1 - (1 - self.rIH) * rdrate

    def update_icu_all(self, t, otherInfo):
        if 'rIH' in otherInfo.keys():
            if t in otherInfo['rIH'].keys():
                self.rIH = otherInfo['rIH'][t]
            else:
                self.rIH = self.rIH0
        if 'HICUR' in otherInfo.keys():
            if t in otherInfo['HICUR'].keys():
                self.HICUR = otherInfo['HICUR'][t]
            else:
                self.HICUR = self.HICUR0
        if 'mu' in otherInfo.keys():
            if t in otherInfo['mu'].keys():
                self.mu = self.mu0.copy() / otherInfo['mu'][t]
            else:
                self.mu = self.mu0.copy()
        self.nu = self.gamma_IH * self.HICUR / (self.mu + (self.gamma_IH - self.mu) * self.HICUR)

class ParamDistribution:
    '''
        A class to encapsulate epi paramters that are random
        Attrs:
            is_inverse (bool): if True, the parameter is used in the model as 1 / x.
            param_name (str): Name of the parameter, used in EpiParams as attribute name.
            distribution_name (str): Name of the distribution, matching functions in np.random.
            det_val (float): Value of the parameter for deterministic simulations.
            params (list): paramters if the distribution
    '''

    def __init__(self, inv_opt, param_name, distribution_name, det_val, params):
        if inv_opt == "rnd_inverse":
            self.is_inverse = True
        elif inv_opt == "rnd":
            self.is_inverse = False
        self.param_name = param_name
        self.distribution_name = distribution_name
        self.det_val = det_val
        self.params = params

    def sample(self, rnd_stream, dim=1):
        '''
            Sample random variable with given distribution name, parameters and dimension.
            Args:
                rnd_stream (np.RandomState): a random stream. If None, det_val is returned.
                dim (int or tuple): dimmention of the parameter (default is 1).
        '''
        if rnd_stream is not None:
            dist_func = getattr(rnd_stream, self.distribution_name)
            args = self.params
            if self.is_inverse:
                return np.squeeze(1 / dist_func(*args, dim))
            else:
                return np.squeeze(dist_func(*args, dim))
        else:
            if self.is_inverse:
                return 1 / self.det_val
            else:
                return self.det_val

class VaccineGroup:
    def __init__(self, v_name, v_beta_reduct, v_tau_reduct, v_beta_reduct_delta, v_tau_reduct_delta, v_tau_reduct_omicron, instance):
        '''
        Define each vaccine status as a group. Define each set of compartments for vaccine group.

        Parameters
        ----------
        v_name : string
            vaccine status type.
        v_beta_reduct : float [0,1]
            reduction in transmission.
        v_tau_reduct : float [0,1]
            reduction in symptomatic infection.
        instance :
            problem instance.

        '''
        self.v_beta_reduct = v_beta_reduct
        self.v_tau_reduct = v_tau_reduct
        self.v_beta_reduct_delta = v_beta_reduct_delta
        self.v_tau_reduct_delta = v_tau_reduct_delta
        self.v_tau_reduct_omicron = v_tau_reduct_omicron
        self.v_name = v_name

        if self.v_name == 'v_0':
            self.v_in = ()
            self.v_out = ('v_first',)

        elif self.v_name == 'v_1':
            self.v_in = ('v_first',)
            self.v_out = ('v_second',)

        elif self.v_name == 'v_2':
            self.v_in = ('v_second', 'v_booster')
            self.v_out = ('v_wane',)

        else:
            self.v_in = ('v_wane',)
            self.v_out = ('v_booster',)

        self.N = instance.N
        self.I0 = instance.I0
        self.A = instance.A
        self.L = instance.L
        self.age_risk_matrix_shape = (self.A, self.L)
        self.types = "float"

        step_size = instance.config['step_size']

        self.S = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.E = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.IA = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.IY = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.PA = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.PY = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.R = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.D = np.zeros(self.age_risk_matrix_shape, dtype=self.types)

        self.IH = np.zeros(self.age_risk_matrix_shape, dtype=self.types)
        self.ICU = np.zeros(self.age_risk_matrix_shape, dtype=self.types)

        self.IYIH = np.zeros(self.age_risk_matrix_shape)
        self.IYICU = np.zeros(self.age_risk_matrix_shape)
        self.IHICU = np.zeros(self.age_risk_matrix_shape)
        self.ToICU = np.zeros(self.age_risk_matrix_shape)
        self.ToIHT = np.zeros(self.age_risk_matrix_shape)
        self.ToICUD = np.zeros(self.age_risk_matrix_shape)
        self.ToIYD = np.zeros(self.age_risk_matrix_shape)
        self.ToIA = np.zeros(self.age_risk_matrix_shape)
        self.ToIY = np.zeros(self.age_risk_matrix_shape)

        self._S = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._E = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._IA = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._IY = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._PA = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._PY = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._IH = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._ICU = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._R = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)
        self._D = np.zeros((step_size + 1, self.A, self.L), dtype=self.types)

        self._IYIH = np.zeros((step_size, self.A, self.L))
        self._IYICU = np.zeros((step_size, self.A, self.L))
        self._IHICU = np.zeros((step_size, self.A, self.L))
        self._ToICU = np.zeros((step_size, self.A, self.L))
        self._ToIHT = np.zeros((step_size, self.A, self.L))
        self._ToICUD = np.zeros((step_size, self.A, self.L))
        self._ToIYD = np.zeros((step_size, self.A, self.L))
        self._ToIA = np.zeros((step_size, self.A, self.L))
        self._ToIY = np.zeros((step_size, self.A, self.L))

        if self.v_name == 'v_0':
            # Initial Conditions (assumed)
            self.PY = self.I0
            self.R = 0
            self.S = self.N - self.PY - self.IY

        self._S[0] = self.S
        self._E[0] = self.E
        self._IA[0] = self.IA
        self._IY[0] = self.IY
        self._PA[0] = self.PA
        self._PY[0] = self.PY
        self._R[0] = self.R
        self._D[0] = self.D

        self._IH[0] = self.IH
        self._ICU[0] = self.ICU

    def delta_update(self, prev):
        '''
            Update efficacy according to delta variant (VoC) prevelance.
        '''

        self.v_beta_reduct = self.v_beta_reduct * (1 - prev) + self.v_beta_reduct_delta * prev #decreased efficacy against infection.
        self.v_tau_reduct = self.v_tau_reduct * (1 - prev) + self.v_tau_reduct_delta * prev #decreased efficacy against symptomatic infection.

    def omicron_update(self, prev):
        '''
            Update efficacy according to omicron variant (VoC) prevelance.
        '''

        self.v_tau_reduct = self.v_tau_reduct * (1 - prev) + self.v_tau_reduct_omicron * prev


