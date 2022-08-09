'''
This module runs a simulation of the SEIYARDH model for a single city,
considering different age groups and seven compartments. This model is
based of Zhanwei's Du model, Meyer's Group.

This module also contains functions to run the simulations in parallel
and a class to properly define a calendar (SimCalendar).
'''

import datetime as dt
import numpy as np
from vaccine_params import VaccineGroup
from VaccineAllocation import config
import copy

class SimulationReplication:
    '''
    Simulates an SIR-type model with seven compartments, multiple age groups,
    and risk different age groups:
    Compartments
        S: Susceptible
        E: Exposed
        IY: Infected symptomatic
        IA: Infected asymptomatic
        IH: Infected hospitalized
        ICU: Infected ICU
        R: Recovered
        D: Death
    Connections between compartments:
        S-E, E-IY, E-IA, IY-IH, IY-R, IA-R, IH-R, IH-ICU, ICU-D, ICU-R
    '''

    def __init__(self, instance, vaccine, rng_seed):

        self.instance = instance
        self.vaccine = vaccine
        self.rng_seed = rng_seed
        self.types = "float"
        self.step_size = config["step_size"]

        self.define_epi()
        self.define_groups()

        self.age_risk_matrix_shape = (self.instance.A, self.instance.L)

        self.ICU_history = [np.zeros(self.age_risk_matrix_shape, dtype=self.types)]
        self.IH_history = [np.zeros(self.age_risk_matrix_shape, dtype=self.types)]

    def define_epi(self):

        self.rng_generator = np.random.RandomState(self.rng_seed) if self.rng_seed >= 0 else None

        epi = self.instance.epi
        epi_orig = copy.deepcopy(epi)
        epi_rand = copy.deepcopy(epi)

        epi_rand.update_rnd_stream(self.rng_generator)
        epi_orig.update_rnd_stream(None)

        epi_rand.update_hosp_duration()
        epi_orig.update_hosp_duration()

        self.epi = epi_rand
        self.epi_rand = epi_rand

    def define_groups(self):
        '''
             Define different vaccine groups.
             {
                 group 0: unvaccinated group,
                 group 1: partially vaccinated,
                 group 2: fully vaccinated,
                 group 3: waning efficacy group.
             }

             Assuming:
                 - one type of vaccine with two-doses
                 - after first dose move to group 1 compartment.
                 - after second dose move to group 2 compartment.
                 - after efficacy wanes, move to group 3 compartment.
                 - If booster shot is administered, move from group 3 compartment to group 2 compartment.
        '''
        self.vaccine_groups = []
        self.vaccine_groups.append(VaccineGroup('v_0', 0, 0, 0, 0, 0, self.instance))  # unvaccinated
        self.vaccine_groups.append(
            VaccineGroup('v_1', self.vaccine.beta_reduct[1], self.vaccine.tau_reduct[1], self.vaccine.beta_reduct_delta[1],
                          self.vaccine.tau_reduct_delta[1], self.vaccine.tau_reduct_omicron[1],
                          self.instance))  # partially vaccinated
        self.vaccine_groups.append(
            VaccineGroup('v_2', self.vaccine.beta_reduct[2], self.vaccine.tau_reduct[2], self.vaccine.beta_reduct_delta[2],
                          self.vaccine.tau_reduct_delta[2], self.vaccine.tau_reduct_omicron[2], self.instance))  # fully vaccinated
        self.vaccine_groups.append(
            VaccineGroup('v_3', self.vaccine.beta_reduct[0], self.vaccine.tau_reduct[0], self.vaccine.beta_reduct_delta[0],
                          self.vaccine.tau_reduct_delta[0], self.vaccine.tau_reduct_omicron[0], self.instance))  # waning efficacy
        self.vaccine_groups = tuple(self.vaccine_groups)

    def immune_escape(self, immune_escape_rate, t):
        '''
            This function move recovered and vaccinated individuals to waning
            efficacy susceptible compartment after omicron become the prevelant
            virus type.
        '''
        for idx, v_groups in enumerate(self.vaccine_groups):
            if self.types == 'int':
                moving_people = (v_groups._R[self.step_size] * immune_escape_rate).astype(int)
            else:
                moving_people = v_groups._R[self.step_size] * immune_escape_rate

            v_groups.R -= moving_people
            self.vaccine_groups[3].S += moving_people

            if v_groups.v_name == 'v_1' or v_groups.v_name == 'v_2':
                if self.types == 'int':
                    moving_people = (v_groups._S[self.step_size] * immune_escape_rate).astype(int)
                else:
                    moving_people = v_groups._S[self.step_size] * immune_escape_rate

                v_groups.S -= moving_people
                self.vaccine_groups[3].S += moving_people

    def simulate_time_period(self, time_start, time_end, data):

        for t in range(time_start, time_end):

            self.simulate_t(t)

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

            # Additional tracking variables (for triggers)
            self.IYIH = np.zeros(self.age_risk_matrix_shape)
            self.IYICU = np.zeros(self.age_risk_matrix_shape)
            self.IHICU = np.zeros(self.age_risk_matrix_shape)
            self.ToICU = np.zeros(self.age_risk_matrix_shape)
            self.ToIHT = np.zeros(self.age_risk_matrix_shape)
            self.ToICUD = np.zeros(self.age_risk_matrix_shape)
            self.ToIYD = np.zeros(self.age_risk_matrix_shape)
            self.ToIA = np.zeros(self.age_risk_matrix_shape)
            self.ToIY = np.zeros(self.age_risk_matrix_shape)

            for v_group in self.vaccine_groups:

                self.S += v_group.S
                self.E += v_group.E
                self.IA += v_group.IA
                self.IY += v_group.IY
                self.PA += v_group.PA
                self.PY += v_group.PY
                self.R += v_group.R
                self.D += v_group.D

                self.IH += v_group.IH
                self.ICU += v_group.ICU

                # Update daily values
                self.IYIH += v_group.IYIH
                self.IYICU += v_group.IYICU
                self.IHICU += v_group.IHICU
                self.ToICU += v_group.ToICU
                self.ToIHT += v_group.ToIHT
                self.ToIYD += v_group.ToIYD
                self.ToICUD += v_group.ToICUD
                self.ToIA += v_group.ToIA
                self.ToIY += v_group.ToIY

            self.ICU_history.append(self.ICU)
            self.IH_history.append(self.IH)

            total_imbalance = np.sum(self.S + self.E + self.IA + self.IY + self.R + self.D + self.PA + self.PY + self.IH + self.ICU) - np.sum(self.instance.N)

            assert np.abs(total_imbalance) < 1E-2, f'fPop unbalanced {total_imbalance} at time {calendar[t]}, {t}'

    def simulate_t(self, t_date):

        A, L = self.instance.A, self.instance.L
        N = self.instance.N

        calendar = self.instance.cal.calendar

        for t_idx in range(1):
            t = t_date
            # Get dynamic intervention and corresponding contact matrix
            #k_t, kwargs = policy(t, criStat=eval(kwargs["policy_field"])[:t], IH=IH[:t], **kwargs)

            # LP uncommented
            # k_t =  policy._intervention_history[t]
            # phi_t = interventions[k_t].phi(calendar.get_day_type(t))

            # school, cocooning, social_distance, demographics, day_type

            self.epi = copy.deepcopy(self.epi_rand)

            if t < len(self.instance.real_hosp):
                phi_t = self.epi.effective_phi(self.instance.cal.schools_closed[t],
                                       self.instance.cal.fixed_cocooning[t],
                                       self.instance.cal.fixed_transmission_reduction[t],
                                       N / N.sum(),
                                       self.instance.cal.get_day_type(t))
            # NEED TO DELETE THIS THIS IS JUST TO MAKE IT WORK FOR PAST HISTORICAL PERIOD
            else:
                phi_t = self.epi.effective_phi(self.instance.cal.schools_closed[len(self.instance.real_hosp) - 1],
                                       self.instance.cal.fixed_cocooning[len(self.instance.real_hosp) - 1],
                                       self.instance.cal.fixed_transmission_reduction[len(self.instance.real_hosp) - 1],
                                       N / N.sum(),
                                       self.instance.cal.get_day_type(len(self.instance.real_hosp)))

            if calendar[t] >= self.instance.delta_start:
                days_since_delta_start = (calendar[t] - self.instance.delta_start).days
                for v_groups in self.vaccine_groups:
                    v_groups.delta_update(self.instance.delta_prev[days_since_delta_start])
                self.epi.delta_update_param(self.instance.delta_prev[days_since_delta_start])

            #Update epi parameters for omicron:
            if calendar[t] >= self.instance.omicron_start:
                days_since_omicron_start = (calendar[t] - self.instance.omicron_start).days
                self.epi.omicron_update_param(self.instance.omicron_prev[days_since_omicron_start])
                for v_groups in self.vaccine_groups:
                    v_groups.omicron_update(self.instance.delta_prev[days_since_delta_start])

            # Assume an imaginary new variant in May, 2022:
            if self.epi.new_variant == True:
                days_since_variant_start = (calendar[t] - self.instance.variant_start).days
                if calendar[t] >= self.instance.variant_start:
                    self.epi.variant_update_param(self.instance.variant_prev[days_since_variant_start])

            if self.instance.otherInfo == {}:
                if t > kwargs["rd_start"] and t <= kwargs["rd_end"]:
                    self.epi.update_icu_params(kwargs["rd_rate"])
            else:
                self.epi.update_icu_all(t,self.instance.otherInfo)

            rate_E = discrete_approx(self.epi.sigma_E, self.step_size)

            rate_IYR = discrete_approx(np.array([[(1 - self.epi.pi[a, l]) * self.epi.gamma_IY * (1 - self.epi.alpha4) for l in range(L)] for a in range(A)]), self.step_size)
            rate_IYD = discrete_approx(np.array([[(1 - self.epi.pi[a, l]) * self.epi.gamma_IY * self.epi.alpha4 for l in range(L)] for a in range(A)]), self.step_size)
            rate_IAR = discrete_approx(np.tile(self.epi.gamma_IA, (L, A)).transpose(), self.step_size)
            rate_PAIA = discrete_approx(np.tile(self.epi.rho_A, (L, A)).transpose(), self.step_size)
            rate_PYIY = discrete_approx(np.tile(self.epi.rho_Y, (L, A)).transpose(), self.step_size)
            rate_IYH = discrete_approx(np.array([[(self.epi.pi[a, l]) * self.epi.Eta[a] * self.epi.rIH for l in range(L)] for a in range(A)]), self.step_size)
            rate_IYICU = discrete_approx(np.array([[(self.epi.pi[a, l]) * self.epi.Eta[a] * (1 - self.epi.rIH) for l in range(L)] for a in range(A)]), self.step_size)
            rate_IHICU = discrete_approx(self.epi.nu*self.epi.mu,self.step_size)
            rate_IHR = discrete_approx((1 - self.epi.nu)*self.epi.gamma_IH, self.step_size)
            rate_ICUD = discrete_approx(self.epi.nu_ICU*self.epi.mu_ICU, self.step_size)
            rate_ICUR = discrete_approx((1 - self.epi.nu_ICU)*self.epi.gamma_ICU, self.step_size)

            if t >= 711: #date corresponding to 02/07/2022
                rate_immune = discrete_approx(self.epi.immune_evasion, self.step_size)

            for _t in range(self.step_size):
                # Dynamics for dS

                for v_groups in self.vaccine_groups:

                    dSprob_sum = np.zeros((5,2))

                    for v_groups_temp in self.vaccine_groups:

                        # Vectorized version for efficiency. For-loop version commented below
                        temp1 = np.matmul(np.diag(self.epi.omega_PY), v_groups_temp._PY[_t, :, :]) + \
                            np.matmul(np.diag(self.epi.omega_PA), v_groups_temp._PA[_t, :, :]) + \
                                self.epi.omega_IA * v_groups_temp._IA[_t, :, :] + \
                                    self.epi.omega_IY * v_groups_temp._IY[_t, :, :]

                        temp2 = np.sum(N, axis=1)[np.newaxis].T
                        temp3 = np.divide(np.multiply(self.epi.beta * phi_t / self.step_size, temp1), temp2)

                        dSprob = np.sum(temp3, axis=(2, 3))
                        dSprob_sum = dSprob_sum + dSprob

                    if t >= 711 and v_groups.v_name == 'v_2':#date corresponding to 02/07/2022
                        _dS = self.rv_gen(v_groups._S[_t], rate_immune + (1 - v_groups.v_beta_reduct)*dSprob_sum)
                            # Dynamics for E
                        if self.types == 'int':
                            _dSE = np.round( _dS * ((1 - v_groups.v_beta_reduct)*dSprob_sum) / (rate_immune + (1 - v_groups.v_beta_reduct)*dSprob_sum))
                        else:
                            _dSE = _dS * ((1 - v_groups.v_beta_reduct)*dSprob_sum) / (rate_immune + (1 - v_groups.v_beta_reduct)*dSprob_sum)

                        E_out = self.rv_gen(v_groups._E[_t], rate_E)
                        v_groups._E[_t + 1] = v_groups._E[_t] + _dSE - E_out

                        _dSR = _dS - _dSE
                        self.vaccine_groups[3]._S[_t + 1] = self.vaccine_groups[3]._S[_t + 1] + _dSR

                    else:
                        _dS = self.rv_gen(v_groups._S[_t], (1 - v_groups.v_beta_reduct)*dSprob_sum)
                        # Dynamics for E
                        E_out = self.rv_gen(v_groups._E[_t], rate_E)
                        v_groups._E[_t + 1] = v_groups._E[_t] + _dS - E_out


                    if t >= 711 and v_groups.v_name != 'v_3':
                        immune_escape_R = self.rv_gen(v_groups._R[_t], rate_immune)
                        self.vaccine_groups[3]._S[_t + 1] = self.vaccine_groups[3]._S[_t + 1] + immune_escape_R

                    v_groups._S[_t + 1] = v_groups._S[_t + 1] + v_groups._S[_t] - _dS

                    # Dynamics for PY
                    EPY = self.rv_gen(E_out, self.epi.tau * ( 1 - v_groups.v_tau_reduct))
                    PYIY = self.rv_gen(v_groups._PY[_t], rate_PYIY)
                    v_groups._PY[_t + 1] = v_groups._PY[_t] + EPY - PYIY

                    # Dynamics for PA
                    EPA = E_out - EPY
                    PAIA = self.rv_gen(v_groups._PA[_t], rate_PAIA)
                    v_groups._PA[_t + 1] = v_groups._PA[_t] + EPA - PAIA

                    # Dynamics for IA
                    IAR = self.rv_gen(v_groups._IA[_t], rate_IAR)
                    v_groups._IA[_t + 1] = v_groups._IA[_t] + PAIA - IAR

                    # Dynamics for IY
                    IYR = self.rv_gen(v_groups._IY[_t], rate_IYR)
                    IYD = self.rv_gen(v_groups._IY[_t] - IYR, rate_IYD)
                    v_groups._IYIH[_t] = self.rv_gen(v_groups._IY[_t] - IYR - IYD, rate_IYH)
                    v_groups._IYICU[_t] = self.rv_gen(v_groups._IY[_t] - IYR - IYD - v_groups._IYIH[_t], rate_IYICU)
                    v_groups._IY[_t + 1] = v_groups._IY[_t] + PYIY - IYR - IYD - v_groups._IYIH[_t] - v_groups._IYICU[_t]

                    # Dynamics for IH
                    IHR = self.rv_gen(v_groups._IH[_t], rate_IHR)
                    v_groups._IHICU[_t] = self.rv_gen(v_groups._IH[_t] - IHR, rate_IHICU)
                    v_groups._IH[_t + 1] = v_groups._IH[_t] + v_groups._IYIH[_t] - IHR - v_groups._IHICU[_t]

                    # Dynamics for ICU
                    ICUR = self.rv_gen(v_groups._ICU[_t], rate_ICUR)
                    ICUD = self.rv_gen(v_groups._ICU[_t] - ICUR, rate_ICUD)
                    v_groups._ICU[_t + 1] = v_groups._ICU[_t] + v_groups._IHICU[_t] - ICUD - ICUR + v_groups._IYICU[_t]
                    v_groups._ToICU[_t] = v_groups._IYICU[_t] + v_groups._IHICU[_t]
                    v_groups._ToIHT[_t] = v_groups._IYICU[_t] + v_groups._IYIH[_t]

                    # Dynamics for R
                    if t >= 711 and v_groups.v_name != 'v_3':
                        v_groups._R[_t + 1] = v_groups._R[_t] + IHR + IYR + IAR + ICUR - immune_escape_R
                    else:
                        v_groups._R[_t + 1] = v_groups._R[_t] + IHR + IYR + IAR + ICUR

                    # Dynamics for D
                    v_groups._D[_t + 1] = v_groups._D[_t] + ICUD + IYD
                    v_groups._ToICUD[_t] = ICUD
                    v_groups._ToIYD[_t] = IYD
                    v_groups._ToIA[_t] = PAIA
                    v_groups._ToIY[_t] = PYIY

            for idx, v_groups in enumerate(self.vaccine_groups):
                # End of the daily disctretization
                v_groups.S = v_groups._S[self.step_size].copy()
                v_groups.E = v_groups._E[self.step_size].copy()
                v_groups.IA = v_groups._IA[self.step_size].copy()
                v_groups.IY = v_groups._IY[self.step_size].copy()
                v_groups.PA = v_groups._PA[self.step_size].copy()
                v_groups.PY = v_groups._PY[self.step_size].copy()
                v_groups.R = v_groups._R[self.step_size].copy()
                v_groups.D = v_groups._D[self.step_size].copy()

                v_groups.IH = v_groups._IH[self.step_size].copy()
                v_groups.ICU = v_groups._ICU[self.step_size].copy()

                v_groups.IYIH = v_groups._IYIH.sum(axis=0)
                v_groups.IYICU = v_groups._IYICU.sum(axis=0)
                v_groups.IHICU = v_groups._IHICU.sum(axis=0)
                v_groups.ToICU = v_groups._ToICU.sum(axis=0)
                v_groups.ToIHT = v_groups._ToIHT.sum(axis=0)
                v_groups.ToICUD = v_groups._ToICUD.sum(axis=0)
                v_groups.ToIYD = v_groups._ToIYD.sum(axis=0)
                v_groups.ToIY = v_groups._ToIY.sum(axis=0)
                v_groups.ToIA = v_groups._ToIA.sum(axis=0)

            if calendar[t] == self.instance.omicron_start:
                # Move almost half of the people from recovered to susceptible:
                self.immune_escape(self.epi.immune_escape_rate, t)

            if t >= self.vaccine.vaccine_start_time:

                S_before = np.zeros((5, 2))

                for idx, v_groups in enumerate(self.vaccine_groups):
                    S_before += v_groups.S

                for idx, v_groups in enumerate(self.vaccine_groups):

                    out_sum = np.zeros(self.age_risk_matrix_shape)
                    S_out = np.zeros((A*L, 1))
                    N_out = np.zeros((A*L, 1))

                    for vaccine_type in v_groups.v_out:
                        event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                        if event is not None:

                            S_out = np.reshape(self.vaccine.vaccine_allocation[vaccine_type][event]["assignment"], (A*L, 1))
                            if calendar[t] >= self.instance.omicron_start:
                                if v_groups.v_name == "v_1" or v_groups.v_name == "v_2":
                                    S_out = self.epi.immune_escape_rate * np.reshape(self.vaccine.vaccine_allocation[vaccine_type][event]["assignment"], (A*L, 1))

                            N_out = self.vaccine.get_num_eligible(N, A * L, v_groups.v_name, v_groups.v_in, v_groups.v_out, calendar[t])

                            ratio_S_N = np.array([0 if N_out[i] == 0 else float(S_out[i]/N_out[i]) for i in range(len(N_out))]).reshape(self.age_risk_matrix_shape)

                            if self.types == 'int':
                                out_sum += np.round(ratio_S_N*v_groups._S[self.step_size])
                            else:
                                out_sum += ratio_S_N*v_groups._S[self.step_size]

                    in_sum = np.zeros(self.age_risk_matrix_shape)
                    S_in = np.zeros((A*L, 1))
                    N_in = np.zeros((A*L, 1))

                    for vaccine_type in v_groups.v_in:

                        for v_g in self.vaccine_groups:
                            if v_g.v_name == self.vaccine.vaccine_allocation[vaccine_type][0]["from"]:
                                v_temp = v_g

                        event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                        if event is not None:
                            S_in = np.reshape(self.vaccine.vaccine_allocation[vaccine_type][event]["assignment"], (A*L, 1))

                            if calendar[t] >= self.instance.omicron_start:
                                if (v_groups.v_name == "v_3" and v_temp.v_name == "v_2") or (v_groups.v_name == "v_2" and v_temp.v_name == "v_1"):
                                    S_in = self.epi.immune_escape_rate * np.reshape(self.vaccine.vaccine_allocation[vaccine_type][event]["assignment"], (A*L, 1))

                            N_in = self.vaccine.get_num_eligible(N, A * L, v_temp.v_name, v_temp.v_in, v_temp.v_out, calendar[t])
                            ratio_S_N = np.array([0 if N_in[i] == 0 else float(S_in[i]/N_in[i]) for i in range(len(N_in))]).reshape(self.age_risk_matrix_shape)

                            if self.types == 'int':
                                in_sum += np.round(ratio_S_N*v_temp._S[self.step_size])
                            else:
                                in_sum += ratio_S_N*v_temp._S[self.step_size]

                    if self.types == "float":
                        v_groups.S = v_groups.S + (np.array(in_sum - out_sum))
                    else:
                        out_sum = np.round(out_sum)
                        in_sum = np.round(in_sum)
                        v_groups.S = v_groups.S + np.round(np.array(in_sum - out_sum))

                    S_after = np.zeros((5, 2))

                for idx, v_groups in enumerate(self.vaccine_groups):
                    S_after += v_groups.S

                imbalance = np.abs(np.sum(S_before - S_after, axis = (0,1)))

                assert (imbalance < 1E-2).any(), f'fPop inbalance in vaccine flow in between compartment S {imbalance} at time {calendar[t]}, {t}'

            for idx, v_groups in enumerate(self.vaccine_groups):
                v_groups._S = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._E = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._IA = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._IY = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._PA = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._PY = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._IH = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._ICU = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._R = np.zeros((self.step_size + 1, A, L), dtype=self.types)
                v_groups._D = np.zeros((self.step_size + 1, A, L), dtype=self.types)

                v_groups._IYIH = np.zeros((self.step_size, A, L))
                v_groups._IYICU = np.zeros((self.step_size, A, L))
                v_groups._IHICU = np.zeros((self.step_size, A, L))
                v_groups._ToICU = np.zeros((self.step_size, A, L))
                v_groups._ToIHT = np.zeros((self.step_size, A, L))
                v_groups._ToICUD = np.zeros((self.step_size, A, L))
                v_groups._ToIYD = np.zeros((self.step_size, A, L))
                v_groups._ToIA = np.zeros((self.step_size, A, L))
                v_groups._ToIY = np.zeros((self.step_size, A, L))

                v_groups._S[0] = v_groups.S
                v_groups._E[0] = v_groups.E
                v_groups._IA[0] = v_groups.IA
                v_groups._IY[0] = v_groups.IY
                v_groups._PA[0] = v_groups.PA
                v_groups._PY[0] = v_groups.PY
                v_groups._R[0] = v_groups.R
                v_groups._D[0] = v_groups.D

                v_groups._IH[0] = v_groups.IH
                v_groups._ICU[0] = v_groups.ICU

    def rv_gen(self, n, p, round_opt=1):

        if self.rng_generator is None:
            return n * p
        else:
            if round_opt:
                nInt = np.round(n)
                return self.rng_generator.binomial(nInt.astype(int), p)
            else:
                return self.rng_generator.binomial(n, p)


def system_simulation(mp_sim_input):
    '''
        Simulation function that gets mapped when running simulations in parallel.
        Args:
            mp_sim_input (tuple):
                instance, policy, cost_func, interventions, kwargs (as a dict)
        Returns:
            out_sim (dict): output of the simulation
            policy_cost (float): evaluation of cost_func
            policy (object): the policy used in the simulation
            seed (int): seed used in the simulation
            kwargs (dict): additional parameters used
    '''

    instance, policy, cost_func, interventions, thrs, seed, kwargs = mp_sim_input
    out_sim = simulate_vaccine(instance, policy, interventions, thrs, seed, **kwargs)
    policy_cost, cost_info = cost_func(instance, policy, out_sim, **kwargs)
    kwargs_new = kwargs.copy()
    kwargs_new["cost_info"] = cost_info
    #breakpoint()
    return out_sim, policy_cost, policy, thrs, seed, kwargs_new

WEEKDAY = 1
WEEKEND = 2
HOLIDAY = 3
LONG_HOLIDAY = 4


class SimCalendar():
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
        
        currentTemp = get_next_month(self.start)
        while currentTemp <= self.calendar[-1]:
            month_starts.append(self.calendar_ix[currentTemp])
            currentTemp = get_next_month(currentTemp)
        
        return month_starts
    
    def __len__(self):
        return len(self.calendar)


def get_next_month(dateG):
    if dateG.month == 12:
        startMonth = 1
        startYear = dateG.year + 1
    else:
        startMonth = dateG.month + 1
        startYear = dateG.year
    return dt.datetime(startYear, startMonth, 1)


def discrete_approx(rate, timestep):
    #return (1 - (1 - rate)**(1 / timestep))
    return (1 - np.exp(-rate/timestep))