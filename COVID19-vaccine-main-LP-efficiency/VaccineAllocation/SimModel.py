import numpy as np
from SimObjects import VaccineGroup
import copy

class SimReplication:

    def __init__(self, instance, vaccine, policy, rng_seed):

        self.instance = instance
        self.step_size = instance.config["step_size"]
        self.t_historical_data_end = len(self.instance.real_hosp)
        A = self.instance.A
        L = self.instance.L

        self.rng_seed = rng_seed

        self.vaccine = vaccine
        self.policy = policy

        self.define_epi()
        self.define_groups()

        self.ICU_history = [np.zeros((A, L))]
        self.IH_history = [np.zeros((A, L))]

        self.ToIHT_history = []
        self.ToIY_history = []

        # The next t that is simulated
        # This instance has simulated up to but not including time next_t
        self.next_t = 0

        self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")
        self.tracking_vars = ("IYIH", "IYICU", "IHICU", "ToICU", "ToIHT", "ToICUD", "ToIYD", "ToIA", "ToIY")

    def compute_ICU_violation(self):

        return np.any(np.array(self.ICU_history).sum(axis=(1, 2))[self.t_historical_data_end:] > self.instance.icu)

    def compute_rsq(self):

        f_benchmark = self.instance.real_hosp

        IH_sim = np.array(self.ICU_history) + np.array(self.IH_history)
        IH_sim = IH_sim.sum(axis=(2, 1))
        IH_sim = IH_sim[:self.t_historical_data_end]

        if self.next_t < self.t_historical_data_end:
            IH_sim = IH_sim[:self.next_t]
            f_benchmark = f_benchmark[:self.next_t]

        rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark)) ** 2)) / sum(
            (np.array(f_benchmark) - np.mean(np.array(f_benchmark))) ** 2)

        return rsq

    def define_epi(self):

        self.rng = np.random.RandomState(self.rng_seed) if self.rng_seed >= 0 else None

        epi_rand = copy.deepcopy(self.instance.base_epi)
        epi_rand.sample_random_params(self.rng)
        epi_rand.setup_base_params()

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

        for v_groups in self.vaccine_groups:
            moving_people = v_groups._R[self.step_size] * immune_escape_rate

            v_groups.R -= moving_people
            self.vaccine_groups[3].S += moving_people

            if v_groups.v_name == 'v_1' or v_groups.v_name == 'v_2':
                moving_people = v_groups._S[self.step_size] * immune_escape_rate

                v_groups.S -= moving_people
                self.vaccine_groups[3].S += moving_people

    def simulate_time_period(self, time_start, time_end):

        for t in range(time_start, time_end):

            self.next_t += 1

            self.simulate_t(t)

            A = self.instance.A
            L = self.instance.L

            for attribute in self.state_vars + self.tracking_vars:
                setattr(self, attribute, np.zeros((A, L)))

            for attribute in self.state_vars + self.tracking_vars:
                sum_across_vaccine_groups = 0
                for v_group in self.vaccine_groups:
                    sum_across_vaccine_groups += getattr(v_group, attribute)
                setattr(self, attribute, sum_across_vaccine_groups)

            self.ICU_history.append(self.ICU)
            self.IH_history.append(self.IH)

            self.ToIHT_history.append(self.ToIHT)
            self.ToIY_history.append(self.ToIY)

            total_imbalance = np.sum(self.S + self.E + self.IA + self.IY + self.R + self.D + self.PA + self.PY + self.IH + self.ICU) - np.sum(self.instance.N)

            assert np.abs(total_imbalance) < 1E-2, f'fPop unbalanced {total_imbalance} at time {self.instance.cal.calendar[t]}, {t}'

    def simulate_t(self, t_date):

        A = self.instance.A
        L = self.instance.L
        N = self.instance.N

        calendar = self.instance.cal.calendar

        t = t_date

        epi = copy.deepcopy(self.epi_rand)

        if t < len(self.instance.real_hosp):
            phi_t = epi.effective_phi(self.instance.cal.schools_closed[t],
                                   self.instance.cal.fixed_cocooning[t],
                                   self.instance.cal.fixed_transmission_reduction[t],
                                   N / N.sum(),
                                   self.instance.cal.get_day_type(t))
        else:
            #  ToIHT, IH, ToIY, ICU
            self.policy(t, self.ToIHT_history, self.IH_history, self.ToIY_history, self.ICU_history)
            current_tier = self.policy.tier_history[t]
            phi_t = epi.effective_phi(self.policy.tiers[current_tier]["school_closure"],
                                   self.policy.tiers[current_tier]["cocooning"],
                                   self.policy.tiers[current_tier]["transmission_reduction"],
                                   N / N.sum(),
                                   self.instance.cal.get_day_type(t))

        if calendar[t] >= self.instance.delta_start:
            days_since_delta_start = (calendar[t] - self.instance.delta_start).days
            for v_groups in self.vaccine_groups:
                v_groups.delta_update(self.instance.delta_prev[days_since_delta_start])
            epi.delta_update_param(self.instance.delta_prev[days_since_delta_start])

        #Update epi parameters for omicron:
        if calendar[t] >= self.instance.omicron_start:
            days_since_omicron_start = (calendar[t] - self.instance.omicron_start).days
            epi.omicron_update_param(self.instance.omicron_prev[days_since_omicron_start])
            for v_groups in self.vaccine_groups:
                v_groups.omicron_update(self.instance.omicron_prev[days_since_omicron_start])

        # Assume an imaginary new variant in May, 2022:
        if epi.new_variant == True:
            days_since_variant_start = (calendar[t] - self.instance.variant_start).days
            if calendar[t] >= self.instance.variant_start:
                epi.variant_update_param(self.instance.variant_prev[days_since_variant_start])

        if self.instance.otherInfo == {}:
            if t > kwargs["rd_start"] and t <= kwargs["rd_end"]:
                epi.update_icu_params(kwargs["rd_rate"])
        else:
            epi.update_icu_all(t,self.instance.otherInfo)

        # sigma_E, pi, gamma_IY, alpha4
        # gamma_IA, rho_A, rho_Y,
        # Eta, rIH, nu, mu,
        # gamma_IH, nu_ICU, mu_ICU,
        # gamma_ICU,
        # immune_evasion
        # omega_PY, omega_PA, omega_IA, omega_IY,
        # beta, tau

        rate_E = self.discrete_approx(epi.sigma_E, self.step_size)
        rate_IYR = self.discrete_approx(np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY * (1 - epi.alpha4) for l in range(L)] for a in range(A)]), self.step_size)
        rate_IYD = self.discrete_approx(np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY * epi.alpha4 for l in range(L)] for a in range(A)]), self.step_size)
        rate_IAR = self.discrete_approx(np.tile(epi.gamma_IA, (L, A)).transpose(), self.step_size)
        rate_PAIA = self.discrete_approx(np.tile(epi.rho_A, (L, A)).transpose(), self.step_size)
        rate_PYIY = self.discrete_approx(np.tile(epi.rho_Y, (L, A)).transpose(), self.step_size)
        rate_IYH = self.discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] * epi.rIH for l in range(L)] for a in range(A)]), self.step_size)
        rate_IYICU = self.discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] * (1 - epi.rIH) for l in range(L)] for a in range(A)]), self.step_size)
        rate_IHICU = self.discrete_approx(epi.nu*epi.mu,self.step_size)
        rate_IHR = self.discrete_approx((1 - epi.nu)*epi.gamma_IH, self.step_size)
        rate_ICUD = self.discrete_approx(epi.nu_ICU*epi.mu_ICU, self.step_size)
        rate_ICUR = self.discrete_approx((1 - epi.nu_ICU)*epi.gamma_ICU, self.step_size)

        if t >= 711: #date corresponding to 02/07/2022
            rate_immune = self.discrete_approx(epi.immune_evasion, self.step_size)

        for _t in range(self.step_size):
            # Dynamics for dS

            for v_groups in self.vaccine_groups:

                dSprob_sum = np.zeros((5,2))

                for v_groups_temp in self.vaccine_groups:

                    # Vectorized version for efficiency. For-loop version commented below
                    temp1 = np.matmul(np.diag(epi.omega_PY), v_groups_temp._PY[_t, :, :]) + \
                        np.matmul(np.diag(epi.omega_PA), v_groups_temp._PA[_t, :, :]) + \
                            epi.omega_IA * v_groups_temp._IA[_t, :, :] + \
                                epi.omega_IY * v_groups_temp._IY[_t, :, :]

                    temp2 = np.sum(N, axis=1)[np.newaxis].T
                    temp3 = np.divide(np.multiply(epi.beta * phi_t / self.step_size, temp1), temp2)

                    dSprob = np.sum(temp3, axis=(2, 3))
                    dSprob_sum = dSprob_sum + dSprob

                if t >= 711 and v_groups.v_name == 'v_2':#date corresponding to 02/07/2022
                    _dS = self.rv_gen(v_groups._S[_t], rate_immune + (1 - v_groups.v_beta_reduct)*dSprob_sum)
                    # Dynamics for E
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
                EPY = self.rv_gen(E_out, epi.tau * ( 1 - v_groups.v_tau_reduct))
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

        for v_groups in self.vaccine_groups:
            # End of the daily disctretization
            for attribute in self.state_vars:
                setattr(v_groups, attribute, getattr(v_groups, "_" + attribute)[self.step_size].copy())

            for attribute in self.tracking_vars:
                setattr(v_groups, attribute, getattr(v_groups, "_" + attribute).sum(axis=0))

            # v_groups.S = v_groups._S[self.step_size].copy()
            # v_groups.E = v_groups._E[self.step_size].copy()
            # v_groups.IA = v_groups._IA[self.step_size].copy()
            # v_groups.IY = v_groups._IY[self.step_size].copy()
            # v_groups.PA = v_groups._PA[self.step_size].copy()
            # v_groups.PY = v_groups._PY[self.step_size].copy()
            # v_groups.R = v_groups._R[self.step_size].copy()
            # v_groups.D = v_groups._D[self.step_size].copy()
            # v_groups.IH = v_groups._IH[self.step_size].copy()
            # v_groups.ICU = v_groups._ICU[self.step_size].copy()
            #
            # v_groups.IYIH = v_groups._IYIH.sum(axis=0)
            # v_groups.IYICU = v_groups._IYICU.sum(axis=0)
            # v_groups.IHICU = v_groups._IHICU.sum(axis=0)
            # v_groups.ToICU = v_groups._ToICU.sum(axis=0)
            # v_groups.ToIHT = v_groups._ToIHT.sum(axis=0)
            # v_groups.ToICUD = v_groups._ToICUD.sum(axis=0)
            # v_groups.ToIYD = v_groups._ToIYD.sum(axis=0)
            # v_groups.ToIY = v_groups._ToIY.sum(axis=0)
            # v_groups.ToIA = v_groups._ToIA.sum(axis=0)

        if calendar[t] == self.instance.omicron_start:
            # Move almost half of the people from recovered to susceptible:
            self.immune_escape(epi.immune_escape_rate, t)

        if t >= self.vaccine.vaccine_start_time:

            S_before = np.zeros((5, 2))

            for v_groups in self.vaccine_groups:
                S_before += v_groups.S

            for v_groups in self.vaccine_groups:

                out_sum = np.zeros((A, L))
                S_out = np.zeros((A*L, 1))
                N_out = np.zeros((A*L, 1))

                for vaccine_type in v_groups.v_out:
                    event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                    if event is not None:

                        S_out = np.reshape(self.vaccine.vaccine_allocation[vaccine_type][event]["assignment"], (A*L, 1))
                        if calendar[t] >= self.instance.omicron_start:
                            if v_groups.v_name == "v_1" or v_groups.v_name == "v_2":
                                S_out = epi.immune_escape_rate * np.reshape(self.vaccine.vaccine_allocation[vaccine_type][event]["assignment"], (A*L, 1))

                        N_out = self.vaccine.get_num_eligible(N, A * L, v_groups.v_name, v_groups.v_in, v_groups.v_out, calendar[t])

                        ratio_S_N = np.array([0 if N_out[i] == 0 else float(S_out[i]/N_out[i]) for i in range(len(N_out))]).reshape((A, L))

                        out_sum += ratio_S_N*v_groups._S[self.step_size]

                in_sum = np.zeros((A, L))
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
                                S_in = epi.immune_escape_rate * np.reshape(self.vaccine.vaccine_allocation[vaccine_type][event]["assignment"], (A*L, 1))

                        N_in = self.vaccine.get_num_eligible(N, A * L, v_temp.v_name, v_temp.v_in, v_temp.v_out, calendar[t])
                        ratio_S_N = np.array([0 if N_in[i] == 0 else float(S_in[i]/N_in[i]) for i in range(len(N_in))]).reshape((A, L))

                        in_sum += ratio_S_N*v_temp._S[self.step_size]

                v_groups.S = v_groups.S + (np.array(in_sum - out_sum))

                S_after = np.zeros((5, 2))

            for v_groups in self.vaccine_groups:
                S_after += v_groups.S

            imbalance = np.abs(np.sum(S_before - S_after, axis = (0,1)))

            assert (imbalance < 1E-2).any(), f'fPop inbalance in vaccine flow in between compartment S {imbalance} at time {calendar[t]}, {t}'

        for v_groups in self.vaccine_groups:

            for attribute in self.state_vars:
                setattr(v_groups, "_" + attribute, np.zeros((self.step_size + 1, A, L)))
                vars(v_groups)["_" + attribute][0] = vars(v_groups)[attribute]

            for attribute in self.tracking_vars:
                setattr(v_groups, "_" + attribute, np.zeros((self.step_size, A, L)))

            # v_groups._S = np.zeros((self.step_size + 1, A, L))
            # v_groups._E = np.zeros((self.step_size + 1, A, L))
            # v_groups._IA = np.zeros((self.step_size + 1, A, L))
            # v_groups._IY = np.zeros((self.step_size + 1, A, L))
            # v_groups._PA = np.zeros((self.step_size + 1, A, L))
            # v_groups._PY = np.zeros((self.step_size + 1, A, L))
            # v_groups._IH = np.zeros((self.step_size + 1, A, L))
            # v_groups._ICU = np.zeros((self.step_size + 1, A, L))
            # v_groups._R = np.zeros((self.step_size + 1, A, L))
            # v_groups._D = np.zeros((self.step_size + 1, A, L))

            # v_groups._IYIH = np.zeros((self.step_size, A, L))
            # v_groups._IYICU = np.zeros((self.step_size, A, L))
            # v_groups._IHICU = np.zeros((self.step_size, A, L))
            # v_groups._ToICU = np.zeros((self.step_size, A, L))
            # v_groups._ToIHT = np.zeros((self.step_size, A, L))
            # v_groups._ToICUD = np.zeros((self.step_size, A, L))
            # v_groups._ToIYD = np.zeros((self.step_size, A, L))
            # v_groups._ToIA = np.zeros((self.step_size, A, L))
            # v_groups._ToIY = np.zeros((self.step_size, A, L))

            # v_groups._S[0] = v_groups.S
            # v_groups._E[0] = v_groups.E
            # v_groups._IA[0] = v_groups.IA
            # v_groups._IY[0] = v_groups.IY
            # v_groups._PA[0] = v_groups.PA
            # v_groups._PY[0] = v_groups.PY
            # v_groups._R[0] = v_groups.R
            # v_groups._D[0] = v_groups.D
            # v_groups._IH[0] = v_groups.IH
            # v_groups._ICU[0] = v_groups.ICU

    def rv_gen(self, n, p, round_opt=1):

        if self.rng is None:
            return n * p
        else:
            if round_opt:
                nInt = np.round(n)
                return self.rng.binomial(nInt.astype(int), p)
            else:
                return self.rng.binomial(n, p)

    def discrete_approx(self, rate, timestep):
        return (1 - np.exp(-rate / timestep))

