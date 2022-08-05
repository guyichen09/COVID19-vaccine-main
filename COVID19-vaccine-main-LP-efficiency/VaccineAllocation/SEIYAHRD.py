'''
This module runs a simulation of the SEIYARDH model for a single city,
considering different age groups and seven compartments. This model is
based of Zhanwei's Du model, Meyer's Group.

This module also contains functions to run the simulations in parallel
and a class to properly define a calendar (SimCalendar).
'''

import datetime as dt
import numpy as np
from utils import timeit, roundup
from vaccine_policies import VaccineAllocationPolicy
from trigger_policies import MultiTierPolicy,MultiTierPolicy_ACS, CDCTierPolicy
from VaccineAllocation import config
import copy

def immune_escape(immune_escape_rate, t, types, v_policy, step_size):
    '''
        This function move recovered and vaccinated individuals to waning 
        efficacy susceptible compartment after omicron become the prevelant 
        virus type.
    '''
    for idx, v_groups in enumerate(v_policy._vaccine_groups):
        if types == 'int':
            moving_people = (v_groups._R[step_size] *  immune_escape_rate).astype(int)
        else:
            moving_people = v_groups._R[step_size] *  immune_escape_rate
                
        v_groups.R[t+1] -=  moving_people
        v_policy._vaccine_groups[3].S[t+1] +=  moving_people
            
        if v_groups.v_name == 'v_1' or v_groups.v_name == 'v_2':
            if types == 'int':
                moving_people = (v_groups._S[step_size] *  immune_escape_rate).astype(int)
            else:
                moving_people = v_groups._S[step_size] *  immune_escape_rate
            
            v_groups.S[t+1] -=  moving_people
            v_policy._vaccine_groups[3].S[t+1] +=  moving_people
 

def simulate_vaccine(instance, v_policy, seed=-1, **kwargs):
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

    Args:
        epi (EpiParams): instance of the parameterization
        T(int): number of time periods
        A(int): number of age groups
        L(int): number of risk groups
        F(int): frequency of the  interventions
        interventions (list of Intervention): list of inteventions
        N(ndarray): total population on each age group
        I0 (ndarray): initial infected people on each age group
        z (ndarray): interventions for each day
        policy (func): callabe to get intervention at time t
        calendar (SimCalendar): calendar of the simulation
        seed (int): seed for random generation. Defaults is -1 which runs
            the simulation in a deterministic fashion.
        kwargs (dict): additional parameters that are passed to the policy function
    '''
    kwargs["acs_triggered"] = False

    if "acs_type" in kwargs.keys():
        if kwargs["acs_type"] == "IHT":
            kwargs["_capacity"] = [instance.hosp_beds] * instance.T
        elif kwargs["acs_type"] == "ICU":
            kwargs["_capacity"] = [instance.icu] * instance.T 
    else:
        kwargs["_capacity"] = [instance.hosp_beds] * instance.T
 
    T, A, L = instance.T, instance.A, instance.L
    N = instance.N

    # Compartments
    if config['det_history']:
        types = 'float'
    else:
        types = 'int' if seed >= 0 else 'float'

    # Random stream for stochastic simulations
    if config["det_param"]:
        rnd_epi = None
    else:
        rnd_epi = np.random.RandomState(seed) if seed >= 0 else None
        
    epi = instance.epi
    epi_orig = copy.deepcopy(epi)
    epi_rand = copy.deepcopy(epi)

    epi_rand.update_rnd_stream(rnd_epi)
    epi_orig.update_rnd_stream(None)

    epi_rand.update_hosp_duration()
    epi_orig.update_hosp_duration()
    
    # T0 = v_policy._vaccines.vaccine_start_time[0]
    IH = np.zeros((T, A, L), dtype=types)
    ICU = np.zeros((T, A, L), dtype=types)
    ToIHT = np.zeros((T, A, L), dtype=types)
    ToIY = np.zeros((T, A, L), dtype=types)
       
    for t in range(T - 1):
        kwargs["acs_criStat"] = eval(kwargs["acs_policy_field"])[:t]
        kwargs["t_start"] = len(instance.real_hosp)

        v_policy = simulate_t(instance, v_policy, t, epi_rand, epi_orig, rnd_epi, seed, **kwargs)
        S = np.zeros((T, A, L), dtype=types)
        E = np.zeros((T, A, L), dtype=types)
        IA = np.zeros((T, A, L), dtype=types)
        IY = np.zeros((T, A, L), dtype=types)
        PA = np.zeros((T, A, L), dtype=types)
        PY = np.zeros((T, A, L), dtype=types)
        IH = np.zeros((T, A, L), dtype=types)
        ICU = np.zeros((T, A, L), dtype=types)
        R = np.zeros((T, A, L), dtype=types)
        D = np.zeros((T, A, L), dtype=types)
    
        # Additional tracking variables (for triggers)
        IYIH = np.zeros((T - 1, A, L))
        IYICU = np.zeros((T - 1, A, L))
        IHICU = np.zeros((T - 1, A, L))
        ToICU = np.zeros((T - 1, A, L))
        ToIHT = np.zeros((T - 1, A, L))
        ToICUD = np.zeros((T - 1, A, L))
        ToIYD = np.zeros((T - 1, A, L))
        ToIA = np.zeros((T - 1, A, L))
        ToIY = np.zeros((T - 1, A, L))

        # Track hospitalization variables according to vaccine status:
        # ICU_vac = np.zeros((T, A, L))
        # ToIHT_vac = np.zeros((T - 1, A, L))
        # IH_vac = np.zeros((T, A, L))

        # ICU_unvac = np.zeros((T, A, L))
        # ToIHT_unvac = np.zeros((T - 1, A, L))
        # IH_unvac = np.zeros((T, A, L))
        
        for v_group in v_policy._vaccine_groups:
            # Update compartments
            # if v_group.v_name == 'v_0':
            #    S0=v_policy._vaccine_groups[0].S
            # if v_group.v_name == 'v_1':
            #    S1=v_policy._vaccine_groups[1].S
            # if v_group.v_name == 'v_2':
            #    S2=v_policy._vaccine_groups[2].S
            # if v_group.v_name == 'v_3':
            #    S3=v_policy._vaccine_groups[3].S
                
            S += v_group.S
            E += v_group.E
            IA += v_group.IA
            IY += v_group.IY
            PA += v_group.PA
            PY += v_group.PY
            IH += v_group.IH
            ICU += v_group.ICU
            R += v_group.R
            D += v_group.D
            
            # Update daily values
            IYIH += v_group.IYIH
            IYICU += v_group.IYICU
            IHICU += v_group.IHICU
            ToICU += v_group.ToICU
            ToIHT += v_group.ToIHT
            ToIYD += v_group.ToIYD
            ToICUD += v_group.ToICUD
            ToIA += v_group.ToIA
            ToIY += v_group.ToIY
            dS = S[1:, :] - S[:-1, :]
            
            # if v_group.v_name == 'v_0':
            #    ICU_unvac = v_group.ICU
            #    ToIHT_unvac = v_group.ToIHT
            #    IH_unvac = v_group.IH
       
            # if v_group.v_name == 'v_2':
            #    ICU_vac = v_group.ICU
            #    ToIHT_vac= v_group.ToIHT
            #    IH_vac = v_group.IH
                
        total_imbalance = np.sum(S[t] + E[t] + IA[t] + IY[t] + IH[t] + R[t] + D[t] + PA[t] + PY[t] + ICU[t]) - np.sum(N)

        assert np.abs(total_imbalance) < 1E-2, f'fPop unbalanced {total_imbalance} at time {instance.cal.calendar[t]}, {t}'
    
    # moving_avg_len = config['moving_avg_len']
    # ToIHT_temp = np.sum(ToIHT, axis=(1, 2))[:T]
    # ToIY_temp = np.sum(ToIY, axis=(1, 2))[:T]
    # IHT_temp = np.sum(IH, axis=(1, 2))[:T] +np.sum(ICU, axis=(1, 2))[:T]
    # ToIHT_moving = [ToIHT_temp[i: min(i + moving_avg_len, T)].mean() for i in range(T-moving_avg_len)]
    # ToIHT_total = [ToIHT_temp[i: min(i + moving_avg_len, T)].sum()* 100000/np.sum(N, axis=(0,1))  for i in range(T-moving_avg_len)]
    # ToIY_moving = [ToIY_temp[i: min(i + moving_avg_len, T)].sum()* 100000/np.sum(N, axis=(0,1)) for i in range(T-moving_avg_len)]
    # IHT_moving = [IHT_temp[i: min(i + moving_avg_len, T)].mean()/instance.hosp_beds for i in range(T-moving_avg_len)]
    # ICU_ratio = np.sum(ICU, axis=(1, 2))[:T]/(np.sum(ICU, axis=(1, 2))[:T] + np.sum(IH, axis=(1, 2))[:T])

    output = {
        'S': S,
        # 'S0': S0,
        # 'S1': S1,
        # 'S2': S2,
        # 'S3': S3,
        'E': E,
        'PA': PA,
        'PI': PY,
        'IA': IA,
        'IY': IY,
        'IH': IH,
        # 'IHT_moving': IHT_moving,
        'R': R,
        'D': D,
        'ICU': ICU,
        'dS': dS,
        'IYIH': IYIH,
        'IYICU': IYICU,
        'IHICU': IHICU,
        # 'z': policy.get_interventions_history().copy() if (isinstance(policy, MultiTierPolicy))or(isinstance(policy, MultiTierPolicy_ACS)) \
        #    or (isinstance(policy, CDCTierPolicy))else None,
        #'tier_history': policy.get_tier_history().copy() if (isinstance(policy, MultiTierPolicy))or(isinstance(policy, MultiTierPolicy_ACS)) \
        #    or (isinstance(policy, CDCTierPolicy))else None,
        # 'surge_history': policy.get_surge_history().copy() if (isinstance(policy, CDCTierPolicy))else None,
        'seed': seed,
        'acs_triggered': kwargs["acs_triggered"],
        'capacity': kwargs["_capacity"],
        'ToICU': ToICU,
        'ToIHT': ToIHT,
        # 'ToIHT_moving': ToIHT_moving,
        # 'ToIHT_total': ToIHT_total,
        # 'ToIY_moving': ToIY_moving,
        'ToICUD': ToICUD,
        'ToIYD': ToIYD,
        'ToIA': ToIA,
        'ToIY': ToIY,
        'IHT': ICU+IH,
        # 'ICU_vac': ICU_vac,
        # 'ToIHT_vac': ToIHT_vac,
        # 'IH_vac': IH_vac,
        # 'ICU_unvac': ICU_unvac,
        # 'ToIHT_unvac': ToIHT_unvac,
        # 'IH_unvac': IH_unvac,
        # 'ICU_ratio': ICU_ratio
        }

    return output

def simulate_t(instance, v_policy, t_date, epi_rand, epi_orig, rnd_stream, seed=-1,  **kwargs):

        A, L = instance.A, instance.L
        N = instance.N
        calendar = instance.cal

        # Rates of change
        step_size = config['step_size']
        kwargs["acs_triggered"] = False
        if "acs_type" in kwargs.keys():
            if kwargs["acs_type"] == "IHT":
                kwargs["_capacity"] = [instance.hosp_beds] * instance.T
            elif kwargs["acs_type"] == "ICU":
                kwargs["_capacity"] = [instance.icu] * instance.T 
        else:
            kwargs["_capacity"] = [instance.hosp_beds] * instance.T
 
        # Compartments
        if config['det_history']:
            types = 'float'
        else:
            types = 'int' if seed >= 0 else 'float'
        
        for t_idx in range(1):
            t = t_date
            # Get dynamic intervention and corresponding contact matrix
            #k_t, kwargs = policy(t, criStat=eval(kwargs["policy_field"])[:t], IH=IH[:t], **kwargs)

            # LP uncommented
            # k_t =  policy._intervention_history[t]
            # phi_t = interventions[k_t].phi(calendar.get_day_type(t))

            # school, cocooning, social_distance, demographics, day_type

            if t < len(instance.real_hosp):
                phi_t = instance.epi.effective_phi(instance.cal.schools_closed[t],
                                       instance.cal.fixed_cocooning[t],
                                       instance.cal.fixed_transmission_reduction[t],
                                       N / N.sum(),
                                       calendar.get_day_type(t))
            # NEED TO DELETE THIS THIS IS JUST TO MAKE IT WORK FOR PAST HISTORICAL PERIOD
            else:
                phi_t = instance.epi.effective_phi(instance.cal.schools_closed[len(instance.real_hosp) - 1],
                                       instance.cal.fixed_cocooning[len(instance.real_hosp) - 1],
                                       instance.cal.fixed_transmission_reduction[len(instance.real_hosp) - 1],
                                       N / N.sum(),
                                       calendar.get_day_type(len(instance.real_hosp)))

            # if the current time is within the history
            if config['det_history'] and t < len(instance.real_hosp):
                epi = copy.deepcopy(epi_orig)
            else:
                epi = copy.deepcopy(epi_rand)
                
            #Update epi parameters for delta prevalence:         
            T_delta = np.where(np.array(v_policy._instance.cal.calendar) == instance.delta_start)[0]
            if len(T_delta) > 0:
                T_delta = T_delta[0]
                if t >= T_delta:
                    for v_groups in v_policy._vaccine_groups:
                        v_groups.delta_update(instance.delta_prev[t - T_delta])
                    epi.delta_update_param(instance.delta_prev[t - T_delta])

            #Update epi parameters for omicron:     
            T_omicron = np.where(np.array(v_policy._instance.cal.calendar) == instance.omicron_start)[0]
            if len(T_omicron) > 0:
                T_omicron = T_omicron[0]
                if t >= T_omicron:
                    epi.omicron_update_param(instance.omicron_prev[t - T_omicron])
                    for v_groups in v_policy._vaccine_groups:
                        v_groups.omicron_update(instance.delta_prev[t - T_delta])
            
            # Assume an imaginary new variant in May, 2022:
            if epi.new_variant == True:
                T_variant = np.where(np.array(v_policy._instance.cal.calendar) == instance.variant_start)[0]
                if len(T_variant) > 0:
                    T_variant = T_variant[0]
                    if t >= T_variant:
                        epi.variant_update_param(instance.variant_prev[t - T_variant])
            
            if instance.otherInfo == {}:
                if t > kwargs["rd_start"] and t <= kwargs["rd_end"]:
                    epi.update_icu_params(kwargs["rd_rate"])
            else:
                epi.update_icu_all(t,instance.otherInfo)

            rate_E = discrete_approx(epi.sigma_E, step_size)
            rate_IYR = discrete_approx(np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY * (1 - epi.alpha4) for l in range(L)] for a in range(A)]), step_size)
            rate_IYD = discrete_approx(np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY * epi.alpha4 for l in range(L)] for a in range(A)]), step_size)
            rate_IAR = discrete_approx(np.tile(epi.gamma_IA, (L, A)).transpose(), step_size)
            rate_PAIA = discrete_approx(np.tile(epi.rho_A, (L, A)).transpose(), step_size)
            rate_PYIY = discrete_approx(np.tile(epi.rho_Y, (L, A)).transpose(), step_size)
            rate_IYH = discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] * epi.rIH for l in range(L)] for a in range(A)]), step_size)
            rate_IYICU = discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] * (1 - epi.rIH) for l in range(L)] for a in range(A)]), step_size)
            rate_IHICU = discrete_approx(epi.nu*epi.mu,step_size)
            rate_IHR = discrete_approx((1 - epi.nu)*epi.gamma_IH, step_size)
            rate_ICUD = discrete_approx(epi.nu_ICU*epi.mu_ICU, step_size)
            rate_ICUR = discrete_approx((1 - epi.nu_ICU)*epi.gamma_ICU, step_size)
            
            if t >= 711: #date corresponding to 02/07/2022
                rate_immune = discrete_approx(epi.immune_evasion, step_size) 

            for _t in range(step_size):
                # Dynamics for dS

                for v_groups in v_policy._vaccine_groups:
                    
                    dSprob_sum = np.zeros((5,2))
                    
                    for v_groups_temp in v_policy._vaccine_groups:

                        # Vectorized version for efficiency. For-loop version commented below
                        temp1 = np.matmul(np.diag(epi.omega_PY), v_groups_temp._PY[_t, :, :]) + \
                            np.matmul(np.diag(epi.omega_PA), v_groups_temp._PA[_t, :, :]) + \
                                epi.omega_IA * v_groups_temp._IA[_t, :, :] + \
                                    epi.omega_IY * v_groups_temp._IY[_t, :, :]
                                    
                        temp2 = np.sum(N, axis=1)[np.newaxis].T
                        temp3 = np.divide(np.multiply(epi.beta * phi_t / step_size, temp1), temp2)
                        
                        dSprob = np.sum(temp3, axis=(2, 3))
                        dSprob_sum = dSprob_sum + dSprob
                        
                    if t >= 711 and v_groups.v_name == 'v_2':#date corresponding to 02/07/2022
                        _dS = rv_gen(rnd_stream, v_groups._S[_t], rate_immune + (1 - v_groups.v_beta_reduct)*dSprob_sum)  
                            # Dynamics for E
                        if types == 'int':
                            _dSE = np.round( _dS * ((1 - v_groups.v_beta_reduct)*dSprob_sum) / (rate_immune + (1 - v_groups.v_beta_reduct)*dSprob_sum))
                        else:
                            _dSE = _dS * ((1 - v_groups.v_beta_reduct)*dSprob_sum) / (rate_immune + (1 - v_groups.v_beta_reduct)*dSprob_sum)
    
                        E_out = rv_gen(rnd_stream, v_groups._E[_t], rate_E)
                        v_groups._E[_t + 1] = v_groups._E[_t] + _dSE - E_out
                            
                        _dSR = _dS - _dSE
                        v_policy._vaccine_groups[3]._S[_t + 1] = v_policy._vaccine_groups[3]._S[_t + 1] + _dSR
                        #breakpoint()
                    else:
                        _dS = rv_gen(rnd_stream, v_groups._S[_t], (1 - v_groups.v_beta_reduct)*dSprob_sum)
                        # Dynamics for E
                        E_out = rv_gen(rnd_stream, v_groups._E[_t], rate_E)
                        v_groups._E[_t + 1] = v_groups._E[_t] + _dS - E_out
                    
                    
                    if t >= 711 and v_groups.v_name != 'v_3':
                        immune_escape_R = rv_gen(rnd_stream, v_groups._R[_t], rate_immune)
                        v_policy._vaccine_groups[3]._S[_t + 1] = v_policy._vaccine_groups[3]._S[_t + 1] + immune_escape_R
                    
                    v_groups._S[_t + 1] = v_groups._S[_t + 1] + v_groups._S[_t] - _dS #+ v_policy(t, v_groups)
          
                    # Dynamics for PY
                    EPY = rv_gen(rnd_stream, E_out, epi.tau * ( 1 - v_groups.v_tau_reduct))
                    PYIY = rv_gen(rnd_stream, v_groups._PY[_t], rate_PYIY)
                    v_groups._PY[_t + 1] = v_groups._PY[_t] + EPY - PYIY
                    
                    # Dynamics for PA
                    EPA = E_out - EPY
                    PAIA = rv_gen(rnd_stream, v_groups._PA[_t], rate_PAIA)
                    v_groups._PA[_t + 1] = v_groups._PA[_t] + EPA - PAIA
                
                    # Dynamics for IA
                    IAR = rv_gen(rnd_stream, v_groups._IA[_t], rate_IAR)
                    v_groups._IA[_t + 1] = v_groups._IA[_t] + PAIA - IAR

                    # Dynamics for IY
                    IYR = rv_gen(rnd_stream, v_groups._IY[_t], rate_IYR)
                    IYD = rv_gen(rnd_stream, v_groups._IY[_t] - IYR, rate_IYD)
                    v_groups._IYIH[_t] = rv_gen(rnd_stream, v_groups._IY[_t] - IYR - IYD, rate_IYH)
                    v_groups._IYICU[_t] = rv_gen(rnd_stream, v_groups._IY[_t] - IYR - IYD - v_groups._IYIH[_t], rate_IYICU)
                    v_groups._IY[_t + 1] = v_groups._IY[_t] + PYIY - IYR - IYD - v_groups._IYIH[_t] - v_groups._IYICU[_t]
                    
                    # Dynamics for IH
                    IHR = rv_gen(rnd_stream, v_groups._IH[_t], rate_IHR)
                    v_groups._IHICU[_t] = rv_gen(rnd_stream, v_groups._IH[_t] - IHR, rate_IHICU)
                    v_groups._IH[_t + 1] = v_groups._IH[_t] + v_groups._IYIH[_t] - IHR - v_groups._IHICU[_t]
                
                    # Dynamics for ICU
                    ICUR = rv_gen(rnd_stream, v_groups._ICU[_t], rate_ICUR)
                    ICUD = rv_gen(rnd_stream, v_groups._ICU[_t] - ICUR, rate_ICUD)
                    v_groups._ICU[_t + 1] = v_groups._ICU[_t] + v_groups._IHICU[_t] - ICUD - ICUR + v_groups._IYICU[_t]
                    v_groups._ToICU[_t] = v_groups._IYICU[_t] + v_groups._IHICU[_t]
                    v_groups._ToIHT[_t] = v_groups._IYICU[_t] + v_groups._IYIH[_t]

                    # Dynamics for R
                    #v_groups._R[_t + 1] = v_groups._R[_t] + IHR + IYR + IAR + ICUR
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

            for idx, v_groups in enumerate(v_policy._vaccine_groups):
                # End of the daily disctretization
                v_groups.S[t + 1] = v_groups._S[step_size].copy() 
                v_groups.E[t + 1] = v_groups._E[step_size].copy()
                v_groups.IA[t + 1] = v_groups._IA[step_size].copy()
                v_groups.IY[t + 1] = v_groups._IY[step_size].copy()
                v_groups.PA[t + 1] = v_groups._PA[step_size].copy()
                v_groups.PY[t + 1] = v_groups._PY[step_size].copy()
                v_groups.IH[t + 1] = v_groups._IH[step_size].copy()
                v_groups.ICU[t + 1] = v_groups._ICU[step_size].copy()
                v_groups.R[t + 1] = v_groups._R[step_size].copy()
                v_groups.D[t + 1] = v_groups._D[step_size].copy()
 
                v_groups.IYIH[t] = v_groups._IYIH.sum(axis=0)
                v_groups.IYICU[t] = v_groups._IYICU.sum(axis=0)
                v_groups.IHICU[t] = v_groups._IHICU.sum(axis=0)
                v_groups.ToICU[t] = v_groups._ToICU.sum(axis=0)
                v_groups.ToIHT[t] = v_groups._ToIHT.sum(axis=0)
                v_groups.ToICUD[t] = v_groups._ToICUD.sum(axis=0)
                v_groups.ToIYD[t] = v_groups._ToIYD.sum(axis=0)
                v_groups.ToIY[t] = v_groups._ToIY.sum(axis=0)
                v_groups.ToIA[t] = v_groups._ToIA.sum(axis=0)
         
            if t == T_omicron:
                # Move almost half of the people from recovered to susceptible:
                immune_escape(epi.immune_escape_rate, t, types, v_policy, step_size)

            if t >= v_policy._vaccines.vaccine_start_time:

                S_before = np.zeros((5, 2))

                for idx, v_groups in enumerate(v_policy._vaccine_groups):
                    S_before += v_groups.S[t + 1]

                for idx, v_groups in enumerate(v_policy._vaccine_groups):

                    out_sum = np.zeros((A, L))
                    S_out = np.zeros((A*L, 1))
                    N_out = np.zeros((A*L, 1))

                    for vaccine_type in v_groups.v_out:
                        event = v_policy._vaccines.event_lookup(vaccine_type, v_policy._instance.cal.calendar[t])

                        if event is not None:

                            S_out = np.reshape(v_policy._allocation[vaccine_type][event]["assignment"], (A*L, 1))
                            if t >= T_omicron:
                                if v_groups.v_name == "v_1" or v_groups.v_name == "v_2":
                                    S_out = epi.immune_escape_rate * np.reshape(v_policy._allocation[vaccine_type][event]["assignment"], (A*L, 1))

                            N_out = v_policy._vaccines.get_num_eligible(instance.N, instance.A * instance.L, v_groups.v_name, v_groups.v_in, v_groups.v_out, v_policy._instance.cal.calendar[t])

                            ratio_S_N = np.array([0 if N_out[i] == 0 else float(S_out[i]/N_out[i]) for i in range(len(N_out))]).reshape((A, L))

                            if types == 'int':
                                out_sum += np.round(ratio_S_N*v_groups._S[step_size])
                            else:
                                out_sum += ratio_S_N*v_groups._S[step_size]

                    in_sum = np.zeros((A, L))
                    S_in = np.zeros((A*L, 1))
                    N_in = np.zeros((A*L, 1))

                    for vaccine_type in v_groups.v_in:

                        for v_g in v_policy._vaccine_groups:
                            if v_g.v_name == v_policy._allocation[vaccine_type][0]["from"]:
                                v_temp = v_g

                        event = v_policy._vaccines.event_lookup(vaccine_type, v_policy._instance.cal.calendar[t])

                        if event is not None:
                            S_in = np.reshape(v_policy._allocation[vaccine_type][event]["assignment"], (A*L, 1))

                            if t >= T_omicron:
                                if (v_groups.v_name == "v_3" and v_temp.v_name == "v_2") or (v_groups.v_name == "v_2" and v_temp.v_name == "v_1"):
                                    S_in = epi.immune_escape_rate * np.reshape(v_policy._allocation[vaccine_type][event]["assignment"], (A*L, 1))

                            N_in = v_policy._vaccines.get_num_eligible(instance.N, instance.A * instance.L, v_temp.v_name, v_temp.v_in, v_temp.v_out, v_policy._instance.cal.calendar[t])
                            ratio_S_N = np.array([0 if N_in[i] == 0 else float(S_in[i]/N_in[i]) for i in range(len(N_in))]).reshape((A, L))

                            if types == 'int':
                                in_sum += np.round(ratio_S_N*v_temp._S[step_size])
                            else:
                                in_sum += ratio_S_N*v_temp._S[step_size]

                    if types == "float":
                        v_groups.S[t + 1] = v_groups.S[t + 1] + (np.array(in_sum - out_sum))
                    else:
                        out_sum = np.round(out_sum) 
                        in_sum = np.round(in_sum)
                        v_groups.S[t + 1] = v_groups.S[t + 1] + np.round(np.array(in_sum - out_sum))

                    S_after = np.zeros((5, 2))

                for idx, v_groups in enumerate(v_policy._vaccine_groups):
                    S_after += v_groups.S[t + 1]

                imbalance = np.abs(np.sum(S_before - S_after, axis = (0,1)))

                assert (imbalance < 1E-2).any(), f'fPop inbalance in vaccine flow in between compartment S {imbalance} at time {instance.cal.calendar[t]}, {t}'    

            for idx, v_groups in enumerate(v_policy._vaccine_groups):
                v_groups._S = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._E = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._IA = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._IY = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._PA = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._PY = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._IH = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._ICU = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._R = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._D = np.zeros((step_size + 1, A, L), dtype=types)
                v_groups._IYIH = np.zeros((step_size, A, L))
                v_groups._IYICU = np.zeros((step_size, A, L))
                v_groups._IHICU = np.zeros((step_size, A, L))
                v_groups._ToICU = np.zeros((step_size, A, L))
                v_groups._ToIHT = np.zeros((step_size, A, L))
                v_groups._ToICUD = np.zeros((step_size, A, L))
                v_groups._ToIYD = np.zeros((step_size, A, L))
                v_groups._ToIA = np.zeros((step_size, A, L))
                v_groups._ToIY = np.zeros((step_size, A, L))
                  
                v_groups._S[0] = v_groups.S[t + 1].copy()
                v_groups._E[0] = v_groups.E[t + 1].copy()
                v_groups._IA[0] = v_groups.IA[t + 1].copy()
                v_groups._IY[0] = v_groups.IY[t + 1].copy()
                v_groups._PA[0] = v_groups.PA[t + 1].copy()
                v_groups._PY[0] = v_groups.PY[t + 1].copy()
                v_groups._IH[0] = v_groups.IH[t + 1].copy()
                v_groups._ICU[0] = v_groups.ICU[t + 1].copy()
                v_groups._R[0] = v_groups.R[t + 1].copy() 
                v_groups._D[0] = v_groups.D[t + 1].copy()

        return v_policy

def rv_gen(rnd_stream, n, p, round_opt=1):
    
    if rnd_stream is None:
        return n * p
    else:
        if round_opt:
            nInt = np.round(n)
            return rnd_stream.binomial(nInt.astype(int), p)
        else:
            return rnd_stream.binomial(n, p)


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