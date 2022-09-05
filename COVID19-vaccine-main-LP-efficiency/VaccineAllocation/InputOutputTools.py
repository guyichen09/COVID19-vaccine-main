###############################################################################

# InputOutputTools.py
# Use this module to import and export simulation state data
#   so that a simulation can be loaded on another computer
#   and run from a previous starting point t > 0.
# Use this module to import simulation state data
#   from "realistic sample paths" that match historical data
#   with an R-squared > 0.75, to simulate policies starting
#   from the end of the historical data period onwards.

# Note that importing / exporting does not work for partial days
#   (discrete steps within a day), so we assume days are fully completed.

###############################################################################

# Imports
import json
import numpy as np
import copy

###############################################################################

# Tuples containing names of attributes of SimReplication objects,
#   VaccineGroup objects, and MultiTierPolicy objects that must be
#   imported/exported to save/load a simulation replication correctly.

# The relevant instance of EpiSetup has an attribute
#   "random_params_dict" that is a dictionary of the randomly sampled
#   random variables -- this information is stored on that instance
#   rather than in this module. This is because the user specifies
#   which random variables

# List of names of SimReplication attributes to be serialized as a .json file
#   for saving a simulation replication or loading a simulation replication
#   from a timepoint t > 0 (rather than starting over from scratch)
SimReplication_IO_var_names = ("rng_seed",
                     "ICU_history", "IH_history", "ToIHT_history", "ToIY_history",
                     "next_t",
                     "S", "E", "IA", "IY", "PA", "PY", "R", "D",
                     "IH", "ICU", "IYIH", "IYICU", "IHICU",
                     "ToICU", "ToIHT", "ToICUD", "ToIYD", "ToIA", "ToIY")

# List of names of SimReplication attributes that are lists of arrays
SimReplication_IO_list_of_arrays_var_names = ("ICU_history", "IH_history",
                                              "ToIHT_history", "ToIY_history")

# List of names of SimReplication attributes that are arrays
SimReplication_IO_arrays_var_names = ("S", "E", "IA", "IY", "PA", "PY", "R", "D",
                     "IH", "ICU", "IYIH", "IYICU", "IHICU",
                     "ToICU", "ToIHT", "ToICUD", "ToIYD", "ToIA", "ToIY")

# List of names of VaccineGroup attributes to be serialized as a .json file
VaccineGroup_IO_var_names = ("v_beta_reduct", "v_tau_reduct", "v_beta_reduct_delta",
                                  "v_tau_reduct_delta", "v_tau_reduct_omicron") \
                            + SimReplication_IO_arrays_var_names

# List of names of VaccineGroup attributes that are arrays
VaccineGroup_IO_arrays_var_names = SimReplication_IO_arrays_var_names

# List of names of MultiTierPolicy attributes to be serialized as a .json file
MultiTierPolicy_IO_var_names = ("community_transmission",
                                     "lockdown_thresholds",
                                     "tier_history")

###############################################################################

def load_vars_from_file(sim_rep, sim_rep_filename,
                       vaccine_group_v0_filename, vaccine_group_v1_filename,
                       vaccine_group_v2_filename, vaccine_group_v3_filename,
                        random_params_filename=None, multi_tier_policy_filename=None):
    '''
    Modifies a SimReplication object sim_rep in place to match the
        last state of a previously run simulation replication
    Updates sim_rep attributes according to the data in sim_rep_filename
    Updates sim_rep.policy attributes according to the data in multi_tier_policy_filename
        (this can be None, meaning there is no relevant policy data)
    Updates vaccine group attributes for each instance of VaccineGroup in
        sim_rep.vaccine_groups according to the data in vaccine_group_v0_filename,
        vaccine_group_v1_filename, vaccine_group_v2_filename, and vaccine_group_v3_filename
    Updates sim_rep.epi_rand according to the data in random_params_filename
        (this can be None, meaning that new parameters will be randomly sampled
        and these parameters are different from the ones that generated the
        loaded simulation replication)

    :param sim_rep: [SimReplication obj]
    :param sim_rep_filename: [str] .json file with entries corresponding to
        SimReplication_IO_var_names
    :param vaccine_group_v0_filename: [str] .json file with entries corresponding to
        VaccineGroup_IO_var_names for vaccine group v_0
    :param vaccine_group_v1_filename: [str] .json file with entries corresponding to
        VaccineGroup_IO_var_names for vaccine group v_1
    :param vaccine_group_v2_filename: [str] .json file with entries corresponding to
        VaccineGroup_IO_var_names for vaccine group v_2
    :param vaccine_group_v3_filename: [str] .json file with entries corresponding to
        VaccineGroup_IO_var_names for vaccine group v_3
    :param multi_tier_policy_filename: [str] .json file with entries corresponding to
        MultiTierPolicy_IO_var_names
    :param random_params_filename: [str] .json file with entries corresponding to
        sim_rep.epi_rand.random_params_dict, i.e. parameters that are
        randomly sampled at the beginning of the replication
    :return: [None]
    '''

    # Update sim_rep variables
    d = json.load(open(sim_rep_filename))
    sim_rep_filename.close()
    load_vars_from_dict(sim_rep, d, sim_rep.state_vars + sim_rep.tracking_vars)

    # Update vaccine group variables
    vaccine_group_filenames = [vaccine_group_v0_filename, vaccine_group_v1_filename,
                       vaccine_group_v2_filename, vaccine_group_v3_filename]

    for i in range(len(vaccine_group_filenames)):
        vaccine_group = sim_rep.vaccine_groups[i]
        d = json.load(open(vaccine_group_filenames[i]))
        vaccine_group_filenames[i].close()
        load_vars_from_dict(vaccine_group, d, sim_rep.state_vars + sim_rep.tracking_vars)

        # Modify the first step of the next day so that the
        #   discretization (with steps) of the next day is correct
        for attribute in sim_rep.state_vars:
            vars(vaccine_group)["_" + attribute][0] = getattr(vaccine_group, attribute)

    # (Optional) update policy variables
    if multi_tier_policy_filename is not None:
        d = json.load(open(multi_tier_policy_filename))
        multi_tier_policy_filename.close()
        load_vars_from_dict(sim_rep.policy, d)

    # (Optional) update epidemiological parameters
    if random_params_filename is not None:

        # Load randomly sampled epi parameters
        d = json.load(open(random_params_filename))
        random_params_filename.close()
        load_vars_from_dict(epi_rand, d)

        # Update sim_rep.epi_rand accordingly
        # Create a copy of the base epi parameters that do not change
        #   across simulation replications
        # Update the dictionary storing randomly sampled parameters
        # Recompute key quantities that depend on the randomly
        #   sampled parameters
        # Modify sim_rep.epi_rand in place to reflected loaded changes
        epi_rand = copy.deepcopy(sim_rep.instance.base_epi)
        epi_rand.random_params_dict = d
        epi_rand.setup_base_params()
        sim_rep.epi_rand = epi_rand

def load_vars_from_dict(simulation_object, loaded_dict, keys_to_convert_to_array=[]):
    for k in loaded_dict.keys():
        if k in keys_to_convert_to_array and isinstance(loaded_dict[k], list):
            setattr(simulation_object, k, np.array(loaded_dict[k]))
        else:
            setattr(simulation_object, k, loaded_dict[k])

def export_rep_to_file(sim_rep, sim_rep_filename, multi_tier_policy_filename,
                       vaccine_group_v0_filename, vaccine_group_v1_filename,
                       vaccine_group_v2_filename, vaccine_group_v3_filename, random_params_filename):
    '''
    LP note: there is probably a more efficient way to
        create this sub-dictionary... so this is still in progress...
    '''

    d = {}
    for k in SimReplication_IO_var_names:
        if k in SimReplication_IO_list_of_arrays_var_names:
            list_of_lists = [matrix.tolist() for matrix in getattr(sim_rep, k)]
            d[k] = list_of_lists
        elif k in SimReplication_IO_arrays_var_names:
            d[k] = getattr(sim_rep, k).tolist()
        else:
            d[k] = getattr(sim_rep, k)
    json.dump(d, open(sim_rep_filename, "w"))
    sim_rep_filename.close()

    if multi_tier_policy_filename is not None:
        d = {}
        for k in MultiTierPolicy_IO_var_names:
            d[k] = getattr(sim_rep.policy, k)
        json.dump(d, open(multi_tier_policy_filename, "w"))
        multi_tier_policy_filename.close()

    vaccine_group_filenames = [vaccine_group_v0_filename,
                               vaccine_group_v1_filename,
                               vaccine_group_v2_filename,
                               vaccine_group_v3_filename]

    for i in range(len(vaccine_group_filenames)):
        vaccine_group = sim_rep.vaccine_groups[i]
        d = {}
        for k in VaccineGroup_IO_var_names:
            if k in VaccineGroup_IO_arrays_var_names:
                d[k] = [matrix.tolist() for matrix in getattr(vaccine_group, k)]
            else:
                d[k] = getattr(vaccine_group, k)
        json.dump(d, open(vaccine_group_filenames[i], "w"))
        vaccine_group_filenames[i].close()

    if random_params_filename is not None:
        d = sim_rep.epi_rand.random_params_dict
        for k in d.keys():
            if isinstance(d[k], np.ndarray):
                d[k] = d[k].tolist()
        json.dump(d, open(random_params_filename, "w"))
        random_params_filename.close()