import json

# Note that importing / exporting does not work for partial days, only fully completed days

class SimReplicationIO:

    '''
    Container for methods that act on instances of SimReplication,
        VaccineGroup, and MultiTierPolicy, for input and output
    '''

    # List of names of SimReplication attributes to be serialized as a .json file
    #   for saving a simulation replication or loading a simulation replication
    #   from a timepoint t > 0 (rather than starting over from scratch)
    SimReplication_IO_variable_names = ("rng_seed",
                         "ICU_history", "IH_history", "ToIHT_history", "ToIY_history",
                         "next_t",
                         "S", "E", "IA", "IY", "PA", "PY", "R", "D",
                         "IH", "ICU", "IYIH", "IYICU", "IHICU",
                         "ToICU", "ToIHT", "ToICUD", "ToIYD", "ToIA", "ToIY")

    # List of names of SimReplication attributes that are lists of arrays
    SimReplication_IO_list_of_arrays_variable_names = ("ICU_history", "IH_history", "ToIHT_history", "ToIY_history")

    # List of names of SimReplication attributes that are arrays
    SimReplication_IO_arrays_variable_names = ("S", "E", "IA", "IY", "PA", "PY", "R", "D",
                         "IH", "ICU", "IYIH", "IYICU", "IHICU",
                         "ToICU", "ToIHT", "ToICUD", "ToIYD", "ToIA", "ToIY")

    # List of names of MultiTierPolicy attributes to be serialized as a .json file
    MultiTierPolicy_IO_variable_names = ("community_tranmission",
                                         "lockdown_thresholds",
                                         "tier_history")

    # List of names of VaccineGroup attributes to be serialized as a .json file
    VaccineGroup_IO_variable_names = ("v_beta_reduct", "v_tau_reduct", "v_beta_reduct_delta",
                                      "v_tau_reduct_delta", "v_tau_reduct_omicron")

    # List of names of VaccineGroup attributes that are arrays
    VaccineGroup_IO_arrays_variable_names = SimReplication_IO_arrays_variable_names

    @staticmethod
    def load_vars_from_file(sim_replication, sim_replication_filename, multi_tier_policy_filename,
                           vaccine_group_v0_filename, vaccine_group_v1_filename,
                           vaccine_group_v2_filename, vaccine_group_v3_filename):

        f = open(sim_replication_filename)
        d = json.load(f)
        SimReplicationIO.load_vars_from_dict(sim_replication, d)

        f = open(multi_tier_policy_filename)
        d = json.load(f)
        SimReplicationIO.load_vars_from_dict(sim_replication.policy, d)

        vaccine_group_filenames = [vaccine_group_v0_filename, vaccine_group_v1_filename,
                           vaccine_group_v2_filename, vaccine_group_v3_filename]

        for i in range(len(vaccine_group_filenames)):
            f = open(vaccine_group_filenames[i])
            d = json.load(f)
            SimReplicationIO.load_vars_from_dict(sim_replication.vaccine_groups[i], d)

    @staticmethod
    def load_vars_from_dict(simulation_object, d):
        for k in d.keys():
            setattr(simulation_object, k, d[k])

    @staticmethod
    def export_rep_to_file(sim_replication, sim_replication_filename, multi_tier_policy_filename,
                           vaccine_group_v0_filename, vaccine_group_v1_filename,
                           vaccine_group_v2_filename, vaccine_group_v3_filename):
        '''
        LP note: there is probably a more efficient way to
            create this sub-dictionary... so this is still in progress...
        '''

        sim_replication.rng_state = sim_replication.rng_generator.get_state()

        d = {}
        for k in SimReplicationIO.SimReplication_IO_variable_names:
            if k in SimReplicationIO.SimReplication_IO_list_of_arrays_variable_names:
                list_of_lists = [matrix.tolist() for matrix in vars(sim_replication)[k]]
                d[k] = list_of_lists

            elif k in SimReplicationIO.SimReplication_IO_arrays_variable_names:
                d[k] = vars(sim_replication)[k].tolist()

            else:
                d[k] = vars(sim_replication)[k]
        with open(sim_replication_filename, "w") as f:
            json.dump(d, f)

        d = {}
        for k in SimReplicationIO.MultiTierPolicy_IO_variable_names:
            d[k] = vars(sim_replication.policy)[k]
        with open(multi_tier_policy_filename, "w") as f:
            json.dump(d, f)

        vaccine_group_filenames = [vaccine_group_v0_filename, vaccine_group_v1_filename,
                           vaccine_group_v2_filename, vaccine_group_v3_filename]

        for i in range(len(vaccine_group_filenames)):
            d = {}
            for k in SimReplicationIO.VaccineGroup_IO_variable_names:
                if k in SimReplicationIO.VaccineGroup_IO_arrays_variable_names:
                    list_of_lists = [matrix.tolist() for matrix in vars(sim_replication.vaccine_groups[i])[k]]
                    d[k] = list_of_lists
                else:
                    d[k] = vars(sim_replication.vaccine_groups[i])[k]
            with open(vaccine_group_filenames[i], "w") as f:
                json.dump(d, f)