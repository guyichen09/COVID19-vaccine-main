# LP official scratch file
# Functions/classes I tried out but didn't end up using

class SimReplicationDataContainer:

    def __init__(self, thresholds_array, rep_identifiers_array=None):
        self.thresholds_array = thresholds_array
        self.rep_identifiers_array = rep_identifiers_array

        num_policies = len(thresholds_array)

        self.cost_output = [[] for i in range(num_policies)]
        self.feasibility_output = [[] for i in range(num_policies)]

        d = {}
        i = 0
        for threshold in thresholds_array:
            d[threshold] = i
            i += 1

        self.thresholds_to_ix_dict = d

    def save_output(self, thresholds, cost_output, feasibility_output):
        ix = self.thresholds_to_ix_dict[thresholds]
        self.cost_output[ix].append(cost_output)
        self.feasibility_output[ix].append(feasibility_output)

    def export_output_to_csv(self, thresholds_filename,
                             cost_output_filename,
                             feasibility_output_filename,
                             rep_identifiers_filename=None):
        with open(thresholds_filename, "w", newline="") as f:
            csv.writer(f).writerow(self.thresholds_array)
        with open(cost_output_filename, "w", newline="") as f:
            csv.writer(f).writerows(self.cost_output)
        with open(feasibility_output_filename, "w", newline="") as f:
            csv.writer(f).writerows(self.feasibility_output)
        if rep_identifiers_filename is not None:
            with open(rep_identifiers_filename, "w", newline="") as f:
                csv.writer(f).writerow(self.rep_identifiers_array)