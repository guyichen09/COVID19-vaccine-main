import copy
from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
import OptTools

# Import other Python packages
import numpy as np
import time

# OptTools.aggregate_data_policies_on_sample_paths(15, 300, 3)

for lam in np.arange(10000):
    print(OptTools.WIP(lam))