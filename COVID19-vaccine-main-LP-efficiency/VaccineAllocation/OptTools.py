from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools

def policy_generator(stage1_info, stage2_info, stage3_info, stage4_info, stage5_info):
    '''
    :param stage1_info: [3-tuple] with elements corresponding to
        start point, end point, and step size (all must be integers)
        for candidate values for stage 1
    :param stage2_info: same as above but for stage 2
    :param stage3_info: same as above but for stage 3
    :param stage4_info: same as above but for stage 4
    :param stage5_info: same as above but for stage 5
    :return:
    '''

