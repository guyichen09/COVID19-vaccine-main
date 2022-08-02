############

# 07262022

python3 -m pdb main_allocation.py austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 2 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -tr transmission.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv -n_policy=7  -v_boost booster_allocation_fixed.csv # -gt [-1,5,15,30,50] 

python3 -m pdb main_allocation.py austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 2 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -tr transmission.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv -n_policy=7  -v_boost booster_allocation_fixed.csv # -gt [-1,5,15,30,50] 

# 07142022

# checking if cost bug: testing 2 policies
python3.7 main_allocation.py austin -f=setup_data_Final.json -t=tiers_LP_debug.json -train_reps=2 -test_reps=1 -f_config=austin_test_IHT_LP_debug.json -n_proc=8 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv -gt [-1,1,2,3,4] 
python3.7 main_allocation.py austin -f=setup_data_Final.json -t=tiers_LP_debug.json -train_reps=2 -test_reps=1 -f_config=austin_test_IHT_LP_debug.json -n_proc=8 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv -gt [-1,500,1000,10000,100000] 

# 07052022

python3.7 main_allocation.py austin -f=setup_data_Final.json -t=tiers_LP_debug.json -train_reps=2 -test_reps=1 -f_config=austin_test_IHT_LP_debug.json -n_proc=8 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv # -pub=60 # -gt [-1,1,30,90,500] 

#############

python3 main_allocation.py austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 2 -test_reps 2 -f_config austin_test_IHT.json -n_proc 1 -tr transmission.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv -seed new_seed.p -n_policy 7  -v_boost booster_allocation_fixed.csv -gt [-1,5,15,30,50] 

python3 main_allocation.py austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 2 -test_reps 2 -f_config austin_test_IHT.json -n_proc 1 -tr transmission.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv -n_policy 7  -v_boost booster_allocation_fixed.csv -gt [-1,5,15,30,50] 







python3 -m pdb main_allocation.py austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 2 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -tr transmission.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv -n_policy 7  -v_boost booster_allocation_fixed.csv # -gt [-1,5,15,30,50] 



# Not my debugging

python3 -m pdb main_allocation.py austin -f=setup_data_Final.json -t=tiers5_opt_Final.json -train_reps=2 -test_reps=1 -f_config=austin_test_IHT.json -n_proc=1 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv #-gd="2021,5,5"

# My debugging

python3.7 -m pdb main_allocation.py austin -f=setup_data_Final.json -t=tiers_LP_debug.json -train_reps=100 -test_reps=1 -f_config=austin_test_IHT_LP_debug.json -n_proc=1 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv -pub=60 #-gd="2021,5,5"

python3.7 main_allocation.py austin -f=setup_data_Final.json -t=tiers_LP_debug.json -train_reps=2 -test_reps=1 -f_config=austin_test_IHT_LP_debug.json -n_proc=12 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv -pub=60 #-gd="2021,5,5"

#########################

for v in booster_allocation_fixed.csv 
do
	python3 -m pdb main_allocation.py austin -f=setup_data_Final8_2.json -t=tiers5_opt_Final.json -train_reps=3 -test_reps=3 -f_config=austin_test_IHT.json -n_proc=1 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=$v # -v_boost booster_allocation_fixed.csv # -gd="2020,6,10" # -seed=new_seed.p 
done











# breakpoints

b trigger_policies:48
cont
len(threshold_candidates[0][0])


when grid_size = 50 and tiers 2-5 have 3 values...
 len(threshold_candidates)
81

when grid_size = 5 and tiers 2-5 have 3 values...
 len(threshold_candidates)
81
but then threshold_candidates only has -1, 5, 15, 30, 50 -- none of the intermediate values





python3 -m pdb main_allocation.py austin -f=setup_data_Final.json -t=tiers_LP_debug.json -train_reps=2 -test_reps=1 -f_config=austin_test_IHT_LP_debug.json -n_proc=1 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv -pub=55


