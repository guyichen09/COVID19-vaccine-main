from pathlib import Path
import os
import pickle
city = 'austin'
instances_path = Path(__file__).parent

#To get path of the output directiory file
path_file=''

if __name__ == "__main__":
     # list all .p files from the output folder
     path = instances_path.parent / 'output'
     fileList = os.listdir(path)
     for instance_raw in fileList:
         if ".p" in instance_raw:
             if "austin" in instance_raw:
                 instance_name = instance_raw[:-2]
                 path_file = f'output/{instance_name}.p'


# use this when we generate new seeds and want to write just seeds array
def write_seeds(city, seeds_file_name='seeds.p'):
    seedsinput_old=instances_path.parent  
    seedsinput=instances_path          
    print(seedsinput_old / seeds_file_name)
    try:
        with open(seedsinput_old / path_file, 'rb') as infile:
            seeds_data = pickle.load(infile)
            file_path = seedsinput / 'new_seed.p'
            with open(str(file_path), 'wb') as outfile:
                pickle.dump(((seeds_data[-1][0], seeds_data[-1][1])), 
                            outfile, pickle.HIGHEST_PROTOCOL)
        print(seeds_data[-1][0], seeds_data[-1][1]  )
        return seeds_data[-1][0], seeds_data[-1][1]    
    except:
        return [],[]



seeds_input_file_name='new_seed.p'
def load_seeds(city, seeds_file_name='newseed.p'):
    # read in the seeds file
    seedsinput_old=instances_path.parent  
    seedsinput=instances_path          
    print(seedsinput_old / seeds_file_name)
    try:
        with open(seedsinput_old / seeds_file_name, 'rb') as infile:
            seeds_data = pickle.load(infile)

        return seeds_data[0], seeds_data[1]    

    except:
        return [],[]
    
    

write_seeds(city, path_file)



                