import pandas as pd
import numpy as np
import csv


def cleandata():
    data = pd.read_excel('death_files.xlsx')
    data = data.to_dict()   
    for element in data:
        print(element)
    print(sum(data['total_deaths']))
    for idx in range(len(data['Date1'])):
        difference = data['total_deaths'][idx] - data['hospital_deaths'][idx]
        if difference < 0:
            data['updated_total_deaths'][idx] = data['total_deaths'][idx] - difference
            if idx != len(data['Date1']) -1 :
                data['total_deaths'][idx + 1] += difference
        else:
            data['updated_total_deaths'][idx] = data['total_deaths'][idx]
            
            
    print(sum(data['updated_total_deaths']))
    
    with open('updated_total_deaths.csv', 'w', newline='') as file:
        writer = csv.writer(file)    
        writer.writerow(['date', 'total_deaths', 'hospital_deaths'])
        for idx in range(len(data['Date1'])):
            output = np.concatenate(([data['Date1'][idx]], [data['updated_total_deaths'][idx]], [data['hospital_deaths'][idx]]))
            writer.writerow(output) 
        
cleandata()