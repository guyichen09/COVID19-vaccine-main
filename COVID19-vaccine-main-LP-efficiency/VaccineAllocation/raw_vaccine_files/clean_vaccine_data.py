import pandas
import numpy as np
import csv
import datetime as dt
import os

datetime_formater = '%Y-%m-%d %H:%M:%S'
date_formater = '%Y-%m-%d'

N = [128527,	9350,
     327148,	37451,
     915894,	156209,
     249273,	108196,
     132505,	103763]


def clean_data(file_name, sheet_naming):
   
    data = pandas.read_excel(file_name, sheet_name= sheet_naming).to_dict()
    # for element in data:
    #     print(element)
    # data.pop('Gender') 
    #data.pop('Race/Ethnicity') 
    data.pop('People Fully Vaccinated') 
    data.pop('Doses Administered')
    #print('')
    # for element in data:
    #     print(element)
    
    total_65_under = 0
    total_65_over = 0    
    for idx in range(len(data['Age Group'])):
        if data['Age Group'][idx] == "16-49 years" or data['Age Group'][idx] == '50-64 years' or data['Age Group'][idx] == '16-64 years':
            total_65_under += data['People Vaccinated with at least One Dose'][idx]
        elif data['Age Group'][idx] == "65-79 years" or data['Age Group'][idx] == "80+ years" or data['Age Group'][idx] ==  "85+ years" or data['Age Group'][idx] == '75-84 years' or data['Age Group'][idx] == "65-74 years":
            total_65_over += data['People Vaccinated with at least One Dose'][idx]
    
    percentage_under = total_65_under/(total_65_under + total_65_over)
    # print('under 65', total_65_under)   
    # print('65+', total_65_over)        
    # print('total', total_65_over + total_65_under) 
        

def clean_county_data(file_name, sheet_naming):
 
    data = pandas.read_excel(file_name, sheet_name= sheet_naming).to_dict()
    
    austin_total = 0
    for idx in range(len(data['County Name'])):
        if data['County Name'][idx] == "Bastrop" or data['County Name'][idx] == "Caldwell" or data['County Name'][idx] == "Hays" or data['County Name'][idx] == "Travis" or data['County Name'][idx] == "Williamson":
            austin_total += data['Total Doses Allocated'][idx]

    return austin_total

def clean_county_age_data(file_name, sheet_name):
    try:
        data = pandas.read_excel(file_name, sheet_name= sheet_name)
        print(file_name)
        data.columns = ['County Name', 	'Age Group', 'Doses Administered', 'People Vaccinated with at least One Dose',	'People Fully Vaccinated', 'People with Booster Dose'] 
        data = data.to_dict()
        for element in data:
            print(element)
            
        austin = np.zeros(4)
        for idx in range(len(data['County Name'])):
            if data['County Name'][idx].lower() == "bastrop" or data['County Name'][idx].lower() == "caldwell" or data['County Name'][idx].lower() == "hays" or data['County Name'][idx].lower() == "travis" or data['County Name'][idx].lower() == "williamson":
                age_str=data['Age Group'][idx]
                if type(age_str) == dt.datetime:
                    age_str = age_str.strftime(datetime_formater)  
                if type(age_str) == str:
                    age_str = age_str.strip().lower()   
                    
                if age_str== '12-15 years' or age_str ==  '05-11 years' or age_str ==  '5-11 years' or age_str ==  '2021-05-11' or age_str ==  '2021-12-15':          
                    austin[0] += data['People Vaccinated with at least One Dose'][idx]
                if age_str == '16-49 years':
                    austin[1] += data['People Vaccinated with at least One Dose'][idx]
                elif age_str == '50-64 years':
                    austin[2] += data['People Vaccinated with at least One Dose'][idx]
                elif age_str == '65-79 years' or age_str == '80+ years' or age_str == '65-79 years & 80+ years':
                    austin[3] += data['People Vaccinated with at least One Dose'][idx]
                else:
                    austin[0] += data['People Vaccinated with at least One Dose'][idx]*(N[2] + N[3])*0.5/((N[2] + N[3])*0.5 + N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                    austin[1] += data['People Vaccinated with at least One Dose'][idx]*(N[4] + N[5])/((N[2] + N[3])*0.5 + N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                    austin[2] += data['People Vaccinated with at least One Dose'][idx]*(N[6] + N[7])/((N[2] + N[3])*0.5 + N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                    austin[3] += data['People Vaccinated with at least One Dose'][idx]*(N[8] + N[9])/((N[2] + N[3])*0.5 + N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                    # austin[1] += data['People Vaccinated with at least One Dose'][idx]*(N[4] + N[5])/( N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                    # austin[2] += data['People Vaccinated with at least One Dose'][idx]*(N[6] + N[7])/(N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                    # austin[3] += data['People Vaccinated with at least One Dose'][idx]*(N[8] + N[9])/( N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
        

    except FileNotFoundError:
        print('File is not present')
        return None
    
    except ValueError:
        print('No Worksheet Available')
        return None
    
    except TypeError:
        print('NoneType')
        return None
    
    return austin


file_list = sorted(os.listdir())
#file_list=file_list[49:-5]
file_list=file_list[-50:-5]

with open('austin_first_dose.csv', 'w', newline='') as file:

        writer = csv.writer(file)    
        #writer.writerow(['vaccine_time', 'vaccine_amount', 'A1-R1', "A1-R2", 'A2-R1',	'A2-R2', 'A3-R1', 
        #                            'A3-R2',	'A4-R1', 'A4-R2', 'A5-R1',	'A5-R2' ])
        writer.writerow(['vaccine_time', 'vaccine_amount', '12-15', '16-49', '50-64','65-79'])
        
        for ind in file_list:
            name = ind
            print(name)
            austin = clean_county_age_data(name, 'By County, Age') 
            
            vaccine_amount = sum(austin)/sum(N)
            
            #allocation = [0, 0, 0, 0, 0, austin[0], austin[1]*N[6]/(N[6] + N[7]), 
            #             austin[1]*N[7]/(N[6] + N[7]), austin[2]*N[8]/(N[8] + N[9]), austin[2]*N[9]/(N[8] + N[9])]
            ind=ind[0:10]
            output = np.concatenate(([str(ind)], [vaccine_amount], austin))
            writer.writerow(output) 
   