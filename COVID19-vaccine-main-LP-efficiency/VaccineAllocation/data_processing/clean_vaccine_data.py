import pandas
import numpy as np
import csv
import datetime as dt

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

def clean_county_age_data(file_name, sheet_name, date):
    try:
        data = pandas.read_excel(file_name, sheet_name= sheet_name)
        data.columns = ['County Name', 	'Age Group', 'Doses Administered', 'People Vaccinated with at least One Dose',	'People Fully Vaccinated', 'People with Booster Dose'] 
        data = data.to_dict()
        print('file:', file_name)
        austin = np.zeros(4)
        total = 0
        for idx in range(len(data['County Name'])):
            if data['County Name'][idx].lower() == "bastrop" or data['County Name'][idx].lower() == "caldwell" or data['County Name'][idx].lower() == "hays" or data['County Name'][idx].lower() == "travis" or data['County Name'][idx].lower() == "williamson":
                age_str = data['Age Group'][idx]
                if type(age_str) == dt.datetime:
                    age_str = age_str.strftime(datetime_formater)  
                if type(age_str) == str:
                    age_str = age_str.strip().lower()   
                total += data['People Vaccinated with at least One Dose'][idx]
                if age_str ==  '12-15 years' or age_str ==  '05-11 years' or age_str ==  '5-11 years' or age_str ==  '2021-05-11 00:00:00' or age_str ==  '2021-12-15 00:00:00':          
                    austin[0] += data['People Vaccinated with at least One Dose'][idx]
                elif age_str == '16-49 years' or  age_str == '16-49':
                    austin[1] += data['People Vaccinated with at least One Dose'][idx]
                elif age_str == '50-64 years' or age_str == '50-64':
                    austin[2] += data['People Vaccinated with at least One Dose'][idx]
                elif age_str == '65-79 years' or age_str == '80+ years' or age_str == '65-79 years & 80+ years' or age_str == '65-79' or age_str == '80+':
                    austin[3] += data['People Vaccinated with at least One Dose'][idx]
                elif age_str == 'unknown':
                    if data['Age Group'][idx-1].strip().lower() != 'other':
                        if date > dt.datetime(2021, 5, 12):
                            austin[0] += data['People Vaccinated with at least One Dose'][idx]*(N[2] + N[3])/(N[2] + N[3] + N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                            austin[1] += data['People Vaccinated with at least One Dose'][idx]*(N[4] + N[5])/(N[2] + N[3] + N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                            austin[2] += data['People Vaccinated with at least One Dose'][idx]*(N[6] + N[7])/(N[2] + N[3]+ N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                            austin[3] += data['People Vaccinated with at least One Dose'][idx]*(N[8] + N[9])/(N[2] + N[3]+ N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                        else:
                            austin[1] += data['People Vaccinated with at least One Dose'][idx]*(N[4] + N[5])/( N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                            austin[2] += data['People Vaccinated with at least One Dose'][idx]*(N[6] + N[7])/(N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
                            austin[3] += data['People Vaccinated with at least One Dose'][idx]*(N[8] + N[9])/( N[4] + N[5] + N[6] + N[7] + N[8] + N[9])
        print(sum(austin))
        assert np.abs(sum(austin) - total) < 1E2, f'unbalanced: {sum(austin) - total}' 
        return austin
    
    except FileNotFoundError:
        print('File is not present')
        return None
    
    



with open('austin_first_dose.csv', 'w', newline='') as file:
    writer = csv.writer(file)    
    writer.writerow(['vaccine_time', 'vaccine_amount', '12-15', '16-49', '50-64','65-79'])
    
    for month in range(1, 2):
        for ind in range(1, 4):
            # try:
            date = dt.datetime(2021, month, ind)
            if ind <10:
                name = '2022-0'+ str(month) + '-0' + str(ind) + ' COVID-19 Vaccine Data by County.xlsx'
            else:
                name = '2022-0'+ str(month) + '-' + str(ind) + ' COVID-19 Vaccine Data by County.xlsx'
          
            
            
            austin = clean_county_age_data(name, 'By County, Age', date)
    
            if austin is not None:
                vaccine_amount = sum(austin)/sum(N)
                    # print(vaccine_amount)
                    #print(sum(austin))
            else:
                vaccine_amount = None
                austin = []*4
            output = np.concatenate(([str(month) + '/'+str(ind)+'/21'], [vaccine_amount], austin))
            writer.writerow(output)                
            # except ValueError:
            #     print('day is out of range for month')
    
