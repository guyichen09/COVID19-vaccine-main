import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

#Data Source URL
response = requests.get('https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/Austin_MSA_Key_Indicators_(Public_View)/FeatureServer/0/query?where=0%3D0&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=')
response.raise_for_status()
#Access json content 
jsonResponse = response.json()
#Clean the json dictionary
all_data = jsonResponse['features']
df=pd.json_normalize(all_data)

#transfer from json date to normal date-time
index=0
for d in df['attributes.Date_Updated']:
    l = lambda x: datetime.fromtimestamp(x/1000)
    df['attributes.Date_Updated'][index]=l(d)
    index+=1

#Clean dataframe
df.drop(['attributes.ACS_Admiss', 'attributes.DisplayDate','attributes.No_Data', 'attributes.Record_Status'], inplace=True, axis=1)

#Rename column headers
dict={'attributes.Date_Updated' : 'date', "attributes.F7_Day_MA_NA" : "New Admin. 7D MA " , 
      "attributes.New_Cases_NC" : "New cases", "attributes.F7_Day_MA_NC" : "New cases 7D MA", 
      "attributes.Doubling_Time" : "Doubling time", "attributes.Curr_Hosp_Inpatients" : "Current Hosp. Inpatient",
      "attributes.F7_Day_MA_Hosp" : "New Hosp. 7D MA", "attributes.Curr_ICU": "hospitalized", 
      "attributes.F7_Day_MA_ICU" : "ICU 7D MA", "attributes.Curr_Ventilator" : "Current Ventilator",
      "attributes.F7_Day_MA_Vent" : "Ventilator 7D MA", "attributes.ObjectID": "Object ID",
      "attributes.New_Admissions" : "New Admissions" , "attributes.NC_L7day" : "New Cases 7D"}

df.rename(columns=dict,
          inplace=True)

df=df[['date', 'hospitalized']]

df['date'] = pd.to_datetime(df.date, format='%m/%d/%y')
df["date"] = df["date"].dt.strftime("%m/%d/%y")

instances_path = Path(__file__).parent
path1=path = instances_path.parent / 'instances/austin/austin_real_icu_updated.csv'

df_init=pd.read_csv(path1, index_col=None)
df_init=df_init[1:24]
df_init=df_init[["date", "hospitalized"]]

df_init=df_init.append(df)

#Convert to csv file in the directory 

path2 = instances_path.parent /'instances/austin/scraped_ICU_data.csv'

df_init.to_csv(path2, index=True)


