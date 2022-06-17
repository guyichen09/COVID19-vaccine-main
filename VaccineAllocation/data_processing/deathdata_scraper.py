import pandas as pd
from pathlib import Path
from datetime import datetime

url="https://www.dshs.state.tx.us/coronavirus/TexasCOVID19DailyCountyFatalityCountData.xlsx"

#get data from 2020, 2021, and 2022: could be cleaned more
df_2020 = pd.read_excel(url,sheet_name=0, index_col=0,parse_dates=[0])
df_2020 = df_2020[1:]
df_2020=df_2020.T
df_2020=df_2020[["County", "Hays", "Bastrop", "Caldwell", "Travis", "Williamson"]]

df_2021 = pd.read_excel(url,sheet_name=1, index_col=0,parse_dates=[0])
df_2021 = df_2021[1:]
df_2021=df_2021.T
df_2021=df_2021[["County", "Hays", "Bastrop", "Caldwell", "Travis", "Williamson"]]


df_2022 = pd.read_excel(url,sheet_name=2, index_col=0,parse_dates=[0])
df_2022 = df_2022[1:]
df_2022=df_2022.T
df_2022=df_2022[["County", "Hays", "Bastrop", "Caldwell", "Travis", "Williamson"]]

#df_all has all of the data for 5 counties from 2020-2022
df_all=df_2020.append(df_2021)
df_all=df_all.append(df_2022)

dict = {'County' : 'date'}
df_all.rename(columns=dict,
          inplace=True)

df_all['date'] = pd.to_datetime(df_all.date, format='%m/%d/%Y')
df_all["date"] = df_all["date"].dt.strftime("%m/%d/%y")

#This converts an accumulated data to a daily data that doesn't accumulate
df_all['Sum'] = df_all['Hays'] + df_all['Bastrop'] + df_all['Caldwell'] + df_all['Travis'] + df_all['Williamson']
df_all=df_all[['date','Sum']]
df_all["hospitalized"] = df_all["Sum"] - df_all["Sum"].shift(1)
df_all["hospitalized"][0]=0
df_all=df_all[["date", "hospitalized"]]

#attaches February of 2020 to the dataframe
instances_path = Path(__file__).parent
path1=path = instances_path.parent / 'instances/austin/austin_real_total_death.csv'


df_init=pd.read_csv(path1, index_col=None)
df_init=df_init[1:18]
df_init=df_init[["date", "hospitalized"]]
df_init=df_init.append(df_all)

path2 = instances_path.parent / 'instances/austin/scraped_death_data.csv'

df_init['date'] = pd.to_datetime(df_init['date'])

df_init.to_csv(path2, index=False, date_format='%Y-%m-%d')

