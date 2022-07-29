import pandas as pd
from pathlib import Path
instances_path = Path(__file__).parent
file_path = instances_path / "Vaccineovertime_cleaned.xlsx"

df_all=pd.read_excel(file_path)
df_1=df_all.iloc[: , :6]
df_2=df_all.iloc[: , 6:12]
df_3=df_all.iloc[: , 12:18]
df_1['vaccine_time'] = df_1['vaccine_time'].dt.date
df_2['vaccine_time.1'] = df_2['vaccine_time.1'].dt.date
df_3['vaccine_time.2'] = df_3['vaccine_time.2'].dt.date

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import dates
import seaborn as sns

ax=df_1.plot(x='vaccine_time', y='vaccine_amount', figsize=(15, 8), label='first dose')
df_2.plot(x='vaccine_time.1', y='vaccine_amount.1', figsize=(15, 8),ax=ax, label='second dose')
df_3.plot(x='vaccine_time.2', y='vaccine_amount.2', figsize=(15, 8),ax=ax, label='booster dose')
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=20))
plt.title('Proportion of Total Vaccinated Per Dose Over Time')
plt.xlabel('Date')
plt.ylabel('Proportions')
plt.show()

df_12=df_1['vaccine_time']
df_temp=df_all.iloc[: , 18:23]
df_12=pd.concat([df_12,df_temp], axis=1)
ax=df_12.plot(x='vaccine_time', y='comp_total', figsize=(15, 8), label='total proportions')
df_12.plot(x='vaccine_time', y='comp_1', figsize=(15, 8), ax=ax, label='12-15 proportions')
df_12.plot(x='vaccine_time', y='comp_2', figsize=(15, 8), ax=ax, label='16-49 proportions')
df_12.plot(x='vaccine_time', y='comp_3', figsize=(15, 8), ax=ax, label='50-64 proportions')
df_12.plot(x='vaccine_time', y='comp_4', figsize=(15, 8), ax=ax, label='65-79 proportions')
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=20))
plt.title('Proportion of First Dose to Second Dose Over Time')
plt.xlabel('Date')
plt.ylabel('Proportions')
plt.show()

df_test=pd.concat([df_1[['vaccine_time', '16-49']], df_2['16-49.1']], axis=1)
ax=df_test.plot(x='vaccine_time', y='16-49', figsize=(15, 8), label='first dose 16-49')
df_test.plot(x='vaccine_time', y='16-49.1', figsize=(15, 8),ax=ax, label='second dose 16-49')
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=20))
plt.title('Time Series 16-49 age group, no time lag')
plt.xlabel('Date')
plt.ylabel('Proportions')
plt.show()

df_test["16-49.1"]=df_test["16-49.1"].shift(-21)
df_test=df_test[:-21]
ax=df_test.plot(x='vaccine_time', y='16-49', figsize=(15, 8), label='first dose 16-49')
df_test.plot(x='vaccine_time', y='16-49.1', figsize=(15, 8),ax=ax, label='second dose 16-49')
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=20))
plt.title('Time Series 16-49 age group, time lag')
plt.xlabel('Date')
plt.show()

df_test=pd.concat([df_1[['vaccine_time', '50-64']], df_2['50-64.1']], axis=1)
ax=df_test.plot(x='vaccine_time', y='50-64', figsize=(15, 8), label='first dose 50-64')
df_test.plot(x='vaccine_time', y='50-64.1', figsize=(15, 8),ax=ax, label='second dose 50-64')
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=20))
plt.title('Time Series 50-64 age group, no time lag')
plt.xlabel('Date')
plt.show()

df_test["50-64.1"]=df_test["50-64.1"].shift(-21)
df_test=df_test[:-21]
ax=df_test.plot(x='vaccine_time', y='50-64', figsize=(15, 8), label='first dose 50-64')
df_test.plot(x='vaccine_time', y='50-64.1', figsize=(15, 8),ax=ax, label='second dose 50-64')
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=20))
plt.title('Time Series 50-64 age group, time lag')
plt.xlabel('Date')
plt.show()

df_23=df_1['vaccine_time']
df_temp=df_all.iloc[: , 23:28]
df_23=pd.concat([df_23,df_temp], axis=1)
df_23[-27:]
ax=df_23.plot(x='vaccine_time', y='comp2_total', figsize=(15, 8), label='total proportions')
df_23.plot(x='vaccine_time', y='comp2_1', figsize=(15, 8), ax=ax, label='12-15 proportions')
df_23.plot(x='vaccine_time', y='comp2_2', figsize=(15, 8), ax=ax, label='16-49 proportions')
df_23.plot(x='vaccine_time', y='comp2_3', figsize=(15, 8), ax=ax, label='50-64 proportions')
df_23.plot(x='vaccine_time', y='comp2_4', figsize=(15, 8), ax=ax, label='65-79 proportions')
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=2))
plt.title('Proportion of Second Dose to Third Dose Over Time')
plt.xlabel('Date')
plt.ylabel('Proportions')
plt.show()
