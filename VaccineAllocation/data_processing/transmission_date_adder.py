import pandas as pd

df=pd.read_csv("/Users/kevinli/Documents/GitHub/COVID19-vaccine/VaccineAllocation/instances/austin/transmission_new4.csv")

# df=df[722:]
# df=df.assign(date=df.date.str[0:-1])
# df['date'] = df['date'].astype(str) + '2' 

df['date'] = df['date'].astype('datetime64[ns]')

df['date'] = pd.DataFrame({
    'date':pd.date_range(start='2020-02-15', periods=959)
})

df['date'] = pd.to_datetime(df['date']).dt.date

df.to_csv("/Users/kevinli/Documents/GitHub/COVID19-vaccine/VaccineAllocation/instances/austin/transmission_new4.csv", index=False)
