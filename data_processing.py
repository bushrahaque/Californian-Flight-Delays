# Preprocessing data as required for input to the ML algorithm

# Load Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
path = 'cali_oct19_flight_data.csv'
data = pd.read_csv(path)

# data retrieved from https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236

# Drop necessary columns first
data = data.iloc[:, 0:-1]  # last unamed column

to_keep = ['DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN',
           'DEP_DELAY_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY',
           'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY',
           'CANCELLED']

data = data[to_keep]
# print(data.head())

# Drop necessary rows second
data = data.loc[data.DEP_DELAY_GROUP.isnull() == False]
data = data.loc[data.CANCELLED == 0]

# Create new DataFrame that will feature input for ML Algo
df = pd.DataFrame()

# **** DISCRETIONALIZE ****

# DAY_OF_WEEK:
df['WEEK_DAY'] = data.DAY_OF_WEEK

# OP_CARRIER:
carrier_list = list(data.OP_CARRIER.unique())

carrier_dict = {}

for x, carrier in enumerate(carrier_list):
    carrier_dict[carrier] = x

df['CARRIER'] = data.OP_CARRIER.map(carrier_dict)

# print('Unique Carriers:', df.CARRIER.unique())


# ORIGIN:
# heuristic based on num flights departing from origin in period
origin_dict = {0: 0, 100: 1, 200: 2, 300: 3, 400: 4, 500: 5,
               600: 6, 700: 7, 800: 8, 900: 9, 1000: 10,
               2000: 11, 3000: 12, 4000: 13, 5000: 14}


def discrete_origin(flight):
    n = len(data.loc[data.ORIGIN == flight.ORIGIN])
    if n < 1000:
        lb = (n // 100) * 100
        return origin_dict[lb]
    elif 1000 <= n < 5000:
        lb = (n // 1000) * 1000
        return origin_dict[lb]
    else:
        return origin_dict[5000]


# not required for CatBoost
# df['ORIGIN'] = data.apply(
#    lambda flight: discrete_origin(flight), axis=1)

df['ORIGIN'] = data.ORIGIN

# print('Unique Origins:', df.ORIGIN.unique())

# DEP_DELAY:
col = data.DEP_DELAY_GROUP.astype(int)

df['DELAY_INTERVAL'] = col

# print('Unique Delay Intervals:', data.DEP_DELAY_GROUP.unique())

# CARRIER_DELAY: don't include, all NaN values

# ALL OTHER DELAYS:
#   only 15509 flights don't have NaN for all delays, assume length of delay = 0
for delay in ['WEATHER_DELAY',
              'NAS_DELAY',
              'SECURITY_DELAY',
              'LATE_AIRCRAFT_DELAY']:
    col = data[delay].fillna(0).astype(int)
    df[delay] = col
    # print('Unique Delay Minutes for', delay + ':', data[delay].unique())

# **** ONE-HOT ENCODING ****
# One hot encode the necessary columns
# all other columns stay continuous and thus no one hot encoding
df_week_day_enc = pd.get_dummies(df.WEEK_DAY,
                                 prefix='WEEK_DAY')

df_carrier_enc = pd.get_dummies(df.CARRIER,
                                prefix='CARRIER')


df_enc = pd.concat([df,
                    df_week_day_enc,
                    df_carrier_enc], axis=1)

df_enc = df_enc.drop(['WEEK_DAY',
                      'CARRIER'], axis=1)

# **** SPLIT DATA ****
# separate features and target
X = df_enc.drop(columns='DELAY_INTERVAL',
                inplace=False)

y = df_enc.DELAY_INTERVAL

# split into train and test
split_factor = 0.2

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=split_factor,
                                                    random_state=42)

for name, data in zip(['X_train', 'X_test', 'y_train', 'y_test'],
                      [X_train, X_test, y_train, y_test]):
    fname = name + '.csv'
    if name in ['y_train', 'y_test']:
        data.to_csv(fname, header='DELAY_INTERVAL', index=False)
    else:
        data.to_csv(fname, index=False)

# end
