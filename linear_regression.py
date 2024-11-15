# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:37:09 2024

@author: dobme
"""
"""
the initial ideas came from reading this paper:
https://www.researchgate.net/publication/382638380_A_Comprehensive_Analysis_of_Machine_Learning_Models_for_Algorithmic_Trading_of_Bitcoin

after recreating some of the concepts from the paper i went on to carry out my own analysis\upgrades
"""

"""
the plan:
    
1) create the test values
2) get all indicators needed
3) apply any necessary transformations
4) train model on rolling periods of 1, 3, 7, 14, 21, 28, 45, 60, 80, 150, 365 and 720
"""

"""
an important thing to note is that in the testing my second accuracy metric is if
the model predicted a higher price than the current days close and the next day
closed higher this would be considered a success. vice versa for bear days. the
reasoning for this is that it is a slightly creative way of generating predictions
which therefore means there might be alpha in it
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from ta.volume import AccDistIndexIndicator, MFIIndicator
from ta.volatility import KeltnerChannel, BollingerBands 
from ta.trend import PSARIndicator
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import quantstats as qs

df = pd.read_csv('1BTC-1d-520wks-data.csv') # read in data

# creating results value
df['result'] = df['close']
        
df['result'] = df['result'].shift(-1)
df = df.drop(df.index[-1])
        
# first indicator - accumulation/distribution index
ad_index = AccDistIndexIndicator(high = df['high'], low = df['low'], close = df['close'], volume = df['volume'])
df['accum_dist_index'] = ad_index.acc_dist_index()

# second indicator - money flow index
mfi_indicator = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
df['mfi'] = mfi_indicator.money_flow_index()

# bollinger bands
bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
df['bollinger_mavg'] = bollinger.bollinger_mavg()
df['bollinger_upper'] = bollinger.bollinger_hband()
df['bollinger_lower'] = bollinger.bollinger_lband()

# keltner channel width
keltner = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20, window_atr=10)
df['keltner_bandwidth'] = keltner.keltner_channel_wband()

# parabolic sar
psar = PSARIndicator(high=df['high'], low=df['low'], close=df['close'], step=0.02, max_step=0.2)
df['psar'] = psar.psar()

plt.figure(figsize=(14, 12))

# Plot Close Price with Bollinger Bands and Parabolic SAR
plt.subplot(3, 1, 1)
plt.plot(df['close'], label='Close Price', color='black', linewidth=1)
plt.plot(df['bollinger_mavg'], label='Bollinger Middle Band', color='blue', linestyle='--')
plt.plot(df['bollinger_upper'], label='Bollinger Upper Band', color='green', linestyle='--')
plt.plot(df['bollinger_lower'], label='Bollinger Lower Band', color='red', linestyle='--')
plt.fill_between(df.index, df['bollinger_upper'], df['bollinger_lower'], color='gray', alpha=0.2, label='Bollinger Bands')
plt.scatter(df.index, df['psar'], color='purple', marker='.', label='Parabolic SAR', s=10)

plt.title('Close Price with Bollinger Bands and Parabolic SAR')
plt.xlabel('Period')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.grid(True)

# Plot Accumulation/Distribution Index
plt.subplot(3, 1, 2)
plt.plot(df['accum_dist_index'], label='Accumulation/Distribution Index', color='brown', linewidth=1.5)
plt.title('Accumulation/Distribution Index')
plt.xlabel('Period')
plt.ylabel('A/D Index')
plt.legend(loc='upper left')
plt.grid(True)

# Plot Money Flow Index and Keltner Channel Width
plt.subplot(3, 1, 3)
plt.plot(df['mfi'], label='Money Flow Index (MFI)', color='orange', linewidth=1.5)
plt.plot(df['keltner_bandwidth'], label='Keltner Channel Width', color='blue', linestyle='--')
plt.title('Money Flow Index and Keltner Channel Width')
plt.xlabel('Period')
plt.ylabel('Index Value')
plt.legend(loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

"""
it's clear from the plots that price, bollinger bands, parabolic sar and accumulation/distribution index
have rising trend and variance. i will apply difference log to these
"""

df1 = df.copy()

df1 = df1.iloc[20:] # getting rid of nan values from indicators

# log/differencing
df1.loc[ : ,'open'] = (np.log(df1['open'])).diff()
df1.loc[ : ,'high'] = (np.log(df1['high'])).diff()
df1.loc[ : ,'low'] = (np.log(df1['low'])).diff()
df1.loc[ : ,'close'] = (np.log(df1['close'])).diff()
df1.loc[ : ,'volume'] = (np.log(df1['volume'])).diff()
df1.loc[ : ,'accum_dist_index'] = (np.log(df1['accum_dist_index'] + abs(min(df1['accum_dist_index'])) + 0.01)).diff() # here i have added the absolute of the mimimum and a small constant to make all values positive so they can be put though a log transformation
df1.loc[ : ,'bollinger_mavg'] = (np.log(df1['bollinger_mavg'])).diff()
df1.loc[ : ,'bollinger_upper'] = (np.log(df1['bollinger_upper'])).diff()
df1.loc[ : ,'bollinger_lower'] = (np.log(df1['bollinger_lower'])).diff()
df1.loc[ : ,'psar'] = (np.log(df1['psar'])).diff()
df1.loc[ : ,'result'] = (np.log(df1['result'])).diff()

df1 = df1.iloc[21:] # getting rid of first value since differencing sets it to nan
df1.reset_index(drop=True, inplace=True)

#plot everything againn and check for stationary
plt.figure(figsize=(14, 12))

# Plot Close Price with Bollinger Bands and Parabolic SAR
plt.subplot(3, 1, 1)
plt.plot(df1['close'], label='Close Price', color='black', linewidth=1)
plt.plot(df1['bollinger_mavg'], label='Bollinger Middle Band', color='blue', linestyle='--')
plt.plot(df1['bollinger_upper'], label='Bollinger Upper Band', color='green', linestyle='--')
plt.plot(df1['bollinger_lower'], label='Bollinger Lower Band', color='red', linestyle='--')
plt.fill_between(df1.index, df1['bollinger_upper'], df1['bollinger_lower'], color='gray', alpha=0.2, label='Bollinger Bands')
plt.scatter(df1.index, df1['psar'], color='purple', marker='.', label='Parabolic SAR', s=10)

plt.title('Close Price with Bollinger Bands and Parabolic SAR')
plt.xlabel('Period')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.grid(True)

# Plot Accumulation/Distribution Index
plt.subplot(3, 1, 2)
plt.plot(df1['accum_dist_index'], label='Accumulation/Distribution Index', color='brown', linewidth=1.5)
plt.title('Accumulation/Distribution Index')
plt.xlabel('Period')
plt.ylabel('A/D Index')
plt.legend(loc='upper left')
plt.grid(True)

# Plot Money Flow Index and Keltner Channel Width
plt.subplot(3, 1, 3)
plt.plot(df1['mfi'], label='Money Flow Index (MFI)', color='orange', linewidth=1.5)
plt.plot(df1['keltner_bandwidth'], label='Keltner Channel Width', color='blue', linestyle='--')
plt.title('Money Flow Index and Keltner Channel Width')
plt.xlabel('Period')
plt.ylabel('Index Value')
plt.legend(loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

"""
everything looks reasonable now i am more comfortable using this data than the previous data

one thing to note is that the accumulation distribution index gave noticably high values
in some areas (after prep). i am aware of this and unwilling to make adjustments for it because perfect data
is not something i know i can have, i will leave it as it is. if it is something that affects the performance
of the model i will be aware of the data before i take a trade in future so i can always just simply
not take the trade if the training data contains similar spikes.
"""

"""
testing procedure:

1) create rolling windows
2) train model on rolling windows to find best parameters
3) use best parameters on the next (test) day and predict result
4) store predicted result vs actual result
5) compare all rolling windows
"""

#create rolling periods of 1, 3, 7, 14, 21, 28, 45, 60, 80, 150, 365 and 720
df2 = df1.copy()

#drop variables with high corrolation
df2 = df2.drop(columns = 'bollinger_mavg')

oldvalue = 0 # train loop start
testvalue = 1 # train loop end test day start
rp1_accuracy = []
rp1_accuracydirection = []
rp1_testdates = []

model = LinearRegression() # model used

for value in range(len(df2)-2): # -2 because -1 is test window and final is test value
    
    rolling_period = df2.iloc[oldvalue:testvalue] # rolling train window
    test_period = df2.iloc[testvalue] # rolling test window
    
    x_train = rolling_period.drop(columns = ['datetime', 'result']) # train parameters for model
    y_train = rolling_period['result'] # prediction for model
    
    model.fit(x_train, y_train) # fitting model
    
    x_test = test_period.drop(['datetime', 'result']) # test parameters
    x_test = x_test.to_frame().T
    y_test = test_period['result'] # test prediction
    
    y_pred = model.predict(x_test)[0] # actual prediction from test
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp1_accuracy.append('right') # direction + price right
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp1_accuracy.append('right') # direction + price right
    
    else:
        rp1_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp1_accuracydirection.append('right') # direction right
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp1_accuracydirection.append('right') # direction right
    
    else:
        rp1_accuracydirection.append('wrong')
        
    rp1_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1 # updating rolling train window
    testvalue = testvalue + 1 # updating rolling test window

rp1_accuracy_overall = (rp1_accuracy.count('right'))/len(rp1_accuracy) # accuracy of direction + price prediction
rp1_accuracydirection_overall = (rp1_accuracydirection.count('right'))/len(rp1_accuracydirection) # accuracy of direction prediction

oldvalue = 0
testvalue = 3
rp3_accuracy = []
rp3_accuracydirection = []
rp3_testdates = []

model = LinearRegression()

for value in range(len(df2)-4):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp3_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp3_accuracy.append('right')
    
    else:
        rp3_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp3_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp3_accuracydirection.append('right')
    
    else:
        rp3_accuracydirection.append('wrong')
        
    rp3_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp3_accuracy_overall = (rp3_accuracy.count('right'))/len(rp3_accuracy)
rp3_accuracydirection_overall = (rp3_accuracydirection.count('right'))/len(rp3_accuracydirection)

oldvalue = 0
testvalue = 7
rp7_accuracy = []
rp7_accuracydirection = []
rp7_testdates = []

model = LinearRegression()

for value in range(len(df2)-8):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp7_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp7_accuracy.append('right')
    
    else:
        rp7_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp7_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp7_accuracydirection.append('right')
    
    else:
        rp7_accuracydirection.append('wrong')
   
    rp7_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
   
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp7_accuracy_overall = (rp7_accuracy.count('right'))/len(rp7_accuracy)
rp7_accuracydirection_overall = (rp7_accuracydirection.count('right'))/len(rp7_accuracydirection)
    
oldvalue = 0
testvalue = 14
rp14_accuracy = []
rp14_accuracydirection = []
rp14_testdates = []

model = LinearRegression()

for value in range(len(df2)-15):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp14_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp14_accuracy.append('right')
    
    else:
        rp14_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp14_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp14_accuracydirection.append('right')
    
    else:
        rp14_accuracydirection.append('wrong')
    
    rp14_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp14_accuracy_overall = (rp14_accuracy.count('right'))/len(rp14_accuracy)
rp14_accuracydirection_overall = (rp14_accuracydirection.count('right'))/len(rp14_accuracydirection)


oldvalue = 0
testvalue = 21
rp21_accuracy = []
rp21_accuracydirection = []
rp21_testdates = []

model = LinearRegression()

for value in range(len(df2)-22):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp21_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp21_accuracy.append('right')
    
    else:
        rp21_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp21_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp21_accuracydirection.append('right')
    
    else:
        rp21_accuracydirection.append('wrong')
        
    rp21_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp21_accuracy_overall = (rp21_accuracy.count('right'))/len(rp21_accuracy)
rp21_accuracydirection_overall = (rp21_accuracydirection.count('right'))/len(rp21_accuracydirection)

oldvalue = 0
testvalue = 28
rp28_accuracy = []
rp28_accuracydirection = []
rp28_testdates = []

model = LinearRegression()

for value in range(len(df2)-29):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp28_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp28_accuracy.append('right')
    
    else:
        rp28_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp28_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp28_accuracydirection.append('right')
    
    else:
        rp28_accuracydirection.append('wrong')
    
    rp28_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day    

    oldvalue = oldvalue + 1
    testvalue = testvalue + 1


rp28_accuracy_overall = (rp28_accuracy.count('right'))/len(rp28_accuracy)
rp28_accuracydirection_overall = (rp28_accuracydirection.count('right'))/len(rp28_accuracydirection)

oldvalue = 0
testvalue = 45
rp45_accuracy = []
rp45_accuracydirection = []
rp45_testdates = []

model = LinearRegression()

for value in range(len(df2)-46):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp45_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp45_accuracy.append('right')
    
    else:
        rp45_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp45_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp45_accuracydirection.append('right')
    
    else:
        rp45_accuracydirection.append('wrong')
    
    rp45_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp45_accuracy_overall = (rp45_accuracy.count('right'))/len(rp45_accuracy)
rp45_accuracydirection_overall = (rp45_accuracydirection.count('right'))/len(rp45_accuracydirection)

oldvalue = 0
testvalue = 60
rp60_accuracy = []
rp60_accuracydirection = []
rp60_testdates = []

model = LinearRegression()

for value in range(len(df2)-61):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp60_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp60_accuracy.append('right')
    
    else:
        rp60_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp60_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp60_accuracydirection.append('right')
    
    else:
        rp60_accuracydirection.append('wrong')
    
    rp60_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day

    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp60_accuracy_overall = (rp60_accuracy.count('right'))/len(rp60_accuracy)
rp60_accuracydirection_overall = (rp60_accuracydirection.count('right'))/len(rp60_accuracydirection)

oldvalue = 0
testvalue = 80
rp80_accuracy = []
rp80_accuracydirection = []
rp80_testdates = []

model = LinearRegression()

for value in range(len(df2)-81):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp80_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp80_accuracy.append('right')
    
    else:
        rp80_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp80_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp80_accuracydirection.append('right')
    
    else:
        rp80_accuracydirection.append('wrong')
    
    rp80_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp80_accuracy_overall = (rp80_accuracy.count('right'))/len(rp80_accuracy)
rp80_accuracydirection_overall = (rp80_accuracydirection.count('right'))/len(rp80_accuracydirection)

oldvalue = 0
testvalue = 150
rp150_accuracy = []
rp150_accuracydirection = []
rp150_testdates = []

model = LinearRegression()

for value in range(len(df2)-151):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp150_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp150_accuracy.append('right')
    
    else:
        rp150_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp150_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp150_accuracydirection.append('right')
    
    else:
        rp150_accuracydirection.append('wrong')
    
    rp150_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp150_accuracy_overall = (rp150_accuracy.count('right'))/len(rp150_accuracy)
rp150_accuracydirection_overall = (rp150_accuracydirection.count('right'))/len(rp150_accuracydirection)

oldvalue = 0
testvalue = 365
rp365_accuracy = []
rp365_accuracydirection = []
rp365_testdates = []

model = LinearRegression()

for value in range(len(df2)-366):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp365_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp365_accuracy.append('right')
    
    else:
        rp365_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp365_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp365_accuracydirection.append('right')
    
    else:
        rp365_accuracydirection.append('wrong')
    
    rp365_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp365_accuracy_overall = (rp365_accuracy.count('right'))/len(rp365_accuracy)
rp365_accuracydirection_overall = (rp365_accuracydirection.count('right'))/len(rp365_accuracydirection)

oldvalue = 0
testvalue = 720
rp720_accuracy = []
rp720_accuracydirection = []
rp720_testdates = []

model = LinearRegression()

for value in range(len(df2)-721):
    
    rolling_period = df2.iloc[oldvalue:testvalue]
    test_period = df2.iloc[testvalue]
    
    x_train = rolling_period.drop(columns = ['datetime', 'result'])
    y_train = rolling_period['result']
    
    model.fit(x_train, y_train)
    
    x_test = test_period.drop(['datetime', 'result'])
    x_test = x_test.to_frame().T
    y_test = test_period['result']
    
    y_pred = model.predict(x_test)[0] 
    
    if y_test > test_period['close'] and y_pred > test_period['close'] and y_pred <= y_test:
        rp720_accuracy.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close'] and y_pred >= y_test:
        rp720_accuracy.append('right')
    
    else:
        rp720_accuracy.append('wrong')
        
    if y_test > test_period['close'] and y_pred > test_period['close']:
        rp720_accuracydirection.append('right')
        
    elif y_test < test_period['close'] and y_pred < test_period['close']:
        rp720_accuracydirection.append('right')
    
    else:
        rp720_accuracydirection.append('wrong')
    
    rp720_testdates.append(df2.iloc[testvalue + 1]['datetime']) # plus one because the predicted close is the day after the test day
    
    oldvalue = oldvalue + 1
    testvalue = testvalue + 1

rp720_accuracy_overall = (rp720_accuracy.count('right'))/len(rp720_accuracy)
rp720_accuracydirection_overall = (rp720_accuracydirection.count('right'))/len(rp720_accuracydirection)

print(rp1_accuracy_overall)
print(rp1_accuracydirection_overall)
print(rp3_accuracy_overall)
print(rp3_accuracydirection_overall)
print(rp7_accuracy_overall)
print(rp7_accuracydirection_overall)
print(rp14_accuracy_overall)
print(rp14_accuracydirection_overall)
print(rp21_accuracy_overall)
print(rp21_accuracydirection_overall)
print(rp28_accuracy_overall)
print(rp28_accuracydirection_overall)
print(rp45_accuracy_overall)
print(rp45_accuracydirection_overall)
print(rp60_accuracy_overall)
print(rp60_accuracydirection_overall)
print(rp80_accuracy_overall)
print(rp80_accuracydirection_overall)
print(rp150_accuracy_overall)
print(rp150_accuracydirection_overall)
print(rp365_accuracy_overall)
print(rp365_accuracydirection_overall)
print(rp720_accuracy_overall)
print(rp720_accuracydirection_overall)

# i just built this function to check if there is severe drawdowns 
def max_consecutive_occurrences(lst, target):
    max_count = 0
    current_count = 0

    for item in lst:
        if item == target:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0  # Reset count when a non-target item is found

    return max_count

max_consecutive_occurrences(rp80_accuracydirection, 'wrong')

"""
the next step is to check accuracy vs volatility. the model seems
reasonably accurate so far but it also needs to be useable. what this means is
that there needs to be enough volatility there when the model is correct that
profit can be made
"""

# making a dataframe that has dates and volatility

volatility_list = []
direction_list = []
date_list = []

for index in df.index[1:]:
    
    v = (abs(df.iloc[index]['close'] - df.iloc[index - 1]['close']))/(df.iloc[index - 1]['close']) # getting volatility
    
    if df.iloc[index]['open'] > df.iloc[index]['close']: # getting direction
        direction_list.append('bear')
        
    elif df.iloc[index]['close'] > df.iloc[index]['open']: # getting direction
        direction_list.append('bull')
        
    else:
        direction_list.append('neutral')
        
    volatility_list.append(v) # adding volatility to list
    date_list.append(df.iloc[index]['datetime']) # adding date to list
    
volatilitydf = pd.DataFrame({'date': date_list, 'volatility': volatility_list, 'direction': direction_list}) # creating dataframe

# make dataframe that has accuracy and dates and combine with volatility
accuracy1_df = pd.DataFrame({'date': rp1_testdates, 'accuracy': rp1_accuracydirection})
accuracy_volatility1 = pd.merge(volatilitydf, accuracy1_df, on='date', how='inner')

accuracy3_df = pd.DataFrame({'date': rp3_testdates, 'accuracy': rp3_accuracydirection})
accuracy_volatility3 = pd.merge(volatilitydf, accuracy3_df, on='date', how='inner')

accuracy7_df = pd.DataFrame({'date': rp7_testdates, 'accuracy': rp7_accuracydirection})
accuracy_volatility7 = pd.merge(volatilitydf, accuracy7_df, on='date', how='inner')

accuracy14_df = pd.DataFrame({'date': rp14_testdates, 'accuracy': rp14_accuracydirection})
accuracy_volatility14 = pd.merge(volatilitydf, accuracy14_df, on='date', how='inner')

accuracy21_df = pd.DataFrame({'date': rp21_testdates, 'accuracy': rp21_accuracydirection})
accuracy_volatility21 = pd.merge(volatilitydf, accuracy21_df, on='date', how='inner')

accuracy28_df = pd.DataFrame({'date': rp28_testdates, 'accuracy': rp28_accuracydirection})
accuracy_volatility28 = pd.merge(volatilitydf, accuracy28_df, on='date', how='inner')

accuracy45_df = pd.DataFrame({'date': rp45_testdates, 'accuracy': rp45_accuracydirection})
accuracy_volatility45 = pd.merge(volatilitydf, accuracy45_df, on='date', how='inner')

accuracy60_df = pd.DataFrame({'date': rp60_testdates, 'accuracy': rp60_accuracydirection})
accuracy_volatility60 = pd.merge(volatilitydf, accuracy60_df, on='date', how='inner')

accuracy80_df = pd.DataFrame({'date': rp80_testdates, 'accuracy': rp80_accuracydirection})
accuracy_volatility80 = pd.merge(volatilitydf, accuracy80_df, on='date', how='inner')

accuracy150_df = pd.DataFrame({'date': rp150_testdates, 'accuracy': rp150_accuracydirection})
accuracy_volatility150 = pd.merge(volatilitydf, accuracy150_df, on='date', how='inner')

accuracy365_df = pd.DataFrame({'date': rp365_testdates, 'accuracy': rp365_accuracydirection})
accuracy_volatility365 = pd.merge(volatilitydf, accuracy365_df, on='date', how='inner')

accuracy720_df = pd.DataFrame({'date': rp720_testdates, 'accuracy': rp720_accuracydirection})
accuracy_volatility720 = pd.merge(volatilitydf, accuracy720_df, on='date', how='inner')

# compute expected values
timeframes = {
    # '1': accuracy_volatility1,
    # '3': accuracy_volatility3,
    # '7': accuracy_volatility7,
    # '14': accuracy_volatility14,
    # '21': accuracy_volatility21,
    '28': accuracy_volatility28,
    '45': accuracy_volatility45,
    '60': accuracy_volatility60,
    '80': accuracy_volatility80,
    '150': accuracy_volatility150,
    '365': accuracy_volatility365,
    '720': accuracy_volatility720
} # have not included the first few as they are not worth it

results = {}
acc = {}
ev = {}

for period, dfx in timeframes.items():

    bull_right = dfx[(dfx['accuracy'] == 'right') & (dfx['direction'] == 'bull')]['volatility'].mean() # average return of being right on bull days
    bear_right = dfx[(dfx['accuracy'] == 'right') & (dfx['direction'] == 'bear')]['volatility'].mean() # average return of being right on bear days
    bull_wrong = dfx[(dfx['accuracy'] == 'wrong') & (dfx['direction'] == 'bull')]['volatility'].mean() # average return of being wrong on bull days
    bear_wrong = dfx[(dfx['accuracy'] == 'wrong') & (dfx['direction'] == 'bear')]['volatility'].mean() # average return of being wrong on bear days
    
    bull_right_acc = len(dfx[(dfx['accuracy'] == 'right') & (dfx['direction'] == 'bull')])/(len(dfx[(dfx['accuracy'] == 'right') & (dfx['direction'] == 'bull')]) + len(dfx[(dfx['accuracy'] == 'wrong') & (dfx['direction'] == 'bear')])) # overall accuracy on bull days - right/(right+wrong)
    bear_right_acc = len(dfx[(dfx['accuracy'] == 'right') & (dfx['direction'] == 'bear')])/(len(dfx[(dfx['accuracy'] == 'right') & (dfx['direction'] == 'bear')]) + len(dfx[(dfx['accuracy'] == 'wrong') & (dfx['direction'] == 'bull')])) # overall accuracy on bear days - right/(right+wrong)

    # Store results in the dictionary
    results[period] = {
        'bull_right': bull_right,
        'bear_right': bear_right,
        'bull_wrong': bull_wrong,
        'bear_wrong': bear_wrong
    }
    
    acc[period] = {
        'bull_right_acc': bull_right_acc,
        'bear_right_acc': bear_right_acc
    }
    
    ev[period] = {
        'bull_ev': (acc[period]['bull_right_acc'])*(results[period]['bull_right']) + (1 - acc[period]['bull_right_acc'])*(-1*(results[period]['bull_wrong'])),
        'bear_ev': (acc[period]['bear_right_acc'])*(results[period]['bear_right']) + (1 - acc[period]['bear_right_acc'])*(-1*(results[period]['bear_wrong']))
    } # ev values

# getting profit and loss
pnl28 = accuracy_volatility28.copy()

for index in pnl28.index:
    if pnl28['accuracy'].iloc[index] == 'wrong':
        
        pnl28.loc[index, 'volatility'] = pnl28.loc[index, 'volatility'] * -1 # changing payoff when wrong to negative
        
pnl45 = accuracy_volatility45.copy()

for index in pnl45.index:
    if pnl45['accuracy'].iloc[index] == 'wrong':
        
        pnl45.loc[index, 'volatility'] = pnl45.loc[index, 'volatility'] * -1
        
pnl60 = accuracy_volatility60.copy()

for index in pnl60.index:
    if pnl60['accuracy'].iloc[index] == 'wrong':
        
        pnl60.loc[index, 'volatility'] = pnl60.loc[index, 'volatility'] * -1
        
pnl80 = accuracy_volatility80.copy()

for index in pnl80.index:
    if pnl80['accuracy'].iloc[index] == 'wrong':
        
        pnl80.loc[index, 'volatility'] = pnl80.loc[index, 'volatility'] * -1

pnl150 = accuracy_volatility150.copy()

for index in pnl150.index:
    if pnl150['accuracy'].iloc[index] == 'wrong':
        
        pnl150.loc[index, 'volatility'] = pnl150.loc[index, 'volatility'] * -1

pnl365 = accuracy_volatility365.copy()

for index in pnl365.index:
    if pnl365['accuracy'].iloc[index] == 'wrong':
        
        pnl365.loc[index, 'volatility'] = pnl365.loc[index, 'volatility'] * -1

pnl720 = accuracy_volatility720.copy()

for index in pnl720.index:
    if pnl720['accuracy'].iloc[index] == 'wrong':
        
        pnl720.loc[index, 'volatility'] = pnl720.loc[index, 'volatility'] * -1

pnl28 = pnl28.set_index('date')
pnl28.index = pd.to_datetime(pnl28.index) # setting dates as index as quantstats wants it this way

pnl45 = pnl45.set_index('date')
pnl45.index = pd.to_datetime(pnl45.index)

pnl60 = pnl60.set_index('date')
pnl60.index = pd.to_datetime(pnl60.index)

pnl80 = pnl80.set_index('date')
pnl80.index = pd.to_datetime(pnl80.index)

pnl150 = pnl150.set_index('date')
pnl150.index = pd.to_datetime(pnl150.index)

pnl365 = pnl365.set_index('date')
pnl365.index = pd.to_datetime(pnl365.index)

pnl720 = pnl720.set_index('date')
pnl720.index = pd.to_datetime(pnl720.index)

# all returns
results28 = qs.reports.metrics(pnl28['volatility'])
results45 = qs.reports.metrics(pnl45['volatility'])
results60 = qs.reports.metrics(pnl60['volatility'])
results80 = qs.reports.metrics(pnl80['volatility'])
results150 = qs.reports.metrics(pnl150['volatility'])
results365 = qs.reports.metrics(pnl365['volatility'])
results720 = qs.reports.metrics(pnl720['volatility'])

"""
large returns but drawdown periods are too long. i will now check single direction trades
and implement stop losses
"""

"""
the following is single direction trades
"""

# making long only dataframes
pnl28_longonly = pnl28[(pnl28['direction'] == 'bull') & (pnl28['accuracy'] == 'right') | (pnl28['direction'] == 'bear') & (pnl28['accuracy'] == 'wrong')]
pnl45_longonly = pnl45[(pnl45['direction'] == 'bull') & (pnl45['accuracy'] == 'right') | (pnl45['direction'] == 'bear') & (pnl45['accuracy'] == 'wrong')]
pnl60_longonly = pnl60[(pnl60['direction'] == 'bull') & (pnl60['accuracy'] == 'right') | (pnl60['direction'] == 'bear') & (pnl60['accuracy'] == 'wrong')]
pnl80_longonly = pnl80[(pnl80['direction'] == 'bull') & (pnl80['accuracy'] == 'right') | (pnl80['direction'] == 'bear') & (pnl80['accuracy'] == 'wrong')]
pnl150_longonly = pnl150[(pnl150['direction'] == 'bull') & (pnl150['accuracy'] == 'right') | (pnl150['direction'] == 'bear') & (pnl150['accuracy'] == 'wrong')]
pnl365_longonly = pnl365[(pnl365['direction'] == 'bull') & (pnl365['accuracy'] == 'right') | (pnl365['direction'] == 'bear') & (pnl365['accuracy'] == 'wrong')]
pnl720_longonly = pnl720[(pnl720['direction'] == 'bull') & (pnl720['accuracy'] == 'right') | (pnl720['direction'] == 'bear') & (pnl720['accuracy'] == 'wrong')]

# long only results
longonly_results28 = qs.reports.metrics(pnl28_longonly['volatility'])
longonly_results45 = qs.reports.metrics(pnl45_longonly['volatility'])
longonly_results60 = qs.reports.metrics(pnl60_longonly['volatility'])
longonly_results80 = qs.reports.metrics(pnl80_longonly['volatility'])
longonly_results150 = qs.reports.metrics(pnl150_longonly['volatility'])
longonly_results365 = qs.reports.metrics(pnl365_longonly['volatility'])
longonly_results720 = qs.reports.metrics(pnl720_longonly['volatility'])

# making short only dataframes
pnl28_shortonly = pnl28[(pnl28['direction'] == 'bear') & (pnl28['accuracy'] == 'right') | (pnl28['direction'] == 'bull') & (pnl28['accuracy'] == 'wrong')]
pnl45_shortonly = pnl45[(pnl45['direction'] == 'bear') & (pnl45['accuracy'] == 'right') | (pnl45['direction'] == 'bull') & (pnl45['accuracy'] == 'wrong')]
pnl60_shortonly = pnl60[(pnl60['direction'] == 'bear') & (pnl60['accuracy'] == 'right') | (pnl60['direction'] == 'bull') & (pnl60['accuracy'] == 'wrong')]
pnl80_shortonly = pnl80[(pnl80['direction'] == 'bear') & (pnl80['accuracy'] == 'right') | (pnl80['direction'] == 'bull') & (pnl80['accuracy'] == 'wrong')]
pnl150_shortonly = pnl150[(pnl150['direction'] == 'bear') & (pnl150['accuracy'] == 'right') | (pnl150['direction'] == 'bull') & (pnl150['accuracy'] == 'wrong')]
pnl365_shortonly = pnl365[(pnl365['direction'] == 'bear') & (pnl365['accuracy'] == 'right') | (pnl365['direction'] == 'bull') & (pnl365['accuracy'] == 'wrong')]
pnl720_shortonly = pnl720[(pnl720['direction'] == 'bear') & (pnl720['accuracy'] == 'right') | (pnl720['direction'] == 'bull') & (pnl720['accuracy'] == 'wrong')]

# short only results
shortonly_results28 = qs.reports.metrics(pnl28_shortonly['volatility'])
shortonly_results45 = qs.reports.metrics(pnl45_shortonly['volatility'])
shortonly_results60 = qs.reports.metrics(pnl60_shortonly['volatility'])
shortonly_results80 = qs.reports.metrics(pnl80_shortonly['volatility'])
shortonly_results150 = qs.reports.metrics(pnl150_shortonly['volatility'])
shortonly_results365 = qs.reports.metrics(pnl365_shortonly['volatility'])
shortonly_results720 = qs.reports.metrics(pnl720_shortonly['volatility'])

"""
the following is implementing stop losses

i am going to use rolling windows here againn. the purpose of this is to capture
distributions of how low price goes on positive days and vice versa on negative days.
this will tell me where i can leave the position knowing theres a low probability my
prediction is right
"""
    
bull_low_list = []
bull_high_list = []
bear_low_list = []
bear_high_list = []
date1_list = []

dfstat = df.copy()
dfstat = dfstat.set_index('datetime') # setting index to dates
dfstat.index = pd.to_datetime(dfstat.index) # putting dates in datetime format
volatilitydf2 = volatilitydf.copy()
volatilitydf2 = volatilitydf2.set_index('date') # setting index to dates
volatilitydf2.index = pd.to_datetime(volatilitydf2.index) # putting dates in datetime format

for index in volatilitydf2.index[1:-1]:
    
    idx = dfstat.index.get_loc(index) # getting index location (number form) for future use
    vidx = volatilitydf2.index.get_loc(index)
    
    if volatilitydf2['direction'].iloc[vidx] == 'bull':
        bull_low_list.append((abs(dfstat.iloc[idx]['low'] - dfstat.iloc[idx - 1]['close']))/(dfstat.iloc[idx - 1]['close'])) # bull low as a fraction: (low - yesterdays close)/yesterdays close
        bull_high_list.append((abs(dfstat.iloc[idx]['high'] - dfstat.iloc[idx - 1]['close']))/(dfstat.iloc[idx - 1]['close']))# bull high as a fraction: (high - yesterdays close)/yesterdays close
        bear_low_list.append('bull day')
        bear_high_list.append('bull day')
        
    elif volatilitydf2['direction'].iloc[vidx] == 'bear':
        bear_low_list.append((abs(dfstat.iloc[idx]['low'] - dfstat.iloc[idx - 1]['close']))/(dfstat.iloc[idx - 1]['close'])) # bear low as a fraction: (low - yesterdays close)/yesterdays close
        bear_high_list.append((abs(dfstat.iloc[idx]['high'] - dfstat.iloc[idx - 1]['close']))/(dfstat.iloc[idx - 1]['close'])) # bear high as a fraction: (high - yesterdays close)/yesterdays close
        bull_low_list.append('bear day')
        bull_high_list.append('bear day')
        
    else: # this is just a precaution for if the close price equals the open price
        bear_low_list.append('neutral day')
        bear_high_list.append('neutral day')
        bull_low_list.append('neutral day')
        bull_high_list.append('neutral day')
        
    date1_list.append(df['datetime'].iloc[idx])
    
bull_beardf = pd.DataFrame({'bull low': bull_low_list, 'bull high': bull_high_list,
                            'bear low': bear_low_list, 'bear high': bear_high_list,
                            'date': date1_list}) # combine all to df

bull_beardf = bull_beardf.set_index('date') # set index to date
bull_beardf.index = pd.to_datetime(bull_beardf.index) # save as datetime

# implementing stop losses using 720 day train
pnl720_withsl = pnl720.copy()

for index in pnl720_withsl.index[30:-1]: # given theres a rolling period that learns past distributions starting at the 31 row gives breathing room for this
    
    idx_loc_pnl = pnl720_withsl.index.get_loc(index) # get index location (numeric value) for further use
    idx_loc_bb = bull_beardf.index.get_loc(index) # get index location (numeric value) for further use

    filtered_bull_low = [x for x in bull_beardf[:index]['bull low'] if isinstance(x, float)] # get numeric bull low values
    rollingbull_low = filtered_bull_low[-20:] #take last 20 values
    rollingbull_low = sorted(rollingbull_low) # sort smallest to biggest
    cutoff_index_bull_low = int(len(rollingbull_low) * 0.8) 
    first_80_bull_low = rollingbull_low[:cutoff_index_bull_low] # get first 80% (cut out outliers)
    bulllow_sl = max(first_80_bull_low) # get max value of first 80%
    
    #next three same structure as above
    filtered_bull_high = [x for x in bull_beardf[:index]['bull high'] if isinstance(x, float)]
    rollingbull_high = filtered_bull_high[-20:]
    rollingbull_high = sorted(rollingbull_high)
    cutoff_index_bull_high = int(len(rollingbull_high) * 0.8)
    first_80_bull_high = rollingbull_high[:cutoff_index_bull_high]
    bullhigh_sl = max(first_80_bull_high)
    
    filtered_bear_low = [x for x in bull_beardf[:index]['bear low'] if isinstance(x, float)]
    rollingbear_low = filtered_bear_low[-20:]
    rollingbear_low = sorted(rollingbear_low)
    cutoff_index_bear_low = int(len(rollingbear_low) * 0.8)
    first_80_bear_low = rollingbear_low[:cutoff_index_bear_low]
    bearlow_sl = max(first_80_bear_low)
    
    filtered_bear_high = [x for x in bull_beardf[:index]['bear high'] if isinstance(x, float)]
    rollingbear_high = filtered_bear_high[-20:]
    rollingbear_high = sorted(rollingbear_high)
    cutoff_index_bear_high = int(len(rollingbear_high) * 0.8)
    first_80_bear_high = rollingbear_high[:cutoff_index_bear_high]
    bearhigh_sl = max(first_80_bear_high)
    
    # on days when stop loss exceeded, changing the pnl to the stop loss
    # note below all values changed to a fixed stop of -3% because the rolling values was tried and was not as successful as a fixed stop
    if pnl720_withsl.iloc[idx_loc_pnl]['accuracy'] == 'right' and pnl720_withsl.iloc[idx_loc_pnl]['direction'] == 'bull':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bull low']) > 0.03: #bulllow_sl: # gone long and low greater than -0.03 implement stop loss, next three same process under thier individual condition
            pnl720_withsl.loc[index, 'volatility'] = -0.03 #bulllow_sl*-1
            
    if pnl720_withsl.iloc[idx_loc_pnl]['accuracy'] == 'right' and pnl720_withsl.iloc[idx_loc_pnl]['direction'] == 'bear':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bear high']) > 0.03: #bearhigh_sl:
            pnl720_withsl.loc[index, 'volatility'] = -0.03 #bearhigh_sl*-1

    if pnl720_withsl.iloc[idx_loc_pnl]['accuracy'] == 'wrong' and pnl720_withsl.iloc[idx_loc_pnl]['direction'] == 'bull':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bull high']) > 0.03: #bullhigh_sl:
            pnl720_withsl.loc[index, 'volatility'] = -0.03 #bullhigh_sl*-1
            
    if pnl720_withsl.iloc[idx_loc_pnl]['accuracy'] == 'wrong' and pnl720_withsl.iloc[idx_loc_pnl]['direction'] == 'bear':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bear low']) > 0.03: #bearlow_sl:
            pnl720_withsl.loc[index, 'volatility'] = -0.03 #bearlow_sl*-1
            
# implementing stop losses using 365 day train
pnl365_withsl = pnl365.copy()

for index in pnl365_withsl.index[30:-1]:
    
    idx_loc_pnl = pnl365_withsl.index.get_loc(index)
    idx_loc_bb = bull_beardf.index.get_loc(index)
    
    if pnl365_withsl.iloc[idx_loc_pnl]['accuracy'] == 'right' and pnl365_withsl.iloc[idx_loc_pnl]['direction'] == 'bull':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bull low']) > 0.03: #bulllow_sl:
            pnl365_withsl.loc[index, 'volatility'] = -0.03 #bulllow_sl*-1
            
    if pnl365_withsl.iloc[idx_loc_pnl]['accuracy'] == 'right' and pnl365_withsl.iloc[idx_loc_pnl]['direction'] == 'bear':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bear high']) > 0.03: #bearhigh_sl:
            pnl365_withsl.loc[index, 'volatility'] = -0.03 #bearhigh_sl*-1

    if pnl365_withsl.iloc[idx_loc_pnl]['accuracy'] == 'wrong' and pnl365_withsl.iloc[idx_loc_pnl]['direction'] == 'bull':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bull high']) > 0.03: #bullhigh_sl:
            pnl365_withsl.loc[index, 'volatility'] = -0.03 #bullhigh_sl*-1
            
    if pnl365_withsl.iloc[idx_loc_pnl]['accuracy'] == 'wrong' and pnl365_withsl.iloc[idx_loc_pnl]['direction'] == 'bear':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bear low']) > 0.03: #bearlow_sl:
            pnl365_withsl.loc[index, 'volatility'] = -0.03 #bearlow_sl*-1

# implementing stop losses using 150 day train
pnl150_withsl = pnl150.copy()

for index in pnl150_withsl.index[30:-1]:
    
    idx_loc_pnl = pnl150_withsl.index.get_loc(index)
    idx_loc_bb = bull_beardf.index.get_loc(index)
    
    if pnl150_withsl.iloc[idx_loc_pnl]['accuracy'] == 'right' and pnl150_withsl.iloc[idx_loc_pnl]['direction'] == 'bull':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bull low']) > 0.03: #bulllow_sl:
            pnl150_withsl.loc[index, 'volatility'] = -0.03 #bulllow_sl*-1
            
    if pnl150_withsl.iloc[idx_loc_pnl]['accuracy'] == 'right' and pnl150_withsl.iloc[idx_loc_pnl]['direction'] == 'bear':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bear high']) > 0.03: #bearhigh_sl:
            pnl150_withsl.loc[index, 'volatility'] = -0.03 #bearhigh_sl*-1

    if pnl150_withsl.iloc[idx_loc_pnl]['accuracy'] == 'wrong' and pnl150_withsl.iloc[idx_loc_pnl]['direction'] == 'bull':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bull high']) > 0.03: #bullhigh_sl:
            pnl150_withsl.loc[index, 'volatility'] = -0.03 #bullhigh_sl*-1
            
    if pnl150_withsl.iloc[idx_loc_pnl]['accuracy'] == 'wrong' and pnl150_withsl.iloc[idx_loc_pnl]['direction'] == 'bear':
        
        if abs(bull_beardf.iloc[idx_loc_bb]['bear low']) > 0.03: #bearlow_sl:
            pnl150_withsl.loc[index, 'volatility'] = -0.03 #bearlow_sl*-1
            
#show results with stop losses used
qs.reports.metrics(pnl150_withsl['volatility'].iloc[30:])
qs.reports.metrics(pnl365_withsl['volatility'].iloc[30:])
qs.reports.metrics(pnl720_withsl['volatility'].iloc[30:])

"""
to summarise - tried stop placement underneath/above most of the distribution of how far price
went against your trade during a recent period. this did not improve performance. switched
to a hard fixed stop loss of 3% which increased performance and reduced the drawdown time
"""

"""
the next step is to include commissions - i wanted this to be a seperate result
because every exchange has thier own pricing structure so it depends on the individual and
who they have an account with. i will be using phemex. phemex charge different fees dependent
on contract, market maker/taker and spot but the most expensive is 0.1%. i will use this as it is 
absolute worst case scenario (even though the live model will likely be on contracts).
"""

qs.reports.metrics(pnl150_withsl['volatility'].iloc[30:])
qs.reports.metrics(pnl365_withsl['volatility'].iloc[30:])
qs.reports.metrics(pnl720_withsl['volatility'].iloc[30:])

pnl150_withslandcom = pnl150_withsl['volatility'] - 0.001
pnl365_withslandcom = pnl365_withsl['volatility'] - 0.001
pnl720_withslandcom = pnl720_withsl['volatility'] - 0.001

qs.reports.metrics(pnl150_withslandcom.iloc[30:])
qs.reports.metrics(pnl365_withslandcom.iloc[30:])
qs.reports.metrics(pnl720_withslandcom.iloc[30:])

"""
-- chosen model --

firstly, the results of all models are extremely high and of course this will be taken
with apprehension.

the results of the backtests using a stop loss of 3% outperform the backtests where no
stop loss was used.

the chosen model uses a train window of 720 days with stop losses of 3%

this was chosen as it has the lowest drawdown period (164 days) relative to a contending 
sharpe ratio (2.51) and cumulative return (6,760,935.38%)

the backtest ran from 19\09\2017 to 01\11\2024
"""



