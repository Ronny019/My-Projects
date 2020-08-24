import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

from collections import defaultdict 

import math

import warnings
warnings.filterwarnings('ignore')

# Importing the training set
dataset = pd.read_csv('datasets_557629_1366084_covid_19_india.csv')
dates = dataset.iloc[:,1]
cases = dataset.iloc[:,-1]

dataset_info = pd.concat([dates,cases],axis = 1)
dataset_info = dataset_info.values

data_dict = defaultdict(list)

for date,case in dataset_info:
    data_dict[date].append(case)

date_arr = []
cases_arr = []

for key,value in data_dict.items():
    date_arr.append(key)
    cases_arr.append(sum(value))

cases_np_arr = np.array(cases_arr)
cases_np_arr = np.reshape(cases_np_arr,(-1,1))

#Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
cases_np_arr_nor = scaler.fit_transform(cases_np_arr)

#spliting the train and test set
train_size = int(len(cases_np_arr_nor)*0.67)
test_size = len(cases_np_arr_nor)-train_size

train, test = cases_np_arr_nor[0:train_size, :], cases_np_arr_nor[train_size: len(cases_np_arr_nor), :]

# create dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

#Reshape dataset X= current time, Y= future time 
look_back= 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#Model Building starts

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (1, look_back)))
#regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
#regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(trainX, trainY, epochs = 200, batch_size = 32)

#make predictions
trainPredict = regressor.predict(trainX)
testPredict = regressor.predict(testX)

# Reverse the predicted value to actual values

trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

trainPredict = np.reshape(trainPredict,(-1,1))
testPredict = np.reshape(testPredict,(-1,1))

trainY = np.reshape(trainY,(-1,1))
testY = np.reshape(testY,(-1,1))

trainY = scaler.inverse_transform(trainY)
testY = scaler.inverse_transform(testY)

## Calculate RMSE
#trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
#print('Train : %.2f RMSE' % (trainScore))

#testScore= math.sqrt(mean_squared_error(testY, testPredict))
#print('Test : %.2f RMSE' % (testScore))

from sklearn.metrics import r2_score, mean_squared_error


print("R2, train :",r2_score(trainY, trainPredict))
print("RMSE, train :",math.sqrt(mean_squared_error(trainY, trainPredict)))

print("R2, test :",r2_score(testY, testPredict))
print("RMSE, test :",math.sqrt(mean_squared_error(testY, testPredict)))

#print("R2, train :",r2_score(trainY, trainPredict))
#R2, train : 0.9999024944129438
#print("RMSE, train :",math.sqrt(mean_squared_error(trainY, trainPredict)))
#RMSE, train : 338.7689117418929
#print("R2, test :",r2_score(testY, testPredict))
#R2, test : 0.9626224955422082
#print("RMSE, test :",math.sqrt(mean_squared_error(testY, testPredict)))
#RMSE, test : 59352.494540911546