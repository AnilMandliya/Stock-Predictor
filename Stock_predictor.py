import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import SGD,Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import math

data = pd.read_csv("/content/TTM.csv", index_col='Date', parse_dates=["Date"])

train = data[:'2019'].iloc[:,1:2].values
test = data['2019':].iloc[:,1:2].values

'''
visualization of "High" attribute of the dataset
'''

data["High"][:'2019'].plot(figsize=(16,4), legend=True)
data["High"]["2019":].plot(figsize=(16,4), legend=True)
plt.legend(["Training set (before 2019)", "Test set (from 2019)"])
plt.title("TTM stock prices")
plt.show()
'''
Scaling the data
'''
sc = MinMaxScaler(feature_range=(0,1))
train_scaled = sc.fit_transform(train)


'''
Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
So for each element of training set, we have 60 previous training set elements
'''


x_train = []
y_train = []

for i in range(60,1191):
    x_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)


# reshaping x_train for efficient modelling

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#LSTM architecture


regressor = Sequential()

# add first layer with dropout

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

# add second layer

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# add third layer

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# add fourth layer

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# the output layer

regressor.add(Dense(units=1))



# compiling the LSTM RNN network

regressor.compile(optimizer='Adam', loss='mean_squared_error')

# fit to the training set

regressor.fit(x_train, y_train, epochs=10, batch_size=32)

'''
Generating the test set
'''
dataset_total = pd.concat((data['High'][:'2019'], data['High']['2019':]), axis=0)

inputs = dataset_total[len(dataset_total)-len(test)-60 : ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# preparing x_test

x_test = []
for i in range(60,381):
    x_test.append(inputs[i-60:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape

predicted = regressor.predict(x_test)
predicted = sc.inverse_transform(predicted)

# function which plots ibm stock prices: real and predicted both

def plot_predictions(test, predicted):
    plt.plot(test, color="red", label="real TTM stock price")
    plt.plot(predicted, color="blue", label="predicted stock price")
    plt.title("Tata Motors limited stock price prediction")
    plt.xlabel("time")
    plt.ylabel("IBM stock price")
    plt.legend()
    plt.show()
    
 # visualization

plot_predictions(test, predicted)
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("the root mean squared error is : {}.".format(rmse))
    
 # evaluating the model
return_rmse(test, predicted)
