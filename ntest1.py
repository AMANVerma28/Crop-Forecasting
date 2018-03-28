from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 # load dataset
dataset = read_csv('ntest1.csv', header=0, index_col=0)
dataset.drop(['Exports','Imports','Beginning Stocks'],1,inplace=True)
(a)=len(dataset)-1
values = dataset.values

df1 = dataset[['Production']]

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)

	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

reframed = series_to_supervised(scaled, 1, 1)
print(scaled)
print(reframed)
# drop columns we don't want to predict

#print(reframed.head())
 
# split into train and test sets
values = reframed.values
n_train_hours = 40
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=9, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='test')
pyplot.plot(history.history['val_loss'], label='train')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
chte=100000*inv_yhat
new1 = [[] for i in range(n_train_hours)] 

for i in range(n_train_hours):
	new1[i].append([1960+i])
	new1[i].append([train_y[i]*100000])

new = [[] for i in range(a-n_train_hours)] 

for i in range(a-n_train_hours):
	new[i].append([1960+1+n_train_hours+i])
	new[i].append([chte[i]])

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
plt.rcParams['figure.figsize'] = [15,10]
plt.plot(df1,'--bo',label='Actual')
for i in range(n_train_hours):
	plt.plot(new1[i][0],new1[i][1],'--ro')
for i in range(a-n_train_hours):
	plt.plot(new[i][0],new[i][1],'--yo')
plt.xlabel('Year')
plt.ylabel('Wheat Production(in 1000MT)')
plt.title('Predicting Wheat Production Using LSTM')
plt.legend()
plt.show()
