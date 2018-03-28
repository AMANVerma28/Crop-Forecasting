import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = pd.read_csv('DW crops.csv')
df.set_index('Crop', inplace=True)
crop = df.loc['Rice']
crop.set_index('District_Name', inplace=True)
season = crop.loc['SURGUJA']
season.set_index('Season', inplace=True)
data = season.loc['Kharif     ']
data = data[['Crop_Year', 'Area', 'Production']]

original = pd.DataFrame()
original['Crop_Year'] = data['Crop_Year'].values
original['Production'] = data['Production'].values
# print(original)
# print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')

forecast_col = 'Production'
data.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(data)))
# print("fo",forecast_out)

# print(set(df['Crop']))
# print(data)

data['label'] = data[forecast_col].shift(-forecast_out)

print(data)
print("#######################################################")
X = np.array(data.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# print(X_lately)
data.dropna(inplace=True)
y = np.array(data['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_train, y_train)
forecast_set = clf.predict(X_lately)
print("Forecast set",forecast_set) 
print("Accuracy",accuracy)
print("Forecast out",forecast_out)
print("#######################################################")
data['Forecast'] = np.nan

acc = "Score: " + str(accuracy)
# print(original)

data.set_index('Crop_Year', inplace=True)
original.set_index('Crop_Year', inplace=True)
last_year = data.iloc[-1].name
# last_unix = last_date.timestamp()
# print("last year",last_year)
# one_day = 86400
next_year = last_year

for i in forecast_set:
    next_year += 1
    data.loc[next_year] = [np.nan for _ in range(len(data.columns)-1)] + [i]

# print(data)
original['Production'].plot(style='--ro')
data['Forecast'].plot(style='--bo')
plt.legend(loc=4)
plt.xlabel('Year')
plt.ylabel('Production')
plt.suptitle("Surguja District - Rice Crop - Kharif Season",fontsize = 14)
plt.title(acc)
plt.show()