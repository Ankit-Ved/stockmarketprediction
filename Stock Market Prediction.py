import quandl
import numpy as np
import matplotlib
import math
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = "LDtyYSs62h3VXc4SxJay"
data = quandl.get("WIKI/GOOG",start_date="2017-02-01", end_date="2017-07-28")

data = data[['Open','High','Low','Close']]

data['HLT'] = (data['High']-data['Close'])/data['Close'] * 100
data['pct_change'] = (data['Close']-data['Open'])/data['Open'] * 100
data = data[['Close','HLT','pct_change']]


future_value = ['Close']
data.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.03*len(data)))
print(forecast_out)
X_1=np.array(data)

X_1=preprocessing.scale(X_1)


data['FutureValues'] = data[future_value].shift(-forecast_out)
data.dropna(inplace=True)

print(data)

X=np.array(data.drop(['FutureValues'],1))
data.dropna(inplace=True)
Y=np.array(data['FutureValues'])
X=preprocessing.scale(X)
X_lately=X_1[-7:]

print(len(X),len(Y))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y, test_size= 0.2)

classifier = LinearRegression()
classifier.fit(X_train,Y_train)
accuracy = classifier.score(X_test,Y_test)
prediction = classifier.predict(X_lately)
print(accuracy, prediction)
