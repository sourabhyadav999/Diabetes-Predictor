import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Routines for linear regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Set label size for plots
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

data = np.genfromtxt('diabetes-data.csv', delimiter=',')
features = ['age', 'sex', 'body mass index', 'blood pressure',
            'serum1', 'serum2', 'serum3', 'serum4', 'serum5', 'serum6']
x = data[:, 0:10]  # predictors
y = data[:, 10]  # response variable

regr = linear_model.LinearRegression()
regr.fit(x, y)

pickle.dump(regr, open('diabetesmodel.pkl', 'wb'))

model = pickle.load(open('diabetesmodel.pkl', 'rb'))

p = list(map(float,input().split(" ")))
t = np.reshape(p,(1,-1))

y_prediction = regr.predict(t)

print(y_prediction)
