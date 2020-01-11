from django.shortcuts import render
from django.http import HttpResponse

import os
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn import datasets
from django.conf import settings
from rest_framework import views
from rest_framework import status

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Routines for linear regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from rest_framework.response import Response

# Create your views here.
def home(request):
    return render(request,'home.html',{'name': 'Sourabh'})
def predictor(request):
    data = np.genfromtxt('app/diabetes-data.csv', delimiter=',')
    features = ['age', 'sex', 'body mass index', 'blood pressure','serum1', 'serum2', 'serum3', 'serum4', 'serum5', 'serum6']
    x = data[:, 0:10]  # predictors
    y = data[:, 10]  # response variable
    regr = RandomForestRegressor(n_estimators = 500, random_state = 2,min_impurity_decrease = 0.3,min_samples_leaf=1,min_weight_fraction_leaf=0.0,verbose=0,oob_score = True) 
    regr.fit(x, y)
    age=request.POST['age']
    sex=request.POST['sex']
    bmi=request.POST['bmi']
    bp=request.POST['bp']
    serum1=request.POST['Serum1']
    serum2=request.POST['Serum2']
    serum3=request.POST['Serum3']
    serum4=request.POST['Serum4']
    serum5=request.POST['Serum5']
    serum6=request.POST['Serum6']
    p=[age,sex,bmi,bp,serum1,serum2,serum3,serum4,serum5,serum6]
    t = np.reshape(p,(1,-1))
    y_prediction = regr.predict(t)
    return render(request,'result.html',{'result':y_prediction})

    
            
    