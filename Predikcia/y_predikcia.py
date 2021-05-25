# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:35:01 2020

@author: kamil
"""
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model


X_public = np.load('X_public.npy',allow_pickle=True)
Y_public = np.load('Y_public.npy',allow_pickle=True)
X_eval = np.load('X_eval.npy',allow_pickle=True)

df=pd.DataFrame(X_public)

le = LabelEncoder()

label_encoder = le.fit(X_public[:,180])
integer_classes = label_encoder.transform(label_encoder.classes_)

for i in range(180,200):
    X_public[:, i ] = label_encoder.transform(X_public[:, i])
    

onhotenc = OneHotEncoder()
onehotarray = np.array(X_public[:, 180]).reshape(-1, 1)
one_hot_encoder = onhotenc.fit(onehotarray)

for i in range(180,200):              # bez tohto mean 280*
    onehotarray = np.array(X_public[:, 180]).reshape(-1, 1)
    ohe_coded=onhotenc.transform(onehotarray).toarray()
    X_public = np.delete(X_public, 180, axis=1) 
    X_public = np.append( X_public, ohe_coded, axis=1)
    
#uprava eval


for i in range(180,200):
    X_eval[:, i ] = label_encoder.transform(X_eval[:, i])

for i in range(180,200): 
    onehotarray = np.array(X_eval[:, 180]).reshape(-1, 1)
    ohe_coded=onhotenc.transform(onehotarray).toarray()
    X_eval = np.delete(X_eval, 180, axis=1) 
    X_eval = np.append( X_eval, ohe_coded, axis=1)



    
X_train, X_test, y_train, y_test = train_test_split(X_public, Y_public, test_size=0.2, random_state=2)#som zmenil random z 5 na 3

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X_train)
X_train = imp_mean.transform(X_train)
X_test = imp_mean.transform(X_test)

standardScaler = StandardScaler() # pri pouziti MaXmin aj AB scaler horšie výsledky 

standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)



regrmodel = linear_model.Ridge()
grid_search_params = {
"max_iter": [500, 200,100 ],
 "alpha" : [2.0 ,3.0,1.0],
 "tol" : [2.0 ,3.0,1.0]
 
}

gridsearch = GridSearchCV(regrmodel, grid_search_params, cv=5)
gridsearch.fit(X_train, y_train)

print(gridsearch.best_params_)
print(gridsearch.best_score_)

print("************************")

regrmodel=linear_model.Ridge(max_iter=500,alpha=2.0,tol=2.0)

regrmodel.fit(X_train, y_train)
y_train_pred = regrmodel.predict(X_train)
y_test_pred = regrmodel.predict(X_test)

skore =r2_score(y_test,y_test_pred)

print(skore)

print("MEAN: TEST: %s   TRAIN: %s" % (mean_absolute_error(y_test, y_test_pred), mean_absolute_error(y_train, y_train_pred)))

#generovanie eval dat

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X_public)
X_public = imp_mean.transform(X_public)
X_eval = imp_mean.transform(X_eval)

standardScaler = StandardScaler()
standardScaler.fit(X_public)
X_public = standardScaler.transform(X_public)
X_eval = standardScaler.transform(X_eval)

regrmodel=linear_model.Ridge(max_iter=500,alpha=3.0,tol=2.0)

regrmodel.fit(X_public, Y_public)
y_eval = regrmodel.predict(X_eval)

np.save("y_predikcia.npy", y_eval)