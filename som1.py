# -*- coding: utf-8 -*-
"""
Created on Sun May 10 03:49:05 2020

@author: kingslayer
"""

######## SELF ORGANISING MAPS ##########


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset
dataset=pd.read_csv("Credit_Card_Applications.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X=sc.fit_transform(X)

#Training the SOM
from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15)
som.random_weights_init(X)
som.train_random(X,num_iteration=100)


#Visualisation
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=["o","s"]
colors=["r","g"]
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],markerfacecolor="None",markersize=10,markeredgewidth=2)
show()

#Finding fraud
mapping=som.win_map(X)
frauds=np.concatenate((mapping[(1,1)],mapping[(5,4)]),axis=0) #index to be changed as per requirement
frauds=sc.inverse_transform(frauds)

frauds_to_investigate=[]
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        if dataset.iloc[i,-1]==1:
            frauds_to_investigate.append(dataset.iloc[i,0])
