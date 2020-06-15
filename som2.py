# -*- coding: utf-8 -*-
"""
Created on Sun May 10 04:53:11 2020

@author: kingslayer
"""

#### BUILDING HYBRID DEEP LEARNING MODEL ####

#PART 1(Using SOM)

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
frauds=np.concatenate((mapping[(3,2)],mapping[(4,3)]),axis=0) #Indexing to be changed as per need
frauds=sc.inverse_transform(frauds)

#PART 2(Using Supervised Learning)

customers=dataset.iloc[:,1:].values

isfraud=np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        isfraud[i]=1
        
        
        
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
customers=sc_X.fit_transform(customers)



#building ANN

#importing libraries
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier=Sequential()

#Adding input layer
classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=15))

#Adding hidden layer
classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))

#Adding output layer
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

#Compiling the ANN
classifier.compile(optimizer="adam",metrics=["accuracy"],loss="binary_crossentropy")

#Fitting ANN
classifier.fit(customers,isfraud,batch_size=1,nb_epoch=10)


#Predicting
y_pred=classifier.predict(customers)


y_pred=np.concatenate((dataset.iloc[:,0:1],y_pred),axis=1)
y_pred=y_pred[y_pred[:,1].argsort()]