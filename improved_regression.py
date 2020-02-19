# -*- coding: utf-8 -*- python3
"""
Created on Tue Feb 18 16:23:35 2020

@author: Antiochian
"""
from __future__ import print_function  # for backwards compatibility: uses print() also in python2

from keras.models import Sequential  # feed foorward NN
from keras.layers import Dense  # need only fully connected (dense) layers
from keras import optimizers  # for gradient descent
import numpy as np
import csv
import math

import matplotlib.pyplot as plt

def read_data(filename="Zee.csv"):
    #read CSV file into TRAINING and TEST data (can change later)
    with open(filename, newline= '') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    #we only want pt1, pt2, eta1, eta2, phi1, phi2 
    #datakey ['Run', 'Event', 'pt1', 'eta1', 'phi1', 'Q1', 'type1', 'sigmaEtaEta1', 'HoverE1', 'isoTrack1', 'isoEcal1', 'isoHcal1', 'pt2', 'eta2', 'phi2', 'Q2', 'type2', 'sigmaEtaEta2', 'HoverE2', 'isoTrack2', 'isoEcal2','isoHcal2']
    
    del data[0] #kill header line of data
    
    #split data into two halves
    training_data, testing_data = [],[]
    training_flag = 0
    for row in data:
        if training_flag % 10: #keep 90% of the data for training
            #cropped_data     = [ pt1 *  pt2,     eta1-eta2,     phi1-phi2  ]
            training_data.append([float(row[2])*float(row[12])/10000,float(row[3])-float(row[13]),float(row[4])-float(row[14])])
        else:
            #cropped_data     = [ pt1 *  pt2,     eta1-eta2,     phi1-phi2  ]
            testing_data.append([float(row[2])*float(row[12])/10000,float(row[3])-float(row[13]),float(row[4])-float(row[14])])
        training_flag += 1
    return training_data, testing_data

def compute_invariant_mass(training_data):
    Mz = []
    for row in training_data:
        row = list(map(float,row))
        pt1pt2 = row[0]/10000 #units of GeV
        Deta = row[1]
        Dphi = row[2]
        Mz2 = 2*pt1pt2*(math.cosh(Deta) - math.cos(Dphi))
        Mz.append([math.sqrt(Mz2)]) #units of GeV too
    return Mz

def make_NN(training_data, Mz):
    NN = Sequential()
    NN.add(Dense(100, activation='relu' ,input_dim=3))
    NN.add(Dense(100, activation='relu'))
    NN.add(Dense(1, activation='linear'))
    
    #train with leastsquares error
    learning_rate = 0.1 #THIS IS RELATED TO THE SCALE OF THE DATA SOMEHOW
    batch_size = 32
    loss = 'mean_squared_error'
    epochs = 150
    
    sgd = optimizers.sgd(lr=learning_rate)  # use SGD optimizer
    NN.compile(loss=loss, optimizer=sgd, metrics=['mse'])
    NN.summary()
    
    x = np.asarray(training_data)
    y = np.asarray(Mz)
    #TRAIN MODEL
    y_hat = NN.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return NN, y_hat
    
    
def plot_error(y_hat):
    #plot error loss
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_hat.history['loss'], 'o')  # show average losses
    plt.title("Average losses per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Avg accuracy")
    plt.yscale('log')
    plt.show()
    #plt.savefig("./example_1.2c_loss.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    return

def training_test(NN,Mz,training_data):
    #this compares the known Mz values with those predicted by the algorithm
    training_data = np.asarray(training_data)
    predict_y = NN.predict(training_data)
    predict_y = [ x[0] for x in predict_y]
    rel_err = [predict_y[i]/Mz[i][0] for i in range(len(predict_y))]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(range(len(predict_y)), rel_err)
    plt.title("Relative Error compared to training data")
    ax.set_xlabel('datapoint')
    ax.set_ylabel('Error')
#    plt.scatter(range(len(predict_y)), predict_y)
#    plt.scatter(range(len(predict_y)), [x[0] for x in Mz])
    plt.show()
    plt.close()
    sq_err = [((predict_y[i]-Mz[i][0])/Mz[i][0])**2 for i in range(len(Mz))]
    print("Average (least squares) percentage error: ", round(100*sum(sq_err)/len(sq_err),4),"%")
    return
    
def actual_test(NN,testing_data):
    """this is the "real" test, with the unseen testing data"""
    testing_data = np.asarray(testing_data)
    Mz = compute_invariant_mass(testing_data)
    predict_y = NN.predict(testing_data)
    predict_y = [ x[0] for x in predict_y]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rel_err = [predict_y[i]/Mz[i][0] for i in range(len(predict_y))]
    plt.scatter(range(len(predict_y)), rel_err)
    plt.title("Relative Error compared to testing data")
    ax.set_xlabel('datapoint')
    ax.set_ylabel('Error')
#    plt.scatter(range(len(predict_y)), predict_y)
#    plt.scatter(range(len(predict_y)), [x[0] for x in Mz])
    plt.show()
    plt.close()
    
    sq_err = [((predict_y[i]-Mz[i][0])/Mz[i][0])**2 for i in range(len(Mz))]
    print("Average (least squares) percentage error: ", round(100*sum(sq_err)/len(sq_err),4),"%")
    return    

training_data, testing_data = read_data()
Mz = compute_invariant_mass(training_data)    
print("Data collected. Creating/training neural net...")
NN, y_hat = make_NN(training_data,Mz)
print("Neural net created. Plotting results...")
plot_error(y_hat)
training_test(NN,Mz,training_data)
actual_test(NN,testing_data)