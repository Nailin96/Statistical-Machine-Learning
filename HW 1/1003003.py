#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 02:52:50 2020

@author: Nailin Zhao
"""
# settings for MAC
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy
import torch

CSV_PATH = ''
CSV_NAME = 'boston.csv'
torch.manual_seed(426)
learning_rate = 0.004

def main():
    inputs,target = dataPrep()
    weight,bias = genWeightBias()
    xinput = torch.cat((inputs,torch.ones(505,1)),1)
    loss = training(xinput, weight, bias, target, learning_rate)
#    prediction = linear_model(xinput,weight,bias)
#    print('Prediction: ', prediction)
    plotGraph(loss,learning_rate)
    


def dataPrep():
    csv = CSV_PATH + CSV_NAME
    data = numpy. genfromtxt(csv , delimiter=',', skip_header=1)
    inputs = data [:, [0,1,2]]
    inputs = inputs.astype(numpy.float32)
    inputs = torch.from_numpy(inputs)
    target = data[:,3]
    target = target.astype(numpy.float32)
    target = torch.from_numpy(target)  
    #print (inputs.shape)  # 505，3
    #print (target.shape)  # 505
    return inputs,target


#3.1
def genWeightBias():
    w = torch.rand(3)  # wrm, wrad and wcrim
    b = torch.rand(1)
    w.requires_grad = True
    b.requires_grad = True
    return w,b

def linear_model(xinput, weight, bias):
    return torch.mv(xinput,torch.cat((weight,bias),0))


#3.2
def mse(y, target):
    e = y-target  # error
    s = e**2  # squared
    m = sum(s)/505  #mean
    return m


#3.3
def training(xinput, weight, bias, target, learning_rate=0.01):
    loss_list = []
    for i in range(200):
        print("Epoch", i, ":")
        
        # compute the model predictions
        pred = linear_model(xinput, weight, bias)
        # compute the loss and its gradient
        loss = mse(pred,target)
        loss_list.append(loss)
        loss.backward()
        
        print("Loss=", loss)
        
        with torch.no_grad():
            
            # update the weights
            weight -= learning_rate * weight.grad
            # update the bias
            bias -= learning_rate * bias.grad
            
            weight.grad.zero_()
            bias.grad.zero_()
    
    return loss_list


#3.4
def plotGraph(loss_list,learning_rate,fname='graph.png'):
    ax = plt.axes()
    plt.xlim(0, 200)
    plt.ylim(0, 400)
    plt.title('learning rate = ' + str(learning_rate))
    ax.plot(range(200), loss_list);
    plt.savefig(fname)
    
main()