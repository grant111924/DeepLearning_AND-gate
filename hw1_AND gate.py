# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:08:21 2017

@author: DELL
"""

import theano 
import theano.tensor as T
import random
import numpy as np
import matplotlib.pyplot as plt

x=T.vector() #data feature value
y=T.scalar() #data labeled value 0 or 1
w=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'w')#weight inital value=[1,1]
b=theano.shared(0.)  #bias inital value=0
eta=random.random()  #Learning rate
time = 30000 #training_iterations times

def parameterWXB(weight,vector,bias):#comupte z=wX+b 
    neuron=T.dot(weight,vector)+bias  
    return neuron           
def sigmoid(z):# sigmoid activation function => [0,1] interval value 
    sigmo=1/(1+T.exp(-z)) 
    return  sigmo    

z=parameterWXB(w,x,b) #compute z 
aF=sigmoid(z)  # sigmoid activation function
cost =T.sum((aF-y)**2) #cost function  
dw,db=T.grad(cost,[w,b]) #gradient compute
gradient = theano.function([x,y],updates=[(w,w-eta*dw),(b,b-eta*db)])# update function
# Set inputs and correct output values
X=[[0,0],[1,1],[1,0],[0,1]]
Y=[0,1,0,0]

#Training data
for t in range(time):
     i=random.randrange(0,4)
     trainingX=X[i]
     trainingY=Y[i]
     gradient(trainingX,trainingY)
     #print(w.get_value(),b.get_value())
 
#Get the best Weight and Bias
bestW=w.get_value()
bestB=b.get_value()
dataX=[]
dataY=[]

#draw picture
for i in range(1000):
    x_axis = random.uniform(0,1)
    y_axis = (bestW[0]*x_axis+bestB)/(-bestW[1])
    dataX.append(x_axis)
    dataY.append(y_axis)
   # print(x_axis)
   # print(y_axis)
plt.plot(dataX,dataY,"o")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show() 

#Testing data
test=[[1,1],[1,0],[1,1],[0,1],[1,1],[0,0]]
z=T.dot(bestW,x)+bestB   #compute z=wX+b 
sigmF=1/(1+T.exp(-z))  # sigmoid activation function => [0,1] interval value  
neurnoPredict = theano.function([x],sigmF)  #computer predict result
print("Eta(random):",eta)
print("Iteration:",time)
print("Best B:",bestB)
print("Best W:",bestW)
print("Testing......")
for i in range(len(test)):
    print("input:%r result: %f"%(test[i],neurnoPredict(test[i])))