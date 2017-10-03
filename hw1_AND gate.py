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
eta=0.5   #Learning rate
time = 150000 #training_iterations times
               
z=T.dot(w,x)+b   #compute z=wX+b 
sigmF=1/(1+T.exp(-z))  # sigmoid activation function => [0,1] interval value        
cost =T.sum((sigmF-y)**2) #cost function  
dw,db=T.grad(cost,[w,b]) #gradient compute
gradient = theano.function([x,y],updates=[(w,w-eta*dw),(b,b-eta*db)])# update function
neurnoPredict = theano.function([x],sigmF)  #computer predict result
# Set inputs and correct output values
X=[[0,0],[1,1],[1,0],[0,1]]
Y=[0,1,0,0]

#Training data
for t in range(time):
     i=random.randrange(0,4)
     x=X[i]
     y=Y[i]
     #print(x,y)
     #print (neurno(x))
     gradient(x,y)
     #print(w.get_value(),b.get_value())
 
#Get the best Weight and Bias
W=w.get_value()
B=b.get_value()
dataX=[]
dataY=[]

#draw picture
for i in range(1000):
    x_axis = random.uniform(0,1)
    y_axis = (W[0]*x_axis+B)/(-W[1])
    dataX.append(x_axis)
    dataY.append(y_axis)
   # print(x_axis)
   # print(y_axis)
plt.plot(dataX,dataY,"o")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show() 

#Testing data
test=[[1,1],[1,0],[1,1],[0,1],[1,1]]
for i in range(len(test)):
    print(neurnoPredict(test[i]))

