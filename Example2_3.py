# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:11:12 2017
Define function 
Examples for Matrix
@author: DELL
"""
import theano 
import theano.tensor as T

a=T.matrix()
b=T.matrix()

c=a*b
d=T.dot(a,b)

F1= theano.function([a,b],c)
F2=theano.function([a,b],d)

A=[[1,2],[3,4]]
B=[[2,4],[6,8]]
C=[[1,2],[3,4],[5,6]]

print (F1(A,B))
print(F2(C,B))