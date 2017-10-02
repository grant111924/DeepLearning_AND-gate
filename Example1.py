# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:05:06 2017

Define a function f(x)=x^2, then compute f(-2)
@author: DELL
"""
import theano

x=theano.tensor.scalar()
v=x**2 # **等於平方
f= theano.function([x],v)

print (f(-2))



