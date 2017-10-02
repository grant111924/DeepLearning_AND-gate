# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:11:55 2017
Define Input Variables
@author: DELL
"""

# Step 1 Define Input Variables
import theano
import theano.tensor as T
#a=theano.tensor.scalar()
#b=theano.tensor.matrix()
#c=theano.tensor.matrix("hahaha")

a=T.scalar()
b=T.matrix()
c=T.matrix("hahaha")

print(a)
print(b)
print(c)