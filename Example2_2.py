# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:16:36 2017
Define Output Variables
@author: DELL
"""

import theano.tensor as T
#第零階張量為純量，第一階張量為向量， 第二階張量則成為矩陣
# rank 0 的tensor => scalar
# rank 1 的tensor => vector
# rank 2 的tensor => matrix

x1=T.scalar() 
x2=T.scalar()
x3=T.matrix()
x4=T.matrix()

x1=2
x2=4
y1=x1+x2

y2=x1*x2

y3=x3*x4

y4=T.dot(x3,x4)

print (y1)
print (y2)
print (y3)
print (y4)