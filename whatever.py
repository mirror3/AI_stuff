# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:05:48 2018

@author: sriniv11
"""

li = [[1,2,3,],[4,5,6]]

for i in range(len(li)):
    for j in range(len(li[0])):
        li[i][j]+=10

print(li)