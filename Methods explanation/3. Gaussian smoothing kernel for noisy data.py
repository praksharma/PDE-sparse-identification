# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:14:36 2021

@author: Prakhar Sharma
"""
from matplotlib import pyplot as plt
import numpy as np

"""
Smoother for noisy data

Inpute = x, p, sigma
x = one dimensional series to be smoothed
p = width of smoother
sigma = standard deviation of gaussian smoothing kernel
"""

points = np.arange(0,2,0.008)
data = np.polynomial.polynomial.polyval(points,[0,2,1,-2,-3,2.6,-0.4])
noise = np.random.normal(0,0.1,data.shape)
x = data + noise

p=1
sigma=0.1

n = len(x)
y = np.zeros(n, dtype=np.float64)
g = np.exp(-np.power(np.linspace(-p,p,2*p),2)/(2.0*sigma**2))

for i in range(n):
    a = max([i-p,0])
    b = min([i+p,n])
    c = max([0, p-i])
    d = min([2*p,p+n-i])
    y[i] = np.sum(np.multiply(x[a:b], g[c:d]))/np.sum(g[c:d])
 
plt.figure(1)
plt.plot(points, x, label="noisy data")
plt.plot(points,y,label="smoothed data")
plt.legend(loc=0)
