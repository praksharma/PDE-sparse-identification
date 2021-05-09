# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:53:51 2021

@author: Prakhar Sharma



Takes dth derivative data using 2nd order finite difference method (up to d=3)
Works but with poor accuracy for d > 3

Input:
u = data to be differentiated
dx = Grid spacing.  Assumes uniform spacing
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev

x = np.arange(0,2,0.008)
data = np.polynomial.polynomial.polyval(x,[0,2,1,-2,-3,2.6,-0.4])
deriv=np.gradient(data)

noise = np.random.normal(0,0.1,data.shape)
u = data + noise
    
n = u.size
ux = np.zeros(n, dtype=np.complex64)
dx=0.01


# B-spline
f = splrep(x,u,k=5,s=3) # B-spline


# first derivative
derivSpline=splev(x,f,der=1)  # B-spline derivative
plt.figure(1)
plt.plot(x, derivSpline, label="B-spline")

for i in range(1,n-1):
    ux[i] = (u[i+1]-u[i-1]) / (2*dx)

ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
plt.plot(x, ux, label="FDM")
plt.title('First derivative')
plt.legend(loc=0)


# Second derivative
derivSpline=splev(x,f,der=2)  # B-spline derivative
plt.figure(2)
plt.plot(x, derivSpline, label="B-spline")

for i in range(1,n-1):
    ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2

ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
plt.plot(x, ux, label="FDM")
plt.title('Second derivative')
plt.legend(loc=0)


# Thrid derivative
derivSpline=splev(x,f,der=3)  # B-spline derivative
plt.figure(3)
plt.plot(x, derivSpline, label="B-spline")

for i in range(2,n-2):
    ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3

ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
plt.plot(x, ux, label="FDM")
plt.title('Third derivative')
plt.legend(loc=0)