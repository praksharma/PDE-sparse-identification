# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:01:13 2021

@author: Prakhar Sharma
"""
import numpy as np
import matplotlib.pyplot as plt

"""
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """


x = np.arange(0,2,0.008)
data = np.polynomial.polynomial.polyval(x,[0,2,1,-2,-3,2.6,-0.4])
deriv=np.gradient(data)
noise = np.random.normal(0,0.1,data.shape)
u = data + noise
deg=1

diff=1
width=1

u = u.flatten()
x = x.flatten()

n = len(x)
du = np.zeros((n - 2*width,diff))

# Take the derivatives in the center of the domain
for j in range(width, n-width):

    points = np.arange(j - width, j + width)

    # Fit to a Chebyshev polynomial
    # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
    poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

    # Take derivatives
    for d in range(1,diff+1):
        du[j-width, d-1] = poly.deriv(m=d)(x[j])

plt.figure(1)
plt.plot(x,deriv)
plt.plot(x[:-2],du)
plt.legend(['np.gradient','polynomial'])