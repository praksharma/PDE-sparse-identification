from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix

x = np.arange(0,2,0.008)
data = np.polynomial.polynomial.polyval(x,[0,2,1,-2,-3,2.6,-0.4])
deriv=np.gradient(data)
'''
x = np.arange(0,2,0.008)

print(np.polynomial.polynomial.polyval(2,[1,2,3,4]))
here the expression would be evaluated as 1+ 2*2 + 3* 2^2 + 4* 2^3 +...
'''
noise = np.random.normal(0,0.1,data.shape)
noisy_data = data + noise

f = splrep(x,noisy_data,k=5,s=3) # B-spline
#plt.plot(x, data, label="raw data")
#plt.plot(x, noise, label="noise")
plt.figure(1)
plt.plot(x, noisy_data, label="noisy data")
plt.plot(x, splev(x,f), label="B-Spline")
derivSpline=splev(x,f,der=1)  # B-spline derivative

#plt.plot(x, splev(x,f,der=2)/100, label="2nd derivative")
#plt.hlines(0,0,2)
plt.legend(loc=0)

"""
    Tikhonov differentiation.

    
    """
    
 # Initialize a few things    
n = len(x)
f = np.matrix(data - data[0]).reshape((n,1))
dx=0.008
lam=1

# Get a trapezoidal approximation to an integral
A = np.zeros((n,n))
for i in range(1, n):
    A[i,i] = dx/2
    A[i,0] = dx/2
    for j in range(1,i): A[i,j] = dx

e = np.ones(n-1)
D = sparse.diags([e, -e], [1, 0], shape=(n-1, n)).todense() / dx

# Invert to find derivative
g = np.squeeze(np.asarray(np.linalg.lstsq(A.T.dot(A) + lam*D.T.dot(D),A.T.dot(f),rcond=None)[0]))

plt.figure(2)
plt.plot(x, deriv, label="np.gradient")
plt.plot(x, derivSpline, label="B-spline")
plt.plot(x,g,label="Tikhonov derivative")
plt.legend(loc=0)
plt.title('First derivative')