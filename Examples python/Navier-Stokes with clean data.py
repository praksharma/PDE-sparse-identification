#!/usr/bin/env python
# coding: utf-8
# Don't use this code anymore
# Don't use this code anymore
# Don't use this code anymore
# Don't use this code anymore
# Don't use this code anymore
# # PDE-FIND for identifying Navier-Stokes
# 
# Samuel Rudy, 2016
# 
# This notebook demonstrates PDE-FIND for the vorticity equation given a simulation of fluid flowing around a cylinder.
# \begin{align*}
# \omega_t &= \frac{1}{Re}\nabla^2 \omega - (V \cdot \nabla)\omega\\
# V &= (v,u)\\
# Re &= 100
# \end{align*}
# The x and y components of the velocity field are given as forcing terms to the PDE.  That is, they appear in $\Theta$, but are not differentiated.

# In[1]:


#get_ipython().run_line_magic('pylab', 'inline')
#pylab.rcParams['figure.figsize'] = (12, 8)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PDE_FIND import *
import scipy.io as sio
import itertools
import matplotlib.pyplot as plt

# Load in the data.

# In[2]:

print('Loading data from external files')
# Load data
U = np.load('U.npy')
V = np.load('V.npy')
W = np.load('W.npy')

steps = 101
n = 449
m = 199
W = W.reshape(n,m,steps)      # vorticity
U = U.reshape(n,m,steps)      # x-component of velocity
V = V.reshape(n,m,steps)      # y-component of velocity

dt = 0.2
dx = 0.02
dy = 0.02
print('Data successfully loaded')
# Cut out the portion of the data before the cylinder
xmin = 100
xmax = 425
ymin = 15
ymax = 185
print('Removing gemoetry from the data')
# Cutting out the data involving the cylinder
W = W[xmin:xmax,ymin:ymax,:]
U = U[xmin:xmax,ymin:ymax,:]
V = V[xmin:xmax,ymin:ymax,:]
n,m,steps = W.shape # New n,m,steps after cutting out the cylinder

print('Geometry involving cylinder has been removed')


# Here we take the SVD of the data and reconstruct either with a reduced basis or everything.  It isn't necesarry but is interesting to show that we can still derive the correct PDE with the first 50 modes (maybe less). 

# In[3]:

# May be to check if the cut was successful. Or this reshape would give error
W = W.reshape(n*m,steps)
U = U.reshape(n*m,steps)
V = V.reshape(n*m,steps)
print('Data has been successfully reshaped')

# In[4]:
print('Performing SVD for dimensionality reduction')
# SVD for dimentionality reduction
# uw,sigmaw,vw = np.linalg.svd(W, full_matrices=False); vw = vw.T
# uu,sigmau,vu = np.linalg.svd(U, full_matrices=False); vu = vu.T
# uv,sigmav,vv = np.linalg.svd(V, full_matrices=False); vv = vv.T
print('SVD successfully done')

# In[5]:
# This plot would suggest the most dominated coherent modes that describes the dynamics
'''
plt.semilogy(sigmaw)
plt.semilogy(sigmau)
plt.semilogy(sigmav)
'''

# In[6]:


# Use this code to identify the PDE from reduced basis
# dim = 50
# Wn = uw[:,0:dim].dot(np.diag(sigmaw[0:dim]).dot(vw[:,0:dim].T)).reshape(n,m,steps)
# Un = uu[:,0:dim].dot(np.diag(sigmau[0:dim]).dot(vu[:,0:dim].T)).reshape(n,m,steps)
# Vn = uv[:,0:dim].dot(np.diag(sigmav[0:dim]).dot(vv[:,0:dim].T)).reshape(n,m,steps)

# Or this code to take the full solution
Wn = W.reshape(n,m,steps)
Un = U.reshape(n,m,steps)
Vn = V.reshape(n,m,steps)

print('Data successfully reshaped after SVD')
# ## Sample data points
# 
# Now randomly sample some points to use.  See figure 1, panel 2a-2c for visual description of what we're doing here.  5000 spatial points are samples and 60 timepoints are viewed at each one.  Note that we still need nearby points to take derivatives.

# In[7]:
print('Picking up random points from the data')
# Sample a collection of data points, stay away from edges so I can just use centered finite differences.
# The coordinates is chosen randomly from the sliced data. That is why random.choice is used here
np.random.seed(0) # generates same set of random numbers each time

num_xy = 5000 # number of x and y points to be taken randomly in the collection
num_t = 60  # number of timesteps to be taken in the collection
num_points = num_xy * num_t  # total number of points in the collection
# As we can't take derivative of the corner data using polynomial interpolation
boundary = 5 # Selected points in y axis [boundary:m-boundary]
boundary_x = 10 # Selected points in x axis [boundary_x:n-boundary_x]
points = {}  # dictionary to store the collection
count = 0
scatterx=np.array([])
scattery=np.array([])
for p in range(num_xy): # for
    x = np.random.choice(np.arange(boundary_x,n-boundary_x),1)[0] # randomly choose any x coordinate
    y = np.random.choice(np.arange(boundary,m-boundary),1)[0]    # randomly choose any y coordinate 
    # the following loop would insert same x and y coordinates for each time step
    scatterx=np.append(scatterx, x)
    scattery=np.append(scattery, y)
    for t in range(num_t):
        points[count] = [x,y,t+max(boundary, boundary_x)] 
        # t ranges from max(boundary, boundary_x) to num_t+max(boundary, boundary_x)
        # ensure that max(boundary, boundary_x)!<0 && num_t+max(boundary, boundary_x)!>steps
        # The t starts from max(boundary, boundary_x) because we need previous time steps for partial derivative wrt time
        count = count + 1

'''
                       m
             --------------------
             |                  |
boundary_x   |      domain      |  n
             |                  |
             --------------------
                   boundary

'''

# ## Construct $\Theta (U)$ and compute $U_t$
# 
# Take derivatives and assemble into $\Theta(\omega, u ,v)$

# In[plot]
plt.figure(1)
xx, yy = np.meshgrid(
    np.arange(325)*dx,
    np.arange(170)*dy)
plt.pcolor(xx,yy,Wn[:,:,75].T,cmap='coolwarm', vmin=-4, vmax=4)

# plt.scatter(scatterx, scattery,s=np.pi, c=(0,0,0), alpha=0.5)
# plt.show()
# In[8]:
print('Computing derivatives')
# The x and y components of the velocity field are given as forcing terms to the PDE.  That is, they appear in $\Theta$, but are not differentiated.
# Take up to second order derivatives.
w = np.zeros((num_points,1))
u = np.zeros((num_points,1))
v = np.zeros((num_points,1))
wt = np.zeros((num_points,1))
wx = np.zeros((num_points,1))
wy = np.zeros((num_points,1))
wxx = np.zeros((num_points,1))
wxy = np.zeros((num_points,1))
wyy = np.zeros((num_points,1))

#N = 2*boundary-1  # odd number of points to use in fitting
#Nx = 2*boundary_x-1  # odd number of points to use in fitting
deg = 5 # degree of polynomial to use

for p in points.keys():
    
    if p%10000==0:
        print('Step: ',str(p),' of ',str(len(points)))
    [x,y,t] = points[p]
    w[p] = Wn[x,y,t]
    u[p] = Un[x,y,t]
    v[p] = Vn[x,y,t]
    # def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    # Here the length of u and x should be same for the least squares
    # wt[p] = PolyDiffPoint(Wn[int(x),int(y),int(t-boundary/2):int(t+boundary/2)], np.arange(boundary)*dt, deg, 1)[0] # see line 127
    # x_diff = PolyDiffPoint(Wn[int(x-boundary_x/2):int(x+boundary_x/2),int(y),int(t)], np.arange(boundary_x)*dx, deg, 2)  # x depends on boundary_x (line 123)
    # y_diff = PolyDiffPoint(Wn[int(x),int(y-boundary/2):int(y+boundary/2),int(t)], np.arange(boundary)*dy, deg, 2)          # x depends on boundary (line 124)
    # wx[p] = x_diff[0] 
    # wy[p] = y_diff[0]  
    
    # x_diff_yp = PolyDiffPoint(Wn[int(x-boundary_x/2):int(x+boundary_x/2),int(y+1),t], np.arange(boundary_x)*dx, deg, 2) # w_x^(y+1)
    # x_diff_ym = PolyDiffPoint(Wn[int(x-boundary_x/2):int(x+boundary_x/2),int(y-1),t], np.arange(boundary_x)*dx, deg, 2) # w_x^(y-1)
   
    wt[p] = PolyDiffPoint(Wn[int(x),int(y),int(t-boundary):int(t+boundary)], np.arange(2*boundary)*dt, deg, 1)[0] # see line 127
    x_diff = PolyDiffPoint(Wn[int(x-boundary_x):int(x+boundary_x),int(y),int(t)], np.arange(2*boundary_x)*dx, deg, 2)  # x depends on boundary_x (line 123)
    y_diff = PolyDiffPoint(Wn[int(x),int(y-boundary):int(y+boundary),int(t)], np.arange(2*boundary)*dy, deg, 2)          # x depends on boundary (line 124)
    wx[p] = x_diff[0] 
    wy[p] = y_diff[0]  
    
    x_diff_yp = PolyDiffPoint(Wn[int(x-boundary_x):int(x+boundary_x),int(y+1),t], np.arange(2*boundary_x)*dx, deg, 2) # w_x^(y+1)
    x_diff_ym = PolyDiffPoint(Wn[int(x-boundary_x):int(x+boundary_x),int(y-1),t], np.arange(2*boundary_x)*dx, deg, 2) # w_x^(y-1)
   
    
    # second derivative
    wxx[p] = x_diff[1]
    wxy[p] = (x_diff_yp[0]-x_diff_ym[0])/(2*dy) # central difference
    wyy[p] = y_diff[1]


# In[9]:

print('Building time series library')
# Form a huge matrix using up to quadratic polynomials in all variables.
X_data = np.hstack([w,u,v])
X_ders = np.hstack([np.ones((num_points,1)), wx, wy, wxx, wxy, wyy])
X_ders_descr = ['','w_{x}', 'w_{y}','w_{xx}','w_{xy}','w_{yy}']
X, description = build_Theta(X_data, X_ders, X_ders_descr, 2, data_description = ['w','u','v'])
print ('Candidate terms for PDE[1]',str(description[1:]))


# ## Solve for $\xi$
# 
# TrainSTRidge splits the data up into 80% for training and 20% for validation.  It searches over various tolerances in the STRidge algorithm and finds the one with the best performance on the validation set, including an $\ell^0$ penalty for $\xi$ in the loss function.

# In[10]:

print('Using STRidge for discovered PDE')

lam = 10**-5
d_tol = 5
c = TrainSTRidge(X,wt,lam,d_tol)
print_pde(c, description, ut = 'w_t')


# In[11]:


# err = abs(np.array([(0.009884-0.01)*100/0.01, (0.009902-0.01)*100/0.01, (-0.990371+1)*100, (-0.986629+1)*100]))
# print ("Error using PDE-FIND to identify Navier-Stokes:\n")
# print ("Mean parameter error:", np.mean(err), '%')
# print ("Standard deviation of parameter error:", np.std(err), '%')


# In[ ]:




