# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:59:13 2021

@author: Prakhar Sharma
"""

import numpy as np
import matplotlib.pyplot as plt
from methods.PDE_FIND import *

print('Loading the dataset')

P = np.load('dataset/100_time_steps/P.npy')     # Pressure
U = np.load('dataset/100_time_steps/U.npy')     # x-velocity
V = np.load('dataset/100_time_steps/V.npy')     # y-velocity

# Saving only requisite data without loading the whole file in the RAM
dx = np.load('dataset/100_time_steps/steps.npy', mmap_mode= 'r')[0]
dy = np.load('dataset/100_time_steps/steps.npy', mmap_mode= 'r')[1]
dt = np.load('dataset/100_time_steps/steps.npy', mmap_mode= 'r')[2]
# Reshaping the data to vertical data
# such that x cooresponds to rows and y corresponds to column
# this will save us from brainstorming during derivative computation
P = P.transpose(1,0,2)
U = U.transpose(1,0,2)
V = V.transpose(1,0,2)

steps = np.shape(U[0,0,:])[0]
n = np.shape(U[:,0,0])[0] # number of columns or x-coordinates
m = np.shape(U[0,:,0])[0] # number of rows or y-coordinates

# Checking the data is vertical or not (leave it!!!!)
# xx, yy = np.meshgrid(np.arange(n),np.arange(m))
# plt.pcolormesh( xx, yy, U[:,:,80].T)
# plt.gca().set_aspect('equal', adjustable='box')
# perform SVD. Not necessary
#perform_SVD(P, U, V)

# Choosing random data points
num_xy = 5000 # number of x and y points to be taken randomly in the collection
num_t = 60  # number of timesteps to be taken in the collection
boundary_y = 5 # Selected points in y axis [boundary_y:m-boundary_y]
boundary_x = 11 # Selected points in x axis [boundary_x:n-boundary_x]
points = choose_random_points(num_xy, num_t, boundary_y, boundary_x, n, m) # dictionary to store the collection
num_points = num_xy * num_t

print('Computing derivatives')
# The x and y components of the velocity field are given as forcing terms to the PDE.  That is, they appear in $\Theta$, but are not differentiated.
# Take up to second order derivatives.
p = np.zeros((num_points,1))
u = np.zeros((num_points,1))
v = np.zeros((num_points,1))
# time derivative
ut = np.zeros((num_points,1))
vt = np.zeros((num_points,1))
# pressure derivative
px = np.zeros((num_points,1))
py = np.zeros((num_points,1))
# first derivative of velocity
ux = np.zeros((num_points,1))
uy = np.zeros((num_points,1))
vx = np.zeros((num_points,1))
vy = np.zeros((num_points,1))
# Second derivative of velocity
uxx = np.zeros((num_points,1))
uyy = np.zeros((num_points,1))
uxy = np.zeros((num_points,1))

vxx = np.zeros((num_points,1))
vyy = np.zeros((num_points,1))
vxy = np.zeros((num_points,1))


#N = 2*boundary-1  # odd number of points to use in fitting
#Nx = 2*boundary_x-1  # odd number of points to use in fitting
deg = 4 # degree of polynomial to use for polynomial interpolation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for key in points.keys():
    
    if key%1000==0:
        print('Step: ',str(key),' of ',str(len(points)))
    [x,y,t] = points[key] # x,y,t for each point in points dictionary
    p[key] = P[x,y,t]
    u[key] = U[x,y,t]
    v[key] = V[x,y,t]
    # def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    # Here the length of u and x should be same for the polynomial fitting np.polynomial.chebyshev.Chebyshev.fit
    # The next line throws 11 U-values with corresponding time
    ut[key] = PolyDiffPoint(U[int(x),int(y),int(t-max(boundary_y, boundary_x)/2):int(t+max(boundary_y, boundary_x)/2)], np.arange(max(boundary_y, boundary_x))*dt, deg, 1)[0]
    vt[key] = PolyDiffPoint(V[int(x),int(y),int(t-max(boundary_y, boundary_x)/2):int(t+max(boundary_y, boundary_x)/2)], np.arange(max(boundary_y, boundary_x))*dt, deg, 1)[0]
    
    px[key] = PolyDiffPoint(P[int(x-(boundary_x)/2):int(x+ (boundary_x)/2),int(y),int(t)], np.arange(boundary_x)*dx, deg, 2)[0] # x with boundary_x 
    py[key] = PolyDiffPoint(P[int(x),int(y-(boundary_y)/2):int(y+(boundary_y)/2),int(t)], np.arange(boundary_y)*dx, deg, 2)[0]  # y with boundary_y 
    
    ux_diff = PolyDiffPoint(U[int(x-(boundary_x)/2):int(x+ (boundary_x)/2),int(y),int(t)], np.arange(boundary_x)*dx, deg, 2) # x with boundary_x
    uy_diff = PolyDiffPoint(U[int(x),int(y-(boundary_y)/2):int(y+(boundary_y)/2),int(t)], np.arange(boundary_y)*dx, deg, 2)  # y with boundary_y
    vx_diff = PolyDiffPoint(V[int(x-(boundary_x)/2):int(x+ (boundary_x)/2),int(y),int(t)], np.arange(boundary_x)*dx, deg, 2) # x with boundary_x 
    vy_diff = PolyDiffPoint(V[int(x),int(y-(boundary_y)/2):int(y+(boundary_y)/2),int(t)], np.arange(boundary_y)*dx, deg, 2)  # y with boundary_y 
    ux[key] = ux_diff[0]
    uy[key] = uy_diff[0]
    vx[key] = vx_diff[0]
    vy[key] = vy_diff[0]
    
    # central difference
    ux_diff_yp = PolyDiffPoint(U[int(x-boundary_x/2):int(x+boundary_x/2),int(y+1),int(t)], np.arange(boundary_x)*dx, deg, 2) # w_x^(y+1)
    ux_diff_ym = PolyDiffPoint(U[int(x-boundary_x/2):int(x+boundary_x/2),int(y-1),int(t)], np.arange(boundary_x)*dx, deg, 2) # w_x^(y-1)
    
    vx_diff_yp = PolyDiffPoint(V[int(x-boundary_x/2):int(x+boundary_x/2),int(y+1),int(t)], np.arange(boundary_x)*dx, deg, 2) # w_x^(y+1)
    vx_diff_ym = PolyDiffPoint(V[int(x-boundary_x/2):int(x+boundary_x/2),int(y-1),int(t)], np.arange(boundary_x)*dx, deg, 2) # w_x^(y-1)
    
    uxx[key] = ux_diff[1]
    uyy[key] = uy_diff[1]
    vxx[key] = vx_diff[1]
    vyy[key] = vy_diff[1]
    
    uxy[key] = (ux_diff_yp[0]-ux_diff_ym[0])/(2*dy)
    vxy[key] = (vx_diff_yp[0]-vx_diff_ym[0])/(2*dy)
    

np.savez('dataset/derivatives/derivatives.npz', p=p, u=u, v=v, ut=ut, vt=vt, px=px, py=py, ux=ux, uy=uy, vx=vx, vy=vy, uxx=uxx, 
         uyy=uyy, vxx=vxx, vyy=vyy, uxy=uxy, vxy=vxy, num_points=num_points, )
