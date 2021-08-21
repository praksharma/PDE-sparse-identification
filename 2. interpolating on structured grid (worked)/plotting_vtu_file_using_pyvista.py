# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 04:39:16 2021

@author: Prakhar Sharma
"""

import meshio
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics # for r squared
from scipy.interpolate import Rbf

mesh = meshio.read("circle-2d-drag_1000.vtu")
cells=mesh.cells_dict
pointData=mesh.point_data
Velocity=pointData['Velocity']
points=mesh.points
data=points[:,[0,1]] # extracting x,y data only

# Plotting the y velocity
# Velocity_y=np.sqrt(np.square(Velocity[:,1])+np.square(Velocity[:,0]))
# plt.figure(1)
# b=plt.tricontourf(data[:,0],data[:,1],Velocity_y,cmap="Reds")
# plt.gca().set_aspect('equal', adjustable='box')
#plt.colorbar(b) 

import pyvista as pv
mesh = pv.read("circle-2d-drag_1000.vtu")

cpos = mesh.plot(scalars="Velocity",cpos='xy')

'''
Camera description must be one of the following:

Iterable containing position, focal_point, and view up.  For example:
[(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)]

Iterable containing a view vector.  For example:
[-1.0, 2.0, -5.0]

A string containing the plane orthogonal to the view direction.  For example:
'xy'
'''

#%% Creating structured grid

# Boundaries of the structured grid
xlowerlim=(min(data[:,0]))
xupperlim=(max(data[:,0]))

ylowerlim=(min(data[:,1]))
yupperlim=(max(data[:,1]))

# Number of nodes in the grid
num_x=500
num_y=200

x=np.linspace(xlowerlim,xupperlim,num_x)
y=np.linspace(ylowerlim,yupperlim,num_y)
# creating meshgrid
space_meshgrid=np.meshgrid(x,y)
x_meshgrid=space_meshgrid[0]
y_meshgrid=space_meshgrid[1]

u=np.load('u.npy')
v=np.load('v.npy')

normed_u=np.sqrt(np.square(u)+np.square(v))

plt.figure(2)
plt.pcolor(x_meshgrid,y_meshgrid,normed_u,cmap='viridis',vmin=0, vmax=1)
# here I've used vmin and vmax to trim off the high velocity values interpolated by
# RBF within the cylinder, as it is never the part of the simulation.
# However, while applying the RBF interpolation we considered that part also.
# So, from the input data one can see that the normed-velocity lies in range 0-1
# so trim has been done from vmin=0 to vmax=1
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar(orientation='horizontal')

plt.figure(3)
plt.plot(normed_u)

#%% Cutting out the data involving the cylinder for knowing the index to be used in future work
xmin = 100
xmax = 425
ymin = 15
ymax = 185
print('Removing gemoetry from the data')
# Cutting out the data involving the cylinder
x_meshgrid = x_meshgrid[xmin:xmax,ymin:ymax]
y_meshgrid = y_meshgrid[xmin:xmax,ymin:ymax]
normed_u = normed_u[xmin:xmax,ymin:ymax]

# Plotting trimmed 

#%%
# saving files

# np.save('mesh_coordinates.npy',data)
# np.save('velocity.npy',Velocity)
# np.save('x_meshgrid.npy',x_meshgrid)
# np.save('y_meshgrid.npy',y_meshgrid)
# np.save('normed_velocity.npy',normed_u)