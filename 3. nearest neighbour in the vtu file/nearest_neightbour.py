# Documentation: https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms

import meshio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf

mesh = meshio.read("circle-2d-drag_1000.vtu")
#cells=mesh.cells_dict
pointData=mesh.point_data
Velocity=pointData['Velocity'][:,[0,1]] # extracting the x and y velocity
points=mesh.points
nodal_coordinates=points[:,[0,1]] # extracting x,y data only


#%% Cut out the portion of the data before the cylinder
# see computing_the_cutoff_index.py in the same directory

xlowerlim =  0.44088176352705416
xupperlim =  1.8737474949899802
ylowerlim =  0.06180904522613064
yupperlim =  0.3399497487437185

# boolean to data to be trimmed

bool_data=np.zeros((len(nodal_coordinates)),dtype=bool)
# make sure the data is within x limit
x_limit_bool=np.logical_and(nodal_coordinates[:,0]<xupperlim, nodal_coordinates[:,0]>xlowerlim)
# make sure the data is within y limit
y_limit_bool=np.logical_and(nodal_coordinates[:,1]<yupperlim, nodal_coordinates[:,1]>ylowerlim)
# Using logical and to make sure the data is within both the limits
trimmed_bool_array=np.logical_and(x_limit_bool,y_limit_bool)
nodal_coordinates=nodal_coordinates[trimmed_bool_array]
Velocity=Velocity[trimmed_bool_array]


# for i in range (0,len(nodal_coordinates)):
#     bool_data[i]= ((nodal_coordinates[i,0]<xupperlim and nodal_coordinates[i,0]>xlowerlim) and (nodal_coordinates[i,1]<yupperlim and nodal_coordinates[i,1]>ylowerlim))

# nodal_coordinates[(nodal_coordinates[:,0]<xupperlim & nodal_coordinates[:,0]>xlowerlim) & (nodal_coordinates[:,1]<yupperlim & nodal_coordinates[:,1]>ylowerlim)]


# temp2=nodal_coordinates[bool_data]





#%% Point from structured grid

#Creating structured grid on trimmed mesh

# # Boundaries of the structured grid
# xlowerlim=(min(nodal_coordinates=[:,0]))
# xupperlim=(max(nodal_coordinates=[:,0]))

# ylowerlim=(min(nodal_coordinates=[:,1]))
# yupperlim=(max(nodal_coordinates=[:,1]))

# Number of nodes in the grid
num_x=500
num_y=200

x=np.linspace(xlowerlim,xupperlim,num_x)
y=np.linspace(ylowerlim,yupperlim,num_y)

x_meshgrid,y_meshgrid=np.meshgrid(x,y)

#%%
# RBF interpolation and prediction
def RBF_interpolant(x,y,values):
    return Rbf(x,y,values) # all these arrays should have same size
    
    
def RBF_prediction(RBF_model,x_meshgrid,y_meshgrid):
    return RBF_model(x_meshgrid,y_meshgrid)

RBF_model=RBF_interpolant(nodal_coordinates[:,0],nodal_coordinates[:,1],Velocity[:,0])
u_x=RBF_prediction(RBF_model,x_meshgrid,y_meshgrid)

plt.figure(2)
plt.pcolor(x_meshgrid,y_meshgrid,u_x,cmap='viridis')#,vmin=0, vmax=4)
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar(orientation='horizontal')

# It works really well



#%% Sample data points
'''
Credit of this part of code goes to:
Rudy SH, Brunton SL, Proctor JL, Kutz JN. Data-driven discovery of partial differential equations. Science Advances. 2017 Apr 1;3(4):e1602614.
Now randomly sample some points to use. 
'''
print('Picking up random points from the data')
# Sample a collection of data points, stay away from edges so I can use interpolation.
# The coordinates is chosen randomly from the sliced data. That is why random.choice is used here
np.random.seed(0) # generates same set of random numbers each time
num_xy = 5000 # number of x and y points to be taken randomly in the collection
num_t = 60  # number of timesteps to be taken in the collection
num_points = num_xy * num_t  # total number of points in the collection
# Trimming the data that involves the cylinder
boundary = 5 # Selected points in y axis [boundary:m-boundary]
boundary_x = 11 # Selected points in x axis [boundary_x:n-boundary_x]
points = {}  # dictionary to store the collection
count = 0

