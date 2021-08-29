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
Velocity_y=Velocity[:,1]
plt.figure(1)
b=plt.tricontourf(data[:,0],data[:,1],Velocity_y,cmap="Reds")
plt.colorbar(b) 
xlowerlim=(min(data[:,0]))
xupperlim=(max(data[:,0]))

ylowerlim=(min(data[:,1]))
yupperlim=(max(data[:,1]))
plt.xlim(xlowerlim, xupperlim)
plt.ylim(ylowerlim, yupperlim)
plt.gca().set_aspect('equal', adjustable='box')

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

space_meshgrid=np.meshgrid(x,y)
x_meshgrid=space_meshgrid[0]
y_meshgrid=space_meshgrid[1]
# trying radial basis function for interpolation

#rbfi = Rbf(data[:,0],data[:,1],Velocity_y)

#predicted_y=rbfi(data[:,0],data[:,1])


# RBF interpolation and prediction
def RBF_interpolant(x,y,values):
    return Rbf(x,y,values) # all these arrays should have same size
    #return RBF_model(x_meshgrid,y_meshgrid)
    
def RBF_prediction(RBF_model,x_meshgrid,y_meshgrid):
    return RBF_model(x_meshgrid,y_meshgrid)

# accuracy when you know the y_true
def rbf_metric(true,pred):
    variance_score=metrics.explained_variance_score(true,pred) 
    r_squared=metrics.r2_score(true,pred)
    plt.plot(abs(true-pred))
    plt.xlabel('Samples')
    plt.ylabel('Absolute error')
    plt.title('Error plot')

# u_x
RBF_model=RBF_interpolant(data[:,0],data[:,1],Velocity[:,0])
u_x=RBF_prediction(RBF_model,x_meshgrid,y_meshgrid)
plt.figure(2)
plt.pcolor(x_meshgrid,y_meshgrid,u_x,cmap='Reds')#,vmin=0, vmax=4)
plt.gca().set_aspect('equal', adjustable='box')




