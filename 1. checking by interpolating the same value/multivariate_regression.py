import meshio
import matplotlib.pyplot as plt
import numpy as np

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

# trying Linear regression
from scipy.interpolate import Rbf
rbfi = Rbf(data[:,0],data[:,1],Velocity_y)

predicted_y=rbfi(data[:,0],data[:,1])

# accuracy
import sklearn.metrics as metrics
variance_score=metrics.explained_variance_score(Velocity_y,predicted_y) 
r_squared=metrics.r2_score(Velocity_y,predicted_y)
plt.figure(2)
plt.plot(abs(Velocity_y-predicted_y))