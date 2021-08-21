import meshio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


mesh = meshio.read("circle-2d-drag_1000.vtu")
cells=mesh.cells_dict
pointData=mesh.point_data
Velocity=pointData['Pressure']
points=mesh.points



data=points[:,[0,1]]


plt.figure(1)
a=plt.scatter(data[:,0],data[:,1], c=Velocity,s=0.5)#, cmap="Reds", s=0.1, edgecolors="black")
plt.colorbar(a) 

xlowerlim=(min(data[:,0]))
xupperlim=(max(data[:,0]))

ylowerlim=(min(data[:,1]))
yupperlim=(max(data[:,1]))
plt.xlim(xlowerlim, xupperlim)
plt.ylim(ylowerlim, yupperlim)

plt.gca().set_aspect('equal', adjustable='box')
ax = plt.axes()
ax.set_facecolor("lightcoral")
# dataFile=np.concatenate((data,tracer),axis=0)
# np.savetxt('DataFile.txt',(data,tracer))#,fmt="%4.4f") 

plt.figure(2)
b=plt.tricontourf(data[:,0],data[:,1],(Velocity),cmap="Reds")
plt.colorbar(b) 
xlowerlim=(min(data[:,0]))
xupperlim=(max(data[:,0]))

ylowerlim=(min(data[:,1]))
yupperlim=(max(data[:,1]))
plt.xlim(xlowerlim, xupperlim)
plt.ylim(ylowerlim, yupperlim)
plt.gca().set_aspect('equal', adjustable='box')


