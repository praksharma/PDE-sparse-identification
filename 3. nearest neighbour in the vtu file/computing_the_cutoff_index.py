import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
Rows go from top to bottom so they symbolize y coordinates change
Similarly, columns go from left to right hence they symbolize x coordinate changes

ROWS -> y-coordinate
COLUMNS -> x-coordinates

'''
 
column_min = 100
column_max = 425
row_min = 30
row_max = 165


x_meshgrid=np.load('RBF data/x_meshgrid.npy')
y_meshgrid=np.load('RBF data/y_meshgrid.npy')
normed_u=np.load('RBF data/normed_velocity.npy')
normed_u[normed_u[:,:]>1]=0 # RBF interpolates the veocity in the cylinder very high. ideally it should be zero
# min/max coordinates

xlowerlim= x_meshgrid[0,column_min]
xupperlim= x_meshgrid[0,column_max]

ylowerlim= y_meshgrid[row_min,0]
yupperlim= y_meshgrid[row_max,0]

print('xlowerlim = ',xlowerlim)
print('xupperlim = ',xupperlim)
print('ylowerlim = ',ylowerlim)
print('yupperlim = ',yupperlim)

#%% Potting trimmed section with rectangle
# The anchor point (bottom-left coordinate)
x = x_meshgrid[0,column_min]  # for any row teh values are same (check the meshgrids) so I took the first row 
y = y_meshgrid[row_min,0]
# height of the rectangle
width = abs(x_meshgrid[0,column_max]-x) # width symbolises columns
height = abs(y_meshgrid[row_max,0]-y) # height symbolises rows

plt.figure(1)
plot=plt.pcolor(x_meshgrid,y_meshgrid,normed_u,cmap='jet',vmin=1e-3, vmax=np.max(normed_u))
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar(orientation='horizontal')
plot.cmap.set_over('white')
plot.cmap.set_over('white')
 
rect=patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor='none')
plt.gca().add_patch(rect)

#%% Cutting out the data involving the cylinder for knowing the index to be used in future work

print('Removing geometry from the data')
# Cutting out the data involving the cylinder
x_meshgrid = x_meshgrid[row_min:row_max,column_min:column_max]
y_meshgrid = y_meshgrid[row_min:row_max,column_min:column_max]
normed_u = normed_u[row_min:row_max,column_min:column_max]

# Plotting trimmed flow
plt.figure(2)
plt.pcolor(x_meshgrid,y_meshgrid,normed_u,cmap='jet',vmin=0, vmax=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar(orientation='horizontal')
