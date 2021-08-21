import numpy as np
from methods.methodsImproved import *

print('Initializing inputs')        
path='circledata'

# see computing_the_cutoff_index.py in the same directory
xlowerlim =  0.44088176352705416
xupperlim =  1.8737474949899802
ylowerlim =  0.06180904522613064
yupperlim =  0.3399497487437185

# Choose number of data points in the grid
num_x=500
num_y=200
print('Inputs initialization completed')
limits=np.array([xlowerlim, xupperlim, ylowerlim, yupperlim])
'''
Calling functions
However, this can be better done by creating a class and 
defining an object of the class. All these functions can initialized as the
member functions of the class.
'''
print('\n Calling functions')
no_of_vtu_files=get_vtu_num(path)

generating_structured_coordinates_meshgrid(num_x, num_y, limits)

'''
The next line will take hours or may be days
So if you just want to try uncomment this line

no_of_vtu_files=2
'''
read_vtu_and_convert_to_structured(path,no_of_vtu_files, limits)

'''
The number of timesteps used in the original PDE-FIND is just 101 out of which they choosed 60 timesteps
Here we have 2002 timesteps.

So we can choose to run method read_vtu_and_convert_to_structured for just 100 files may be
For this the method read_vtu_and_convert_to_structured(path,no_of_vtu_files, limits)
in line 55 is needed to be modified as follows

Oringial:
for n in range(0, vtu_num):
    
Modify:
vtu_start = 300
vtu_stop = 400
for n in range(vtu_start, vtu_stop):    
    
Actually this can be simply done by using function overloading for different circumstances.
Overload 1: When read_vtu_and_convert_to_structured is to be run on all timesteps.
read_vtu_and_convert_to_structured(path,no_of_vtu_files, limits)
Overload 2: When read_vtu_and_convert_to_structured is to be run on specific timesteps only.
read_vtu_and_convert_to_structured(path,vtu_start, vtu-stop, limits)
But I am bit lazy guy. Sorry about that.
Also, its better to use the data of fully developed flow
'''

# Exporting dx, dy and dt
exporting_dxdydt(path)