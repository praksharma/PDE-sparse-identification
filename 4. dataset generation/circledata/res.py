import os
import vtktools
import numpy as np
import matplotlib.pyplot as plt
path='D:\\Desktop\\Dissertation\\von-karman\\circledata'
f_list = os.listdir(path) # all files in path
vtu_num = 0
for i in f_list:
    # os.path.splitext(os.listdir(path)[0]) will give filename and extension
   	if os.path.splitext(i)[1] == '.vtu':
   		vtu_num = vtu_num+1
print(vtu_num)
   	
for n in range(vtu_num): # for everyfile  
		filename = path + "/circle-2d-drag_" + str(n)+ ".vtu"# name of vtu files
		data = vtktools.vtu(filename)
		
		
		uvw = data.GetVectorField('Velocity')
		
		ui = np.hsplit(uvw,3)[0].T #velocity of x axis
		vi = np.hsplit(uvw,3)[1].T #velocity of y axis
		wi = np.hsplit(uvw,3)[2].T #velocity of z axis
		veli = np.hstack((ui,vi,wi)) #combine all into 1-d array
		print(n)
		vel = veli if n==0 else np.vstack((vel,veli))
w = vel[:,int(vel.shape[1]/3)*2:]
outputs = vel[:,:int(vel.shape[1]/3)*2] if np.all(w) == 0 else vel
np.save('Velocity(AE-64).npy',outputs)