#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pylab

data_path = './von_karman/'


# In[2]:


filenames = sorted([os.path.join(data_path,f) for f in os.listdir(data_path) if f[-3:] == 'plt'])
timesteps = len(filenames)

U = np.zeros((449,199,timesteps))
V = np.zeros((449,199,timesteps))
W = np.zeros((449,199,timesteps))


# In[3]:


start = time.time()

for timestep in range(timesteps):
    
    timestep_data = np.genfromtxt(filenames[timestep], delimiter=' ',skip_header=6)
    
    for i in range(449):
        
        for j in range(199):
            
            U[i,j,timestep] = timestep_data[i+449*j, 2]
            V[i,j,timestep] = timestep_data[i+449*j, 3]
            W[i,j,timestep] = timestep_data[i+449*j, 4]
            
    print('\rTimestep', timestep+1, 'of', timesteps, 'eta:',int((timesteps-timestep-1)*(timestep+1)/(time.time()-start)), end = 's')


# In[5]:


# plot the data
plt.figure(figsize=(15,8))

xx, yy = np.meshgrid(np.arange(449),np.arange(199))

for j in range(4):
    plt.subplot(2,2,j+1)
    plt.pcolor(xx,yy,W[:,:,20*j].T,cmap='coolwarm', vmin=-4, vmax=4)


# In[ ]:




