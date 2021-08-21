# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:05:10 2021

@author: Prakhar Sharma
"""
import numpy as np
from methods.PDE_FIND import *
#%% Loading the data

data= np.load('dataset/derivatives/derivatives.npz')
keys=data.files

#p, u, v, ut, vt, px, py, ux, uy, vx, vy, uxx, uyy, vxx, vyy, uxy, vxy, num_points = [None]* len(keys)
candidate_functions = [None]* len(keys) #[p, u, v, ut, vt, px, py, ux, uy, vx, vy, uxx, uyy, vxx, vyy, uxy, vxy, num_points]
data['p']
for i in range (len(keys)):
    candidate_functions[i] = data[keys[i]]

p, u, v, ut, vt, px, py, ux, uy, vx, vy, uxx, uyy, vxx, vyy, uxy, vxy, num_points = candidate_functions


# In[9]:
# We don't need the time derivative (LHS) in the theta (RHS)
print('Building time series library')

# The momentum-equation

# time derivative of u or in x direction

# Later on I will concatenate both u_t and v_t in 1 theta
# Form a huge matrix using up to quadratic polynomials in all variables.
X_data = np.hstack([p,u,v])
X_ders = np.hstack([np.ones((num_points,1)), ux, uy, uxx, uxy, uyy, px])
X_ders_descr = ['','u_{x}', 'u_{y}','u_{xx}','u_{xy}','u_{yy}','p_{x}']
X, description = build_Theta(X_data, X_ders, X_ders_descr, 2, data_description = ['p','u','v'])
print ('Candidate terms for PDE[1]'+description[1:])


# ## Solve for $\xi$
# 
# TrainSTRidge splits the data up into 80% for training and 20% for validation. It searches over various tolerances in the STRidge algorithm and finds the one with the best performance on the validation set, including an $\ell^0$ penalty for $\xi$ in the loss function.

#

print('Printing the discovered PDE')

lam = 10**-5
d_tol = 5
c = TrainSTRidge(X,u_t,lam,d_tol)
print_pde(c, description, ut = 'u_t')


# time derivative of u or in x direction

# Later on I will concatenate both u_t and v_t in 1 theta
# Form a huge matrix using up to quadratic polynomials in all variables.
X_data = np.hstack([p,u,v])
X_ders = np.hstack([np.ones((num_points,1)), vx, vy, vxx, vxy, vyy, py])
X_ders_descr = ['','v_{x}', 'v_{y}','v_{xx}','v_{xy}','v_{yy}','p_{y}']
X, description = build_Theta(X_data, X_ders, X_ders_descr, 2, data_description = ['p','u','v'])
print ('Candidate terms for PDE[1]'+description[1:])


# ## Solve for $\xi$
# 
# TrainSTRidge splits the data up into 80% for training and 20% for validation. It searches over various tolerances in the STRidge algorithm and finds the one with the best performance on the validation set, including an $\ell^0$ penalty for $\xi$ in the loss function.

#

print('Printing the discovered PDE')

lam = 10**-5
d_tol = 5
c = TrainSTRidge(X,v_t,lam,d_tol)
print_pde(c, description, ut = 'v_t')

#
# err = abs(np.array([(0.009884-0.01)*100/0.01, (0.009902-0.01)*100/0.01, (-0.990371+1)*100, (-0.986629+1)*100]))
# print ("Error using PDE-FIND to identify Navier-Stokes:\n")
# print ("Mean parameter error:", np.mean(err), '%')
# print ("Standard deviation of parameter error:", np.std(err), '%')


# In[ ]:




