# -*- coding: utf-8 -*-
"""
Created on Mon May  3 23:01:44 2021

@author: Prakhar Sharma
"""
"""
builds a matrix with columns representing polynomials up to degree P of all variables

This is used when we subsample and take all the derivatives point by point or if there is an 
extra input (Q in the paper) to put in.

input:
    data: column 0 is U, and columns [1:end] are Q
    derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
    derivatives_description: description of what derivatives have been passed in
    P: max power of polynomial function of U to be included in Theta

returns:
    Theta = Theta(U,Q)
    descr = description of what all the columns in Theta are
"""

n,d = data.shape
m, d2 = derivatives.shape
if n != m: raise Exception('dimension error')
if data_description is not None: 
    if len(data_description) != d: raise Exception('data descrption error')

# Create a list of all polynomials in d variables up to degree P
rhs_functions = {}
f = lambda x, y : np.prod(np.power(list(x), list(y)))
powers = []            
for p in range(1,P+1):
        size = d + p - 1
        for indices in itertools.combinations(range(size), d-1):
            starts = [0] + [index+1 for index in indices]
            stops = indices + (size,)
            powers.append(tuple(map(operator.sub, stops, starts)))
for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]

# First column of Theta is just ones.
Theta = np.ones((n,1), dtype=np.complex64)
descr = ['']

# Add the derivaitves onto Theta
for D in range(1,derivatives.shape[1]):
    Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
    descr.append(derivatives_description[D])
    
# Add on derivatives times polynomials
for D in range(derivatives.shape[1]):
    for k in rhs_functions.keys():
        func = rhs_functions[k][0]
        new_column = np.zeros((n,1), dtype=np.complex64)
        for i in range(n):
            new_column[i] = func(data[i,:])*derivatives[i,D]
        Theta = np.hstack([Theta, new_column])
        if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
        else:
            function_description = ''
            for j in range(d):
                if rhs_functions[k][1][j] != 0:
                    if rhs_functions[k][1][j] == 1:
                        function_description = function_description + data_description[j]
                    else:
                        function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
            descr.append(function_description + derivatives_description[D])
