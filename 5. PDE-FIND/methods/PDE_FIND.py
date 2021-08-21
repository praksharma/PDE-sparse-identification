# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:21:28 2021

@author: Prakhar Sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import operator

'''
PDE-FIND for identifying Navier-Stokes

Samuel Rudy, 2016
 
All these functions are part of the PDE-FIND codes
https://github.com/snagcliffs/PDE-FIND
I have slightly modified some of the things to promote modularity.
'''
def perform_SVD(P, U, V):
    '''
    SVD for dimensionality reduction. The PDE-FIND also runs very well with reduced basis.

    Parameters
    ----------
    P : n-dimensional array
        pressure meshgrid.
    U : n-dimensional array
        X-velocity meshgrid.
    V : n-dimensional array
        Y-velocity meshgrid.

    Returns
    -------
    None.

    '''
    uw,sigmaw,vw = np.linalg.svd(P, full_matrices=False)
    uu,sigmau,vu = np.linalg.svd(U, full_matrices=False)
    uv,sigmav,vv = np.linalg.svd(V, full_matrices=False)
    # This plot would suggest the most dominated coherent modes that describes the dynamics
    plt.semilogy(sigmaw)
    plt.semilogy(sigmau)
    plt.semilogy(sigmav)
    print('SVD successfully done')
    
    
def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    
    """
    carrying the numerical derivative using the polynomial fitting.

    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """
    #print('u=',np.shape(u))
    #print('x=',np.shape(x),'\n')
    n = len(x)
    if index == None: index = (n-1)/2

    # Fit to a Chebyshev polynomial
    # better conditioned than normal polynomials
    poly = np.polynomial.chebyshev.Chebyshev.fit(x,u,deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1,diff+1):
        derivatives.append(poly.deriv(m=d)(x[int(index)]))
        
    return derivatives

def choose_random_points(num_xy, num_t, boundary_y, boundary_x, n, m):
    '''
    Sample a collection of data points, stay away from edges so I can just use interpolation or FDM.

    Parameters
    ----------
    num_xy : int
        Number of x and y points to be taken randomly in the collection.
    num_t : int
        Number of timesteps to be taken in the collection.
    boundary_y : int
        Selected points in y axis [boundary_y:m-boundary_y].
    boundary_x : int
        Selected points in x axis [boundary_x:n-boundary_x]
    n : int
        Number of columns or x-coordinates.
    m : int
        Number of rows or y-coordinates.

    Returns
    -------
    points : dictionary to store the collection
        Stores [x, y, t]
    '''
    print('Picking up random points from the data')
    # Sample a collection of data points, stay away from edges so I can just use centered finite differences.
    # The coordinates is chosen randomly from the sliced data. That is why random.choice is used here
    np.random.seed(0) # generates same set of random numbers each time
    
    #num_xy = 5000 # number of x and y points to be taken randomly in the collection
    #num_t = 60  # number of timesteps to be taken in the collection
    num_points = num_xy * num_t  # total number of points in the collection
    # We can't take derivative of the corner data using polynomial interpolation
    #boundary_y = 5 # Selected points in y axis [boundary_y:m-boundary_y]
    #boundary_x = 11 # Selected points in x axis [boundary_x:n-boundary_x]
    # Note boundary_x and boundary_y are not coordinates. These are data points between (0,5000)
    # These boundaries depends on how many points we have taken in consideration while using the polynomial interpolation
    points = {}  # dictionary to store the collection
    count = 0
    
    for p in range(num_xy): # for every point in our random data points collection
        # selecting random coordinates within the boundary. see the figure below.
        # These are the points that we will be used for computing thr derivative and for STR
        x = np.random.choice(np.arange(boundary_x,n-boundary_x),1)[0]    # randomly choose any x coordinate
        y = np.random.choice(np.arange(boundary_y,m-boundary_y),1)[0]    # randomly choose any y coordinate 
        # the following loop would insert same x and y coordinates for each time step
        for t in range(num_t): # all these point coexist in every timestep. thus we can compute the time derivative very easily.
            points[count] = [x,y,t+max(boundary_y, boundary_x)] 
            # t ranges from max(boundary_y, boundary_x) to num_t+max(boundary, boundary_x)
            # t + something ensures that we don't select first few timesteps of the whole dataset. here we have 100 timesteps. so t+max(boundary_y, boundary_x) ranges from 0+11 to 60+11
            # ensure that max(boundary_y, boundary_x)!<0 && num_t+max(boundary_y, boundary_x)!>steps
            # The t starts from max(boundary, boundary_x) because we need previous time steps for partial derivative wrt time
            count = count + 1
    
    '''
                           m
                 --------------------
                 |                  |
    boundary_x   |      domain      |  n
                 |                  |
                 --------------------
                       boundary_y 
    
    '''
    return points

def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an 
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
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
    Theta = np.ones((n,1), dtype=np.float64)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n,1), dtype=np.float64)
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

    return Theta, descr

def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = False):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.

    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=None)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print ("Optimal tolerance:", tol_best)

    return w_best

def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.float64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=None)[0]
    else: w = np.linalg.lstsq(X,y)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
    
    
def print_pde(w, rhs_description, ut = 'u_t'):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print (pde)
