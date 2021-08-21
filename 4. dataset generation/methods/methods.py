import os
import meshio
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt


def get_vtu_num(path):
    print('Computing the number of vtu files')
    # count the number of vtu files
    f_list = os.listdir(path) # list of all files in path
    vtu_num = 0
    for i in f_list:
        if  os.path.splitext(i)[1] == '.vtu':
        	vtu_num = vtu_num+1
    print('Number of vtu files found:', vtu_num)
    return vtu_num

def read_vtu_and_convert_to_structured(path, vtu_num, limits):
    '''
    
    Parameters
    ----------
    path : string
        path to the vtu files within the current directory.
    vtu_num : int
        number of vtu files.
    limits : Float array
        limits of x,y coordinates to trim the cylinder.

    Returns
    -------
    None.

    '''
    
    
    # Loading the trimmed structured coordinates meshgrid for the interpolation
    print('Loading the x and y meshgrids')
    coordinates_data=np.load('data/coordinates_data.npy')
    x_meshgrid = coordinates_data[:,:,0]
    y_meshgrid = coordinates_data[:,:,1]
    
    # Defining empty variables for storing u, v and P 
    num_y = np.shape(x_meshgrid)[0] # rows
    num_x = np.shape(x_meshgrid)[1] # columns
    # (num_x) x (num_y) is the meshgrid and every third index stores the meshgrid for each vtu file or timestep
    print('\n Creating empty array of meshgrids')
    U = np.zeros((num_y,num_x,vtu_num))
    V = np.zeros_like(U)
    P = np.zeros_like(U)
    
    # Loading each vtu file one by one
    print('Looping over each vtu file')
    #for n in range(0, vtu_num):
    vtu_start = 300
    vtu_stop = 400
    for n in range(vtu_start, vtu_stop):
        print('File: ', n+1,' of ',vtu_num)
        # In case there are large number of vtu files
        # if n%10==0:
        # print('File: ', n+1,' of ',vtu_num)
        filename = path + "/circle-2d-drag_" + str(n)+ ".vtu"# name of vtu files
        mesh = meshio.read(filename)
        Velocity=mesh.point_data['Velocity'][:,[0,1]] # extracting the x and y velocity
        Pressure=mesh.point_data['Pressure'] # Extracting pressure for each node
        nodal_coordinates=mesh.points[:,[0,1]]   # extracting x,y coordinates only
        '''
        Cut out the portion of the data consisting the cylinder in the original vtu file
        If the data doesn't consist any geometry, then simply comment the line
        which calls the method remove_useless_data()
        '''
        print('Trimming the unstructured data')
        nodal_coordinates, Velocity, Pressure=remove_useless_data(nodal_coordinates, Velocity, Pressure,limits)
        '''
        Generating the RBF model for trimmed data of each vtu file and using them to
        using them to predict the values of velocity and pressure on the trimmed meshgrid    
        '''
        print('Using RBF to predict the field variables\n')
        # For horizontal velocity (u)
        U[:,:,n] = convert_unstructured_to_structured(nodal_coordinates, Velocity[:,0], x_meshgrid,y_meshgrid) 
        # For vertical velocity (v)
        V[:,:,n] = convert_unstructured_to_structured(nodal_coordinates, Velocity[:,1], x_meshgrid,y_meshgrid)
        # For Pressure (P)
        P[:,:,n] = convert_unstructured_to_structured(nodal_coordinates, Pressure, x_meshgrid,y_meshgrid)
    
    # Saving the structured dataset after all vtu files are looped
    print('Saving the data')
    np.save('data/U.npy', U)
    np.save('data/V.npy', V)
    np.save('data/P.npy', P)
        
         
        
def remove_useless_data(nodal_coordinates, Velocity, Pressure,limits):
    # Cut out the portion of the data consisting the cylinder in the original vtu file
    # see computing_the_cutoff_index.py in the same directory
    xlowerlim =  limits[0]
    xupperlim =  limits[1]
    ylowerlim =  limits[2]
    yupperlim =  limits[3]
       
    # boolean to data to be trimmed
    
    bool_data=np.zeros((len(nodal_coordinates)),dtype=bool)
    # make sure the data is within x limit
    x_limit_bool=np.logical_and(nodal_coordinates[:,0]<xupperlim, nodal_coordinates[:,0]>xlowerlim)
    # make sure the data is within y limit
    y_limit_bool=np.logical_and(nodal_coordinates[:,1]<yupperlim, nodal_coordinates[:,1]>ylowerlim)
    # Using logical and to make sure the data is within both the limits
    trimmed_bool_array=np.logical_and(x_limit_bool,y_limit_bool)
    nodal_coordinates=nodal_coordinates[trimmed_bool_array]
    Velocity=Velocity[trimmed_bool_array]
    Pressure=Pressure[trimmed_bool_array]
    # returning trimmed arrays
    return nodal_coordinates, Velocity, Pressure    
    
def convert_unstructured_to_structured(nodal_coordinates, parameter, x_meshgrid, y_meshgrid):
    '''
    Generating the RBF model for trimmed data of each vtu file and using them to
    using them to predict the values of velocity and pressure on the trimmed meshgrid
    
    Parameters
    ----------
    nodal_coordinates : 2D array
        Unstructured trimmed nodal coordinates.
    parameter : n-dimensional array
        The ground truth for RBF interpolation.
    x_meshgrid and y_meshgrid : meshgrids
        Structured coordinates, on which the parameter is to be interpolated.

    Returns
    -------
    The interpolated values of the parameter on structured grid.
    '''
    
    RBF_model=RBF_interpolant(nodal_coordinates[:,0],nodal_coordinates[:,1],parameter)
    return RBF_prediction(RBF_model,x_meshgrid,y_meshgrid)
     
    
def generating_structured_coordinates_meshgrid(num_x, num_y, limits):
    '''
    Saving the coordinates of trimmed structured mesh, because it is going to be same for each vtu file

    Parameters
    ----------
    num_x : Float
        number of columns in the meshgrid.
    num_y : Float
        number of rows in the meshgrids.
    limits : 1D array
        Data exceeding this limit will get trimmed.
        xlowerlim =  limits[0]
        xupperlim =  limits[1]
        ylowerlim =  limits[2]
        yupperlim =  limits[3]

    Returns
    -------
    None.

    '''
    print('\nGenerating the trimmed structured mesh coordinates')
    xlowerlim =  limits[0]
    xupperlim =  limits[1]
    ylowerlim =  limits[2]
    yupperlim =  limits[3]
    
    x=np.linspace(xlowerlim,xupperlim,num_x)
    y=np.linspace(ylowerlim,yupperlim,num_y)
    
    x_meshgrid,y_meshgrid=np.meshgrid(x,y)
    # saving the meshgrids
    print('Saving the coordinates_data')
    coordinates_data=np.zeros((num_y,num_x,2)) # 2 for x and y data and num_x and num_y constitutes the meshgrid
    coordinates_data[:,:,0]=x_meshgrid  # adding the x_meshgrid
    coordinates_data[:,:,1]=y_meshgrid  # adding the y_meshgrid
    
    np.save('data/coordinates_data.npy',coordinates_data)
    print('Coordinates data saved')
    

def RBF_interpolant(x,y,values):
    '''
    RBF interpolation and prediction
    Parameters
    ----------
    x : n-dimensional array
        Array of x-ccordinates.
    y : n-dimensional array
        Array of y-coordinates.
    values : n-dimensional array
        Array of the ground truth.

    Returns
    -------
    TYPE: RBF Model
        Returns the RBF-model.

    '''
    return Rbf(x,y,values) # all these arrays should have same size
    
    
def RBF_prediction(RBF_model,x_meshgrid,y_meshgrid):
    '''
    Interpolates values on structured grid using the RBF-model passed to the method.

    Parameters
    ----------
    RBF_model : method
        Trained model.
    x_meshgrid, y_meshgrid : meshgrids
        Structured coordinates, on which the parameter is to be interpolated.
    
    Returns
    -------
    meshgrid
        interpolated values based on the RBF-model trained.

    '''
    return RBF_model(x_meshgrid,y_meshgrid)

