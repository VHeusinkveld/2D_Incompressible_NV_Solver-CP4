import numpy as np
import scipy as sci
import scipy.sparse as spa
import matplotlib.pyplot as plt
import math

from sksparse.cholmod import cholesky
from types import SimpleNamespace

# -----------------------------------------------------------------------------------------------------------------------
# Initialisation functions
# -----------------------------------------------------------------------------------------------------------------------

def initialisation(const, data):
    data.U = np.zeros((const.nx-1, const.ny), dtype = float)
    data.V = np.zeros((const.nx, const.ny-1), dtype = float)
    return data

# -----------------------------------------------------------------------------------------------------------------------
# Simulation functions
# -----------------------------------------------------------------------------------------------------------------------

def cholesky_decomposition(LP):
    """Make factor object which handels cholesky decomposition.
    
    To solve a sparse system Lx = b use
    factor = cholensky(L)
    x = factor(b) 
    """
    
    LP.Lp_factor = cholesky(LP.Lp)
    LP.Lu_factor = cholesky(LP.Lu)
    LP.Lv_factor = cholesky(LP.Lv)
    LP.Lq_factor = cholesky(LP.Lq)
    
    return LP
    
def laplacian(const, LP):
    LP.Lp = spa.kron(
        spa.eye(const.ny), laplace_m(const.nx, const.hx, 1), format="csc"
    ) + spa.kron(
        laplace_m(const.ny, const.hy, 1), spa.eye(const.nx), format="csc"
    )
    
    LP.Lu = spa.eye((const.nx - 1) * const.ny, format="csc") + (const.dt / const.Re) * (
        spa.kron(
            spa.eye(const.ny), laplace_m(const.nx - 1, const.hx, 2), format="csc"
        ) +
        spa.kron(
            laplace_m(const.ny, const.hy, 3), spa.eye(const.nx - 1), format="csc"
        )
    )
    
    LP.Lv = spa.eye((const.ny-1)*const.nx, format="csc") + (const.dt / const.Re) * (
        spa.kron(
            spa.eye(const.ny-1), laplace_m(const.nx, const.hx, 3), format="csc"
        ) +
        spa.kron(
            laplace_m(const.ny-1, const.hy, 2), spa.eye(const.nx), format="csc"
        )
    )
    
    LP.Lq = spa.kron(
        spa.eye(const.ny-1), laplace_m(const.nx-1, const.hx, 2), format="csc"
    ) + spa.kron(
        laplace_m(const.ny-1, const.hy, 2), spa.eye(const.nx-1), format="csc"
    )
    
    return LP
    
    
def ave(A, axis):
    """
    Calculates the horizontal (axis = 'h') or vertical (axis = 'v') average of matrix A.
    
    Parameters
    ----------
    A : 2D array
    axis : str
        defines the axis of averaging
    
    Returns
    -------
    average : 2D array
        average values of A
          
    """
    
    if axis == 'h':
        A = np.transpose(A)
        average = np.transpose((A[0:-1] + A[1:])/2)
    else:
        average = (A[0:-1] + A[1:])/2
        
    return average

def gamma_calc(const, data):
    """
    Calculates gamma factor
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    data.U : 2D array
        velocity components in x-direction in grid
    data.V : 2D array
        velocity components in y-direction in grid
    
    Returns
    -------
    float
        parameter for transition between centered differencing and upwinding
        
    """
    return np.min((1.2*const.dt*np.max((np.max((np.abs(data.U)/const.hx)),
                                        np.max((np.abs(data.V)/const.hy)))), 1))

def laplace_m(n, h, boundary_condition):
    """
    Calculates the laplacian operator matrices
    
    Parameters
    ----------
    n : int
        number of grid cells in certain direction
    h : float
        velocity components in x-direction in grid
    boundary_condition : int
        1, 2 or 3 which stand for the type of boundary condition:
        Neumann, Dirichlet for point on boundary and Dirichlet for boundary between two point, respectively
    
    Returns
    -------
    L : 2D array
        laplacian operator
        
    """
    
    diag_len = n 

    Lm = 2*np.ones(diag_len)      # n,m   elements
    Lu = -1*np.ones(diag_len-1)   # n,m+1 elements 
    Ld = -1*np.ones(diag_len-1)   # n,m-1 elements 
    Lm[[0,-1]] = boundary_condition
    
    # Compressed sparse column (CSC) format needed for Cholensky decomposition
    L = spa.diags([Lu, Lm, Ld], offsets=[1,0,-1], format="csc")/h**2
    return L

def set_BC(const, bc, situation):
    """
    Sets the boundary conditions within the system. *ONLY 'lid' IS CURRENTLY FUNCTIONAL!!!*
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    bc : Namespace
        floats defining in- and outflow speed of fluid
    situation : str
        string which defines the system situation 
    
    Returns
    -------
    bc : NameSpace
        floats defining in- and outflow speed of fluid
          
    """
    
    velocity = const.velocity
    
    if situation == 'lid':
        bc.uN = const.x*0 + velocity
        bc.uE = ave(const.y, 'h')*0 
        bc.uS = const.x*0
        bc.uW = ave(const.y, 'h')*0

        bc.vN = ave(const.x, 'h')*0 
        bc.vW = const.y*0
        bc.vS = ave(const.x, 'h')*0
        bc.vE = const.y*0
        
    elif situation == 'horizontal_tube':
        bc.uN = const.x*0 
        bc.uE = ave(const.y, 'h')*0 + velocity 
        bc.uS = const.x*0 
        bc.uW = ave(const.y, 'h')*0 + velocity

        bc.vN = ave(const.x, 'h')*0 
        bc.vW = const.y*0
        bc.vS = ave(const.x, 'h')*0
        bc.vE = const.y*0
        
    elif situation == 'vertical_tube':
        bc.uN = const.x*0
        bc.uE = ave(const.y, 'h')*0 
        bc.uS = const.x*0
        bc.uW = ave(const.y, 'h')*0

        bc.vN = ave(const.x, 'h')*0 + velocity
        bc.vW = const.y*0 + velocity
        bc.vS = ave(const.x, 'h')*0 + velocity
        bc.vE = const.y*0 + velocity

    elif situation == 'horizontal_sea':
        bc.uN = const.x*0 + velocity
        bc.uE = ave(const.y, 'h')*0 + velocity 
        bc.uS = const.x*0 + velocity
        bc.uW = ave(const.y, 'h')*0 + velocity

        bc.vN = ave(const.x, 'h')*0 
        bc.vW = const.y*0
        bc.vS = ave(const.x, 'h')*0
        bc.vE = const.y*0
        
    elif situation == 'airfoil':
        bc.uN = const.x*0
        bc.uE = ave(const.y, 'h')*0 
        bc.uS = const.x*0
        bc.uW = ave(const.y, 'h')*0

        bc.vN = ave(const.x, 'h')*0 
        bc.vW = const.y*0
        bc.vS = ave(const.x, 'h')*0
        bc.vE = const.y*0
        
    return bc


def set_BM(const, bc):
    """
    Sets the boundary matrices of the system.
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    bc : Namespace
        floats defining in- and outflow speed of fluid
    
    Returns
    -------
    bc : NameSpace
        floats defining in- and outflow speed of fluid
          
    """
    bc.Ubc = const.dt/const.Re*(
        (np.vstack(
            (2*bc.uS[1:-1], np.zeros((const.nx-1,const.ny-2), dtype=float).T, 2*bc.uN[1:-1])).T
        )/const.hx**2 +
        np.vstack(
            (bc.uW, np.zeros((const.nx-3, const.ny), dtype=float), bc.uE)
        )/const.hy**2 )
    bc.Vbc = const.dt/const.Re*(
        (np.vstack(
            (bc.vS, np.zeros((const.nx,const.ny-3), dtype = float).T, bc.vN)).T
        )/const.hx**2 + 
        np.vstack(
            (2*bc.vW[1:-1], np.zeros((const.nx-2, const.ny-1), dtype=float), 2*bc.vE[1:-1])
        )/const.hy**2 )

    return bc

'''
def update_BC(const, bc, data):
    # Calculate mean velocity at end of flow for horizontal tube (Only functional if object far from boundary)
    #U_meanE = np.mean(data.U[-1,:])
    
    bc.uN = np.append( np.append(bc.uW[-1], data.U[:,-1]), data.U[-1,-1] )
    bc.uE = data.U[-1,:]
    bc.uS = np.append( np.append(bc.uW[0], data.U[:,0]), data.U[-1,0] )
    
    bc.vN = data.V[:,-1]
    bc.vS = data.V[:,0]
    bc.vE = np.append( np.append(bc.vS[-1], data.V[-1,:]), bc.vN[-1] )
    
    bc = set_BM(const, bc)
    
    return bc
'''
'''
def set_BC(const, bc, situation):
    bc.uN = const.x*0
    bc.uE = ave(const.y, 'h')*0 
    bc.uS = const.x*0
    bc.uW = ave(const.y, 'h')*0

    bc.vN = ave(const.x, 'h')*0 
    bc.vW = const.y*0
    bc.vS = ave(const.x, 'h')*0
    bc.vE = const.y*0
    
    return bc
'''
def apply_forcing(const, bc, data):
    data.U = data.U + const.velocity # Exclude object area
    bc.uN = bc.uN + const.velocity
    bc.uS = bc.uS + const.velocity
    
    return bc, data

def update_BC(const, bc, data):
    # Horizontal flow component
    bc.uE = data.U[0,:]
    bc.uW = data.U[-1,:]
    
    # Vertical flow component
    bc.vE = np.append( np.append(bc.vS[-1], data.V[0,:]), bc.vN[-1] )
    bc.vW = np.append( np.append(bc.vS[0], data.V[-1,:]), bc.vN[0] )
    
    # Update Ubc and Vbc
    bc = set_BM(const, bc)
    
    '''
    bc.vN = data.V[:,-1]
    bc.vS = data.V[:,0]
    
    bc.vW = bc.vE
    bc.vE = np.append( np.append(bc.vS[-1], data.V[-1,:]), bc.vN[-1] )
    '''
    
    return bc

def object_boundary(obj):
    airfoil = obj.cgrid.T*1
    roll_down = np.roll(airfoil, -1, 0)
    roll_up = np.roll(airfoil, 1, 0)
    roll_left = np.roll(airfoil, -1, 1)
    roll_right = np.roll(airfoil, 1, 1)
    
    bottom = (roll_down - airfoil).T
    obj.bbound = np.where(bottom == 1)

    top = (roll_up - airfoil).T
    obj.tbound = np.where(top == 1)

    left = (roll_left - airfoil).T
    obj.lbound = np.where(left == 1)

    right = (roll_right - airfoil).T
    obj.rbound = np.where(right == 1)
    
    obj.bound = np.where((roll_up + roll_down + roll_left + roll_right - 4*airfoil).T > 0)
    
    return obj

def laplace_obj(const, LP, obj):
    """WIP 
    Function expects object grids, and LP. Based on the grids LP gets modified
    such that nothing flows into the object or out of it. 
    """
    if obj.sort != None:
        lp_row = const.nx
        lq_row = const.nx-1
        lu_row = const.nx-1
        lv_row = const.nx 

        x_factor = (const.dt / const.Re)/const.hx**2
        y_factor = (const.dt / const.Re)/const.hy**2   

        # Object in Lp 
        for obj_col, obj_row in np.transpose(obj.coord_P):
            m_start = obj_row*lp_row

            obj_n = m_start+obj_col
            obj_m = m_start+obj_col
            
            ## Takes care of 0 seting in Laplace            
            LP.Lp[obj_n+1, obj_m] = 0
            LP.Lp[obj_n-1, obj_m] = 0            
            
            LP.Lp[obj_n, obj_m-1] = 0
            LP.Lp[obj_n, obj_m+1] = 0
            
            LP.Lp[obj_n, obj_m-lp_row] = 0
            LP.Lp[obj_n, obj_m+lp_row] = 0
            
            LP.Lp[obj_n-lp_row, obj_m] = 0
            LP.Lp[obj_n+lp_row, obj_m] = 0
            
            ## Make matrix again postitive definite 
            ## Horizontal neighbours  
            LP.Lp[obj_n-1, obj_m-1] = LP.Lp[obj_n-1, obj_m-1] - 1
            LP.Lp[obj_n+1, obj_m+1] = LP.Lp[obj_n+1, obj_m+1] - 1
            ## Vertical neighbours 
            LP.Lp[obj_n-lp_row, obj_m-lp_row] = LP.Lp[obj_n-lp_row, obj_m-lp_row] - 1
            LP.Lp[obj_n+lp_row, obj_m+lp_row] = LP.Lp[obj_n+lp_row, obj_m+lp_row] - 1 
            
        # Object in Lq
        for obj_col, obj_row in np.transpose(obj.coord_P):
            m_start = obj_row*lq_row

            obj_n = m_start+obj_col
            obj_m = m_start+obj_col
            
            ## Takes care of 0 seting in Laplace            
            LP.Lq[obj_n+1, obj_m] = 0
            LP.Lq[obj_n-1, obj_m] = 0            
            
            LP.Lq[obj_n, obj_m-1] = 0
            LP.Lq[obj_n, obj_m+1] = 0
            
            LP.Lq[obj_n, obj_m-lq_row] = 0
            LP.Lq[obj_n, obj_m+lq_row] = 0
            
            LP.Lq[obj_n-lq_row, obj_m] = 0
            LP.Lq[obj_n+lq_row, obj_m] = 0
            
            ## Make matrix again postitive definite 
            ## Horizontal neighbours  
            LP.Lq[obj_n-1, obj_m-1] = LP.Lq[obj_n-1, obj_m-1] - 1
            LP.Lq[obj_n+1, obj_m+1] = LP.Lq[obj_n+1, obj_m+1] - 1
            ## Vertical neighbours 
            LP.Lq[obj_n-lq_row, obj_m-lq_row] = LP.Lp[obj_n-lq_row, obj_m-lq_row] - 1
            LP.Lq[obj_n+lq_row, obj_m+lq_row] = LP.Lp[obj_n+lq_row, obj_m+lq_row] - 1    

        # Object in Lu
        for obj_col, obj_row in np.transpose(obj.coord_U):
            m_start = obj_row*lu_row

            obj_n = m_start+obj_col
            obj_m = m_start+obj_col
            
            ## Takes care of 0 seting in Laplace            
            LP.Lu[obj_n+1, obj_m] = 0
            LP.Lu[obj_n-1, obj_m] = 0            
            
            LP.Lu[obj_n, obj_m-1] = 0
            LP.Lu[obj_n, obj_m+1] = 0
            
            LP.Lu[obj_n, obj_m-lu_row] = 0
            LP.Lu[obj_n, obj_m+lu_row] = 0
            
            LP.Lu[obj_n-lu_row, obj_m] = 0
            LP.Lu[obj_n+lu_row, obj_m] = 0
            
            ## Make matrix again postitive definite 
            ## Horizontal neighbours 
            #LP.Lu[obj_n-1, obj_m-1] = LP.Lu[obj_n-1, obj_m-1] - x_factor
            #LP.Lu[obj_n+1, obj_m+1] = LP.Lu[obj_n+1, obj_m+1] - x_factor
            
            ## Vertical neighbours 
            #LP.Lu[obj_n-lu_row, obj_m-lu_row] = LP.Lu[obj_n-lu_row, obj_m-lu_row] - y_factor
            #LP.Lu[obj_n+lu_row, obj_m+lu_row] = LP.Lu[obj_n+lu_row, obj_m+lu_row] - y_factor  

        # Object in Lv
        for obj_col, obj_row in np.transpose(obj.coord_V):
            m_start = obj_row*lv_row

            obj_n = m_start+obj_col
            obj_m = m_start+obj_col
            
            ## Takes care of 0 seting in Laplace            
            LP.Lv[obj_n+1, obj_m] = 0
            LP.Lv[obj_n-1, obj_m] = 0            
            
            LP.Lv[obj_n, obj_m-1] = 0
            LP.Lv[obj_n, obj_m+1] = 0
            
            LP.Lv[obj_n, obj_m-lv_row] = 0
            LP.Lv[obj_n, obj_m+lv_row] = 0
            
            LP.Lv[obj_n-lv_row, obj_m] = 0
            LP.Lv[obj_n+lv_row, obj_m] = 0    
            
            ## Make matrix again postitive definite 
            ## Horizontal neighbours
            #LP.Lv[obj_n-1, obj_m-1] = LP.Lv[obj_n-1, obj_m-1] - x_factor
            #LP.Lv[obj_n+1, obj_m+1] = LP.Lv[obj_n+1, obj_m+1] - x_factor
            ## Vertical neighbours 
            #LP.Lv[obj_n-lv_row, obj_m-lv_row] = LP.Lv[obj_n-lv_row, obj_m-lv_row] - y_factor
            #LP.Lv[obj_n+lv_row, obj_m+lv_row] = LP.Lv[obj_n+lv_row, obj_m+lv_row] - y_factor
    
    return LP


def creat_obj(const, obj):
    
    if obj.sort == 'circle':
        # Circular object
        scale = 1/4
        factorxy = 1 #const.lx/const.ly # Now the object is not scaled with the box ratio
        R = const.lx*scale/4
        cx, cy = const.lx/2, const.ly/2

        # Object for a centered grid
        ## Should have shape (nx, ny)
        obj.cgrid = ((const.X_ave-cx)**2 + (factorxy*const.Y_ave-factorxy*cy)**2 <= R**2).T
    
    
    if obj.sort == 'hemicircle':
        scale = 1/3
        factorxy = 1 #const.lx/const.ly # Now the object is not scaled with the box ratio
        R= const.lx*scale/4 # Radius of object
        cx, cy = const.lx/2, const.ly/2 # Centre of object

        fraction = 1
        translation_c = R*(1-fraction)
        theta = 70
        theta = math.radians(theta)

        obj.cgrid = (((const.X_ave-cx)**2 + (factorxy*const.Y_ave-factorxy*cy)**2 <= R**2)*(-(const.Y_ave - (cy + translation_c))*math.tan(theta) <= const.X_ave - (cx + translation_c))).T
        
    
    if obj.sort != None:
        # Translate to P, U and V grid 
        obj.Pgrid = np.zeros((const.nx, const.ny), dtype=float)
        obj.Ugrid = np.zeros((const.nx-1, const.ny), dtype=float)
        obj.Vgrid = np.zeros((const.nx, const.ny-1), dtype=float)
        obj.Qgrid = np.zeros((const.nx-1, const.ny-1), dtype=float)

        # P can center grid have a one to one correspondance
        obj.Pgrid[:,:] = obj.cgrid
    
        # Q grid is a shifted by one P grid
        obj.Qgrid[:,:] = (obj.Pgrid[:-1,:-1] + np.roll(obj.Pgrid[:-1,:-1], -1, axis = 0)) > 0
        obj.Qgrid[:,:] = (obj.Qgrid[:,:] + \
                              np.roll(obj.Qgrid[:,:], -1, axis = 1)) > 0
        
        # Create a 1 layer mesh around the center grid in vertical direction
        obj.Ugrid[:,:] = (obj.Pgrid[:-1,:] + np.roll(obj.Pgrid[:-1,:], -1, axis = 0) ) > 0
        obj.Ugrid[:,:] = (obj.Ugrid[:,:] + \
                              np.roll(obj.Ugrid[:,:], -1, axis = 1) + \
                              np.roll(obj.Ugrid[:,:], 1, axis = 1)) > 0
        
        # Create a 1 layer mesh around the center grid in horizontal direction
        obj.Vgrid[:,:] = (obj.Pgrid[:,:-1] + np.roll(obj.Pgrid[:,:-1], -1, axis = 1) ) > 0
        obj.Vgrid[:,:] = (obj.Vgrid[:,:] + \
                              np.roll(obj.Vgrid[:,:], -1, axis = 0) + \
                              np.roll(obj.Vgrid[:,:], 1, axis = 0)) > 0
    
        # These need to have the form:
        # np.array([x-values], [y-values])
        # ex: np.array([3, 4, 5, 9], [1, 2, 9, 5])
        obj.coord_P = np.where(obj.Pgrid == np.max(obj.Pgrid))
        obj.coord_U = np.where(obj.Ugrid == np.max(obj.Ugrid))
        obj.coord_V = np.where(obj.Vgrid == np.max(obj.Vgrid))
        obj.coord_Q = np.where(obj.Qgrid == np.max(obj.Qgrid))
    
    # Identify the boundaries of the object
    obj = object_boundary(obj)
    
    return obj

# -----------------------------------------------------------------------------------------------------------------------
# Developer functions
# -----------------------------------------------------------------------------------------------------------------------
    
def visualise_matrix(matrix, label_title):
    """ Visualises matrix by making 2D colourplot.
    
    Parameters
    ----------
    matrix : 2D numpy.ndarray, scipy.sparse.bsr.bsr_matrix, 
        scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix 
        or scipy.sparse.coo.coo_matrix
        
    """
   
    if isinstance(matrix, np.ndarray):
        matrix_plot = matrix
        plt.title(label_title)
        plt.imshow(matrix_plot)
        plt.colorbar()
        plt.clim(np.min(matrix_plot), np.max(matrix_plot))
        plt.show() 

    elif isinstance(matrix, spa.bsr.bsr_matrix) or \
         isinstance(matrix, spa.coo.coo_matrix) or \
         isinstance(matrix, spa.csr.csr_matrix) or \
         isinstance(matrix, spa.csc.csc_matrix):   
            
        # Checks if sparse matrix isn't to big to convert to array 
        if np.sqrt(np.size(matrix)) < 50:
            matrix_plot = matrix.toarray()    

            plt.imshow(matrix_plot)
            plt.colorbar()
            plt.clim(np.min(matrix_plot), np.max(matrix_plot))
        
        else:
            plt.spy(matrix)
       
        plt.title(label_title)
        plt.show()
        
    else:
        print('No valid matrix was given for visualisation.')

# -----------------------------------------------------------------------------------------------------------------------
# Code check functions
# -----------------------------------------------------------------------------------------------------------------------

def check_energy(data):
    return 1/2*(np.sum(np.sum(data.U**2)) + np.sum(np.sum(data.V**2)))