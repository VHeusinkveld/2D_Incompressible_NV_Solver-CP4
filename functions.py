import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt 

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
        bc.uN = const.x*0 #+ velocity
        bc.uE = ave(const.y, 'h')*0 #+ velocity 
        bc.uS = const.x*0 #+ velocity
        bc.uW = ave(const.y, 'h')*0 #+ velocity

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

def object_boundary(airfoil):
    roll_up = np.roll(airfoil, -1, 0)
    roll_down = np.roll(airfoil, 1, 0)
    roll_left = np.roll(airfoil, -1, 1)
    roll_right = np.roll(airfoil, 1, 1)
    
    return np.where(roll_up + roll_down + roll_left + roll_right - 4*airfoil > 0)
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