import numpy as np
import scipy.sparse as spa

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

def gamma_calc(const, U, V):
    """
    Calculates gamma factor
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    U : 2D array
        velocity components in x-direction in grid
    V : 2D array
        velocity components in y-direction in grid
    
    Returns
    -------
    float
        parameter for transition between centered differencing and upwinding
        
    """
    return np.min((1.2*const.dt*np.max((np.max((np.abs(U)/const.hx)),
                                        np.max((np.abs(V)/const.hy)))), 1))

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
    if situation == 'lid':
        bc.uN = const.x*0 + 1
        bc.uE = ave(const.y, 'h')*0 
        bc.uS = const.x*0
        bc.uW = ave(const.y, 'h')*0

        bc.vN = ave(const.x, 'h')*0 
        bc.vW = const.y*0
        bc.vS = ave(const.x, 'h')*0
        bc.vE = const.y*0
        
    elif situation == 'horizontal_tube':
        bc.uN = const.x*0 + 1
        bc.uE = ave(const.y, 'h')*0 + 1 
        bc.uS = const.x*0 + 1
        bc.uW = ave(const.y, 'h')*0 + 1

        bc.vN = ave(const.x, 'h')*0 
        bc.vW = const.y*0
        bc.vS = ave(const.x, 'h')*0
        bc.vE = const.y*0
        
    elif situation == 'vertical_tube':
        bc.uN = const.x*0
        bc.uE = ave(const.y, 'h')*0 
        bc.uS = const.x*0
        bc.uW = ave(const.y, 'h')*0

        bc.vN = ave(const.x, 'h')*0 + 1
        bc.vW = const.y*0 + 1
        bc.vS = ave(const.x, 'h')*0 + 1
        bc.vE = const.y*0 + 1
        
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
    Ubc = const.dt/const.Re*(
        (np.vstack(
            (2*bc.uS[1:-1], np.zeros((const.nx-1,const.ny-2), dtype=float).T, 2*bc.uN[1:-1])).T
        )/const.hx**2 +
        np.vstack(
            (bc.uW, np.zeros((const.nx-3, const.ny), dtype=float), bc.uE)
        )/const.hy**2 )
    Vbc = const.dt/const.Re*(
        (np.vstack(
            (bc.vS, np.zeros((const.nx,const.ny-3), dtype = float).T, bc.vN)).T
        )/const.hx**2 + 
        np.vstack(
            (2*bc.vW[1:-1], np.zeros((const.nx-2, const.ny-1), dtype=float), 2*bc.vE[1:-1])
        )/const.hy**2 )

    return Ubc, Vbc

# -----------------------------------------------------------------------------------------------------------------------
# Developer functions
# -----------------------------------------------------------------------------------------------------------------------
    
def visualise_matrix(matrix, label_title):
    """ Visualises matrix by making 2D colourplot.
    
    Parameters
    ----------
    matrix : 2D numpy.ndarray, scipy.sparse.bsr.bsr_matrix, scipy.sparse.csr.csr_matrix or scipy.sparse.coo.coo_matrix
        
    """
    if isinstance(matrix, np.ndarray):
        matrix_plot = matrix
    elif isinstance(matrix, spa.bsr.bsr_matrix) or isinstance(matrix, spa.coo.coo_matrix) or isinstance(spa.csr.csr_matrix):
        matrix_plot = matrix.toarray()
    plt.imshow(matrix_plot)
    plt.colorbar()
    plt.clim(np.min(matrix_plot), np.max(matrix_plot))
    plt.title(label_title)
    plt.show()