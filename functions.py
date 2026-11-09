import numpy as np
import scipy as sci
import scipy.sparse as spa
import matplotlib.pyplot as plt
import math

import scipy.sparse.linalg as spala
from matplotlib.path import Path
from operators import pressure_operator, obstacle_diffusion, velocity_masks

try:
    from sksparse.cholmod import cholesky
except ImportError:
    cholesky = None
from types import SimpleNamespace

# -----------------------------------------------------------------------------------------------------------------------
# Initialisation functions
# -----------------------------------------------------------------------------------------------------------------------

def initialisation(const, data):
    """
    Initialises the velocity arrays
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    data : Namespace
        empty Namespace
    
    Returns
    -------
    data : Namespace
        zero-filled velocity arrays
          
    """
    data.U = np.zeros((const.nx-1, const.ny), dtype = float)
    data.V = np.zeros((const.nx, const.ny-1), dtype = float)
    return data

# -----------------------------------------------------------------------------------------------------------------------
# Simulation functions
# -----------------------------------------------------------------------------------------------------------------------

def cholesky_decomposition(LP):
    """Cache sparse solves; use CHOLMOD when installed, otherwise SciPy LU.

    The historical function name is retained for notebook compatibility.
    """
    factor = cholesky if cholesky is not None else spala.factorized
    LP.Lp_factor = factor(LP.Lp)
    LP.Lu_factor = factor(LP.Lu)
    LP.Lv_factor = factor(LP.Lv)
    return LP

def laplacian(const, LP):
    """
    Calculates the laplacian operator matrices
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    LP : NameSpace
        empty
    
    Returns
    -------
    LP : Namespace
        contains all the laplace matrices of the system without object
        
    """
    blocked_u, blocked_v = velocity_masks(const, SimpleNamespace(sort=None))
    pressure_operator(const, LP, blocked_u, blocked_v)

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


# -----------------------------------------------------------------------------------------------------------------------
# Boundary condition functions
# -----------------------------------------------------------------------------------------------------------------------
def set_BC(const, bc, situation):
    """
    Sets fixed velocity boundaries. Periodic boundaries are not implemented.
    
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
    
    supported = {'lid', 'horizontal_tube', 'vertical_tube', 'horizontal_sea', 'airfoil'}
    if situation not in supported:
        raise ValueError(f'Unsupported boundary type {situation!r}; choose one of {sorted(supported)}')
    bc.kind = situation
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
        arrays with zero entries except for the last row/colom which defines the boundary
          
    """
    bc.Ubc = const.dt/const.Re*(
        (np.vstack(
            (2*bc.uS[1:-1], np.zeros((const.nx-1,const.ny-2), dtype=float).T, 2*bc.uN[1:-1])).T
        )/const.hy**2 +
        np.vstack(
            (bc.uW, np.zeros((const.nx-3, const.ny), dtype=float), bc.uE)
        )/const.hx**2 )
    bc.Vbc = const.dt/const.Re*(
        (np.vstack(
            (bc.vS, np.zeros((const.nx,const.ny-3), dtype = float).T, bc.vN)).T
        )/const.hy**2 +
        np.vstack(
            (2*bc.vW[1:-1], np.zeros((const.nx-2, const.ny-1), dtype=float), 2*bc.vE[1:-1])
        )/const.hx**2 )

    return bc


def update_BC(const, bc, data):
    """Keep prescribed boundaries fixed (retained for existing callers).

    Copying opposite interior rows is not a periodic MAC discretization: the
    pressure and diffusion operators must also wrap for periodic flow.
    """
    if getattr(bc, 'kind', None) not in {
            'lid', 'horizontal_tube', 'vertical_tube', 'horizontal_sea', 'airfoil'}:
        raise ValueError('Set a supported fixed boundary type with set_BC first.')
    return bc


# -----------------------------------------------------------------------------------------------------------------------
# Object functions
# -----------------------------------------------------------------------------------------------------------------------
def creat_obj(const, obj):
    """
    Creates the object in the system. This includes the object itself and its boundary points.
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    obj : Namespace
        Namespace with only the sort of the object. It will contain all specifications of the object after applying the creat_obj function.
    
    Returns
    -------
    obj : NameSpace
        updated version of obj
          
    """
    if obj.sort == 'circle':
        # Circular object
        scale = 1/4
        factorxy = 1 #const.lx/const.ly # Now the object is not scaled with the box ratio
        R = getattr(obj, 'radius', const.lx*scale/4)
        cx, cy = getattr(obj, 'center', (const.lx/2, const.ly/2))

        # Object for a centered grid
        ## Should have shape (nx, ny)
        obj.cgrid = ((const.X_ave-cx)**2 + (factorxy*const.Y_ave-factorxy*cy)**2 <= R**2).T
    
    
    if obj.sort == 'hemicircle':
        scale = 1/3
        factorxy = 2 #const.lx/const.ly # Now the object is not scaled with the box ratio
        R= const.lx*scale/4 # Radius of object
        cx, cy = const.lx/2, const.ly/2 # Centre of object

        fraction = 1
        translation_c = R*(1-fraction)
        theta = 70
        theta = math.radians(theta)

        obj.cgrid = (((const.X_ave-cx)**2 + (factorxy*const.Y_ave-factorxy*cy)**2 <= R**2)*(-(const.Y_ave - (cy + translation_c))*math.tan(theta) <= const.X_ave - (cx + translation_c))).T
        
    if obj.sort == 'naca':
        # Rasterize the continuous outline at pressure-cell centres. This avoids
        # gaps and angle-dependent thickness from rounding/filling boundary pixels.
        chord = getattr(obj, 'chord', min(const.lx, const.ly) / 2)
        cx, cy = getattr(obj, 'center', (const.lx / 2, const.ly / 2))
        theta = -np.deg2rad(obj.angle)
        samples = max(200, int(8 * chord / min(const.hx, const.hy)))
        x, y = naca4(obj.profile, np.linspace(0, 1, samples + 1))
        x, y = (np.asarray(x) - 0.5) * chord, np.asarray(y) * chord
        xr = x * np.cos(theta) - y * np.sin(theta) + cx
        yr = x * np.sin(theta) + y * np.cos(theta) + cy
        points = np.c_[const.X_ave.ravel(), const.Y_ave.ravel()]
        obj.cgrid = Path(np.c_[xr, yr]).contains_points(points).reshape(
            const.X_ave.shape).T
        obj.xcoord, obj.ycoord = xr / const.hx, yr / const.hy

    if obj.sort != None:
        obj.Pgrid = np.asarray(obj.cgrid, dtype=bool)
        if not np.any(obj.Pgrid):
            raise ValueError('Object is unresolved on this grid; refine the mesh.')
        if (np.any(obj.Pgrid[[0, -1], :]) or np.any(obj.Pgrid[:, [0, -1]])):
            raise ValueError('Objects must lie inside the domain, away from outer boundaries.')
        # A wall-normal velocity is blocked if either adjacent pressure cell is solid.
        obj.Ugrid = obj.Pgrid[:-1, :] | obj.Pgrid[1:, :]
        obj.Vgrid = obj.Pgrid[:, :-1] | obj.Pgrid[:, 1:]
        obj.Qgrid = (obj.Pgrid[:-1, :-1] | obj.Pgrid[1:, :-1] |
                     obj.Pgrid[:-1, 1:] | obj.Pgrid[1:, 1:])
        obj.coord_P = np.where(obj.Pgrid)
        obj.coord_U = np.where(obj.Ugrid)
        obj.coord_V = np.where(obj.Vgrid)
        obj.coord_Q = np.where(obj.Qgrid)

        # Identify the boundaries of the object
        obj = object_boundary(obj)
    
    return obj


def object_boundary(obj):
    """
    Identifies the object boundaries
    
    Parameters
    ----------
    obj : Namespace
        Namespace with all specifications of the object
    
    Returns
    -------
    obj : NameSpace
        updated version of obj. Now contains also the left, right, top and bottom boundary of the object.
          
    """
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


def obj_force(const, obj, data):
    """
    Calculates the force exerted on the object from the pressure distribution along the object.
    
    Parameters
    ----------
    const : NameSpace
        contains all system constants (incl. grid specifications)
    obj : Namespace
        Namespace with all specifications of the object
    data : Namespace
        Containing the pressure distribution in the system
    
    Returns
    -------
    force : float
        force exerted on the object in the flow
          
    """
    force = 0
    dx = const.lx/const.nx
    # Calculate pressure on object
    around = int(const.ny/10)
    
    for n, m in np.transpose(obj.tbound):
        force -= np.min(data.P[n,m:m+around])*dx
    
    for n, m in np.transpose(obj.bbound):    
        force += np.max(data.P[n,m-around:m])*dx

    return force


'''
def obj_force(const, obj, data):  
    force = 0
    dx = const.lx/const.nx
    # Calculate pressure on object
    for n, m in np.transpose(obj.tbound):
        force -= data.P[n,m]*dx
    
    for n, m in np.transpose(obj.bbound):    
        force += data.P[n,m]*dx

    return force
'''


def naca4(number, x, finite_TE = False):
    """
    Gives boundaries of the NACA 4-digit object.
    
    Parameters
    ----------
    number : str
        4-digit number specifying profile
        options: ['0009', '0012', '2412', 2414', '2415', '6409' , '0006', '0008', '0010', '0012', '0015']
    x : 1D-ndarray
        all x coordinates of the NACA profile
    finite_TE : Boolean
        True for finite thick TE
        False for zero thick TE
    
    Returns
    -------
    X : 1D ndarray
        x-coordinates of boundary NACA profile
    Y : 1D ndarray
        y-coordinates of boundary NACA profile
        
          
    """
    
    m = float(number[0])/100.0
    p = float(number[1])/10.0
    t = float(number[2:])/100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843

    if finite_TE:
        a4 = -0.1015 # For finite thick TE
    else:
        a4 = -0.1036 # For zero thick TE


    yt = [5*t*(a0*math.sqrt(xx)+a1*xx+a2*math.pow(xx,2)+a3*math.pow(xx,3)+a4*math.pow(xx,4)) for xx in x]

    xc1 = [xx for xx in x if xx <= p]
    xc2 = [xx for xx in x if xx > p]

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = [-xx for xx in yt]

        xc = xc1 + xc2
        zc = [0]*len(xc)
    else:
        yc1 = [m/math.pow(p,2)*xx*(2*p-xx) for xx in xc1]
        yc2 = [m/math.pow(1-p,2)*(1-2*p+xx)*(1-xx) for xx in xc2]

        zc = yc1 + yc2

        dyc1_dx = [m/math.pow(p,2)*(2*p-2*xx) for xx in xc1]
        dyc2_dx = [m/math.pow(1-p,2)*(2*p-2*xx) for xx in xc2]
        dyc_dx = dyc1_dx + dyc2_dx

        theta = [math.atan(xx) for xx in dyc_dx]

        xu = [xx - yy * math.sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yu = [xx + yy * math.cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

        xl = [xx + yy * math.sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yl = [xx - yy * math.cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

    X = np.concatenate((np.asarray(xu)[::-1], np.asarray(xl)[1:]))
    Z = np.concatenate((np.asarray(yu)[::-1], np.asarray(yl)[1:]))

    return X,Z


def laplace_obj(const, LP, obj):
    """Apply stationary no-slip obstacle faces and a compatible pressure solve.

    Call once on fresh operators from laplacian for each geometry.
    """
    blocked_u, blocked_v = velocity_masks(const, obj)
    pressure_operator(const, LP, blocked_u, blocked_v)
    if obj.sort is not None:
        LP.Lu = obstacle_diffusion(LP.Lu, blocked_u, 1,
                                  const.dt / const.Re / const.hy**2)
        LP.Lv = obstacle_diffusion(LP.Lv, blocked_v, 0,
                                  const.dt / const.Re / const.hx**2)
    return LP


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
    """
    Calculates the kinetic energy averaged per grid cell
    
    Parameters
    ----------
    data : Namespace
        Containing the pressure distribution in the system
    
    Returns
    -------
     : float
         kinetic energy averaged per grid cell
          
    """
    return 1/2*(np.sum(np.sum(data.U**2))/(np.shape(data.U)[0]*np.shape(data.U)[1]) + np.sum(np.sum(data.V**2))/(np.shape(data.V)[0]*np.shape(data.V)[1]))