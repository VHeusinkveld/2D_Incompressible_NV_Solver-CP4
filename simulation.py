import numpy as np
import scipy.sparse.linalg as spala 

from functions import *
from data_processing import *
from operators import divergence, pressure_rhs, courant_number

def simulation(const, bc, obj, LP, data):
    """ Simulation of the 2D Navier-stokes equation
    
    Parameters
    ----------
    const : NameSpace
        simulation constants
    bc : NameSpace
        all boundary condition related variables
    obj : NameSpace
        all object related variables
    LP : NameSpace
        all laplace operators of the system
    data : NameSpace
        empty
        
    Returns
    -------
    data : NameSpace
        containing data arrays of all 'measured' quanities
        
    """
    counter = 0
    data.fig_counter = 0
    data.kin_energy = []
    while True:
        counter += 1
        
        # Solving system for U, V and P
        simulation_step(const, bc, obj, LP, data)
        
        # Prescribed boundary velocities stay fixed throughout this run.
        
        data.kin_energy = np.append(data.kin_energy, check_energy(data))
                
        if counter == 1:
            print('Iteration number: ' + str(counter))
            plot_system(const, bc, obj, LP, data)
        
        if counter%const.nsteps == 0:
            if counter%50 == 0:
                print('Iteration number: ' + str(counter))
            plot_system(const, bc, obj, LP, data)
        if counter == const.nt:
            print('Iteration number: ' + str(counter))
            
            if obj.sort != None:
                # Calculate force on object
                data.obj_F = obj_force(const, obj, data)
            
            const.save_fig = False
            plot_system(const, bc, obj, LP, data)
            print('Maximum number of iterations (' + str(counter) + ') has been reached.')
            return data
        
            
def simulation_step(const, bc, obj, LP, data):
    """ Generates data for one iteration
    
    Parameters
    ----------
    const : NameSpace
        simulation constants
    bc : NameSpace
        all boundary condition related variables
    obj : NameSpace
        all object related variables
    LP : NameSpace
        all laplace operators of the system
    data : NameSpace
        empty
        
    Returns
    -------
    data : NameSpace
        containing data arrays of all 'measured' quanities after one iteration
        
    """
    cfl = courant_number(const, bc, data)
    if not np.isfinite(cfl) or cfl > 1:
        raise ValueError(f'Explicit advection CFL is {cfl:.3g}; reduce dt and rebuild operators.')
    # Non linear terms
    Ue = np.vstack((bc.uW, data.U, bc.uE)).T
    Ue = np.vstack((2*bc.uS-Ue[0,:], Ue ,2*bc.uN-Ue[-1,:])).T

    Ve = np.vstack((bc.vS, data.V.T, bc.vN)).T
    Ve = np.vstack((2*bc.vW-Ve[0,:], Ve, 2*bc.vE-Ve[-1,:]))
    
    ## Average and difference matrices
    Ua = ave(Ue, 'h')
    Ud = np.diff(Ue, n=1, axis=1)/2
    Va = ave(Ve, 'v')
    Vd = np.diff(Ve, n=1, axis=0)/2
    
    ## Calculation of gamma (for smooth transition between centered differencing and upwinding)
    gamma = gamma_calc(const, data)
    
    ## Derivative matrices UV_x and UV_y
    UVx = np.diff((Ua*Va - gamma*np.abs(Ua)*Vd), axis=0)/const.hx
    UVy = np.diff((Ua*Va - gamma*Ud*np.abs(Va)), axis=1)/const.hy

    ## Average and difference matrices
    Ua = ave(Ue[:,1:-1], 'v')
    Ud = np.diff(Ue[:,1:-1], n=1, axis=0)/2
    Va = ave(Ve[1:-1,:], 'h')
    Vd = np.diff(Ve[1:-1,:], n=1, axis=1)/2
    
    ## Derivative matrices U^2_x and U^2_y
    U2x = np.diff((Ua**2 - gamma*np.abs(Ua)*Ud), axis=0)/const.hx
    V2y = np.diff((Va**2 - gamma*np.abs(Va)*Vd), axis=1)/const.hy

    ## Change in velocity applied
    data.U = data.U - const.dt*(UVy[1:-1,:] + U2x)
    data.V = data.V - const.dt*(UVx[:,1:-1] + V2y)    
    
    ## This if statement seems to be uneccesary
    ## but keep it for now 
    if obj.sort != None: 
        data.U = data.U*(obj.Ugrid==0)    
        data.V = data.V*(obj.Vgrid==0)
    
    # Implicit viscosity 
    if const.cholesky:
        u = LP.Lu_factor(np.reshape(data.U + bc.Ubc,(-1,), order='F')) 
    else:
        u = spala.spsolve(LP.Lu, np.reshape(data.U + bc.Ubc,(-1,), order='F'))
    data.U = np.reshape(u,(const.nx-1, const.ny), order='F')
    
    if const.cholesky:
        v = LP.Lv_factor(np.reshape(data.V + bc.Vbc,(-1,), order='F')) 
    else:
        v = spala.spsolve(LP.Lv, np.reshape(data.V + bc.Vbc,(-1,), order='F'))
    data.V = np.reshape(v,(const.nx, const.ny-1), order='F')
      
    if obj.sort != None:
        data.U = data.U*(obj.Ugrid==0)    
        data.V = data.V*(obj.Vgrid==0)
    
    # Pressure correction uses exactly the same blocked faces as velocity.
    rhs = pressure_rhs(LP, divergence(const, bc, data), const.dt)
    if const.cholesky:
        p = LP.Lp_factor(rhs)
    else:
        p = spala.spsolve(LP.Lp, rhs)
    data.P = np.reshape(p, (const.nx, const.ny), order='F')

    data.U = data.U - np.diff(data.P, n=1, axis=0)/const.hx*const.dt  
    data.V = data.V - np.diff(data.P, n=1, axis=1)/const.hy*const.dt
    
    if obj.sort != None:    
        data.U = data.U*(obj.Ugrid==0)    
        data.V = data.V*(obj.Vgrid==0)
    
    data.max_divergence = float(np.max(np.abs(divergence(const, bc, data))))
    data.cfl = float(courant_number(const, bc, data))
    if not np.isfinite(data.max_divergence):
        raise FloatingPointError('Non-finite velocity after pressure correction.')
    return data
