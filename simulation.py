import numpy as np
import scipy.sparse.linalg as spala 

from functions import *
from data_processing import *

def simulation(const, bc, LP, data):
    counter = 0
    
    while True:
        counter += 1
        simulation_step(const, bc, LP, data)
        
        if counter == 1:
            print('Iteration number: ' + str(counter))
            plot_system(const, bc , LP, data, False)
        
        equilibrium = is_stable(counter)
        
        if equilibrium:
            print('Iteration number: ' + str(counter))
            plot_system(const, bc , LP, data, False)
            print('Equilibrium has been reached after ' + str(counter) + ' iterations.')
            return data
        if counter%50 == 0:
            print('Iteration number: ' + str(counter))
            plot_system(const, bc , LP, data, False)
        if counter == np.ceil(const.tf/const.dt):
            print('Iteration number: ' + str(counter))
            plot_system(const, bc , LP, data, False)
            print('Maximum number of iterations (' + str(counter) + ') has been reached.')
            return data
            
def is_stable(counter): # NEEDS TO BE DEFINED
    return False
    #if counter == :
        #return True
    

def simulation_step(const, bc, LP, data):
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
        
    # Implicit viscosity 
    u = LP.Lu_factor(np.reshape(data.U + const.Ubc,(-1,), order='F')) #spala.spsolve(LP.Lu, np.reshape(data.U + const.Ubc,(-1,), order='F'))
    data.U = np.reshape(u,(const.nx-1, const.ny), order='F')
    v = LP.Lv_factor(np.reshape(data.V + const.Vbc,(-1,), order='F')) #spala.spsolve(LP.Lv, np.reshape(data.V + const.Vbc,(-1,), order='F'))
    data.V = np.reshape(v,(const.nx, const.ny-1), order='F')
    
    # Pressure correction
    grad_U = np.diff(np.vstack((bc.uW, data.U, bc.uE)), n=1, axis=0)/const.hx +\
             np.diff(np.vstack((bc.vS, data.V.T, bc.vN)).T, n=1, axis=1)/const.hy
    rhs = np.reshape(grad_U,(-1,), order='F')/const.dt 
    p = -LP.Lp_factor(rhs) #spala.spsolve(LP.Lp, rhs)
    data.P = np.reshape(p,(const.nx,const.ny), order='F')    
    data.U = data.U - np.diff(data.P, n=1, axis=0)/const.hx*const.dt  
    data.V = data.V - np.diff(data.P, n=1, axis=1)/const.hy*const.dt
    
    return data
