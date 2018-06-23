import numpy as np
import scipy.sparse.linalg as spala 

from functions import *
from data_processing import *

def simulation(const, bc, obj, LP, data, situation):
    counter = 0
    data.fig_counter = 0
    data.kin_energy = []
    while True:
        counter += 1
        #if counter == 1:
            #bc, data = apply_forcing(const, bc, data)
            #print('uN', bc.uN, 'uE', bc.uE, 'uW', bc.uW, 'uS', bc.uS)
        #else:
        #    bc = update_BC(const, bc, data)    
        #if counter%10 == 0:
         #   bc, data = apply_forcing(const, bc, data)
         #   bc = update_BC(const, bc, data)

        # Solving system for U, V and P
        simulation_step(const, bc, obj, LP, data)
        
        # Periodic BC update
        bc = update_BC(const, bc, data) 
        
        data.kin_energy = np.append(data.kin_energy, check_energy(data))
                
        if counter == 1:
            print('Iteration number: ' + str(counter))
            #plot_system(const, bc, obj, LP, data)
        
        equilibrium = is_stable(counter)
        
        if equilibrium:
            print('Iteration number: ' + str(counter))
            plot_system(const, bc, obj, LP, data)
            print('Equilibrium has been reached after ' + str(counter) + ' iterations.')
            return data
        if counter%const.nsteps == 0:
            if counter%50 == 0:
                print('Iteration number: ' + str(counter))
            plot_system(const, bc, obj, LP, data)
        if counter == np.ceil(const.tf/const.dt):
            print('Iteration number: ' + str(counter))
            
            # Calculate force on object
            data.obj_F = obj_force(const, obj, data)
            
            const.save_fig = False
            plot_system(const, bc, obj, LP, data)
            print('Maximum number of iterations (' + str(counter) + ') has been reached.')
            return data
        
            
def is_stable(counter): # NEEDS TO BE DEFINED
    return False
    #if counter == :
        #return True
    

def simulation_step(const, bc, obj, LP, data):
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
    
    # Pressure correction
    grad_U = np.diff(np.vstack((bc.uW, data.U, bc.uE)), n=1, axis=0)/const.hx +\
             np.diff(np.vstack((bc.vS, (data.V).T, bc.vN)).T, n=1, axis=1)/const.hy
    
    if obj.sort != None:
        grad_U = grad_U*(obj.Pgrid==0)
    
    rhs = np.reshape(grad_U,(-1,), order='F')/const.dt 
    
    if const.cholesky:
        p = -LP.Lp_factor(rhs) 
    else:
        p = -spala.spsolve(LP.Lp, rhs)
    data.P = np.reshape(p,(const.nx,const.ny), order='F') 
    
    data.U = data.U - np.diff(data.P, n=1, axis=0)/const.hx*const.dt  
    data.V = data.V - np.diff(data.P, n=1, axis=1)/const.hy*const.dt
    
    #if obj.sort != None:
    #    data.P = data.P*(obj.Pgrid==0)
    
    if obj.sort != None:    
        data.U = data.U*(obj.Ugrid==0)    
        data.V = data.V*(obj.Vgrid==0)
    
    return data
