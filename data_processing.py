import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spala

from functions import ave
from operators import cell_velocity

def plot_system(const, bc, obj, LP, data):
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
    plot of system
        
    """
    data.fig_counter += 1 
    fig = plt.figure(figsize=(const.lx*5, const.ly*5))
    
    ## Pressure 
    if const.plot_P:
        plt.contourf(const.X_ave, const.Y_ave, data.P.T, 50, cmap='bwr')      
        plt.colorbar()

    ## Streamlines from the actual velocity field, including through-flow.
    if const.plot_Q:
        uc, vc = cell_velocity(const, bc, data)
        if obj.sort is not None:
            uc = np.ma.array(uc, mask=obj.Pgrid)
            vc = np.ma.array(vc, mask=obj.Pgrid)
        plt.streamplot(ave(const.x, 'h'), ave(const.y, 'h'), uc.T, vc.T,
                       color='k', density=1.2, linewidth=0.6)

    ## Velocity
    if const.plot_U:  
        Ue = ave(np.vstack((bc.uW, data.U, bc.uE)).T, 'v')
        Ue = np.vstack((bc.uS, Ue, bc.uN))
        Ve = ave(np.vstack((bc.vS, data.V.T, bc.vN)).T, 'v')
        Ve = np.vstack((bc.vW, Ve, bc.vE)).T 
        v_len = np.sqrt(Ue**2 + Ve**2)

        speed_scale = max(float(np.max(v_len)), np.finfo(float).eps)
        Ue = Ue/speed_scale
        Ve = Ve/speed_scale
        #Ue[v_len!=0] = Ue[v_len!=0]/v_len[v_len!=0]
        #Ve[v_len!=0] = Ve[v_len!=0]/v_len[v_len!=0]

        plt.quiver(const.x[::const.rho_arrowx], 
                   const.y[::const.rho_arrowy], 
                   Ue[::const.rho_arrowy,::const.rho_arrowx], 
                   Ve[::const.rho_arrowy,::const.rho_arrowx])
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([np.min(const.X_ave), np.max(const.X_ave)])
    plt.ylim([np.min(const.Y_ave), np.max(const.Y_ave)])    
    
    if const.save_fig:
        # Save figures 
        fig_name = f'{data.fig_counter:04}' # Pad with zeros infront such that number has 4 digits
        plt.savefig(const.fig_dir + const.fig_id + fig_name, dpi=fig.dpi)
        plt.close()
    else:
        plt.show()
    