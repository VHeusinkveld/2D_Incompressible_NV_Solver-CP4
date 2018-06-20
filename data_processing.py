import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spala

from functions import ave

def plot_system(const, bc , LP, data, SAVE):
    
    fig = plt.figure(figsize=(const.lx*5, const.ly*5))
    
    ## Pressure 
    plt.contourf(const.X_ave, const.Y_ave, data.P.T, 50, cmap='bwr')      
    plt.colorbar()

    ## Flow 
    rhs = np.reshape((np.diff(data.U, n=1, axis=1)/const.hy-np.diff(data.V, n=1, axis=0)/const.hx),(-1,), order='F')/const.dt
    if const.cholesky:
        q = LP.Lq_factor(rhs) 
    else: 
        q = spala.spsolve(LP.Lq, rhs)
    Q = np.zeros((const.nx+1,const.ny+1), dtype=float)
    Q[1:-1,1:-1] = np.reshape((q),(const.nx-1,const.ny-1), order='F')
    plt.contour(const.x, const.y, Q.T, 10, colors='k', linewidths=1)

    ## Velocity
    Ue = ave(np.vstack((bc.uW, data.U, bc.uE)).T, 'v')
    Ue = np.vstack((bc.uS, Ue, bc.uN))
    Ve = ave(np.vstack((bc.vS, data.V.T, bc.vN)).T, 'v')
    Ve = np.vstack((bc.vW, Ve, bc.vE)).T 
    v_len = np.sqrt(Ue**2 + Ve**2)
    
    Ue = Ue/np.max(v_len)
    Ve = Ve/np.max(v_len)
    
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
    plt.show()
    '''
    if SAVE:
        ## Save figures 
        #fig_name = f'{fig_counter:04}' # Pad with zeros infront such that number has 4 digits
        #plt.savefig('./movie_images/' + fig_name, dpi=fig.dpi)
        #plt.close()
    '''