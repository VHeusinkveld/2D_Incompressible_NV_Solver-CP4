# -----------------------------------------------------------------------------------------------------------------------
# System constants
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
from types import SimpleNamespace
from functions import ave

def constants(self):
    """ Sets the system constants based on the set system parameters
    
    Parameters
    ----------
    self : NameSpace
        set simulation constants
        
    Returns
    -------
    self : NameSpace
        updated simulation constants
        
    """
    
    self.rho_arrowx = int(np.max((np.round(self.nx/(self.rho_arrow*self.lx)), 1)))
    self.rho_arrowy = int(np.max((np.round(self.ny/(self.rho_arrow*self.ly)), 1)))
    self.nt = np.ceil(self.tf/self.dt)
    self.dt = self.tf/self.nt;

    # Constuct grid
    self.x, self.y = np.linspace(0, self.lx, self.nx+1), np.linspace(0, self.ly, self.ny+1)
    self.hx, self.hy = self.lx/self.nx, self.ly/self.ny
    [self.X,self.Y] = np.meshgrid(self.y, self.x);
    [self.X_ave, self.Y_ave] = np.meshgrid(ave(self.x, 'h'), ave(self.y, 'h'))

    
    return self