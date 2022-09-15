# -*- coding: utf-8 -*-
"""
"""
import numpy as np

g = -9.81
# Air
rho = 1.225
mu = 1.81e-5

winddir = np.array((1,0,0))
z0 = 0.1
Uref = 0.0
zref = 1.5
kappa = 0.41

def wind_abl(z):
    # For a constant wind:
    # return Uref*winddir

    if z < 0.0:
        z = 0.0
    
    ustar = Uref*kappa/(np.log((zref+z0)/z0))
    u = ustar/kappa*np.log((z+z0)/z0)
    
    return u*winddir
