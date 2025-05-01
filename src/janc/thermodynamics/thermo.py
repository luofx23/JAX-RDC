"""
The module extracts thermodynamic parameters 
from Cantera’s database and employs dedicated functions 
to compute critical thermodynamic state variables for gas mixtures: 
gas constant (R), enthalpy (h), specific heat ratio (gamma) and et al.

dependencies: jax & cantera(python version)
"""

import jax.numpy as jnp
from ..preprocess import nondim

R = None
gamma = None

def set_thermo(thermo_config,nondim_config=None):
    global gamma,R
    gamma = thermo_config['gamma']
    R = thermo_config['R']



    
    
    
