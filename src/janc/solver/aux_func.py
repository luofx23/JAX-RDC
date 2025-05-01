import jax.numpy as jnp
from ..thermodynamics import thermo

user_source = None

def set_source_terms(user_set):
    global user_source
    
    def zero_source_terms(U,theta=None):
        return jnp.zeros_like(U)
    
    if user_set is None:
        user_source = zero_source_terms
    
    else:    
        if user_set['self_defined_source_terms'] is not None:
            user_source  = user_set['self_defined_source_terms']
        else:
            user_source = zero_source_terms
        

def source_terms(U,theta=None):
    return user_source(U,theta)


def U_to_prim(U):
    state = U
    rho = state[0:1,:,:]
    u = state[1:2,:,:]/rho
    v = state[2:3,:,:]/rho
    p = (thermo.gamma-1)*(state[3:4,:,:]-0.5*rho*(u**2+v**2))
    a = jnp.sqrt(thermo.gamma*p/rho)
    return rho,u,v,p,a



