import jax.numpy as jnp
from ..thermodynamics import thermo
from ..solver import aux_func
def left(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    return U_bd_ghost

def right(U_bd):
    gamma = thermo.gamma
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:,:],U_bd[:,-2:-1,:],U_bd[:,-3:-2,:]],axis=1)
    rho,u,v,p,a = aux_func.U_to_prim(U_bd_ghost)
    rho = 2*rho[:,0:1,:]-rho[:,1:2,:]
    u = 2*u[:,0:1,:]-u[:,1:2,:]
    v = 2*v[:,0:1,:]-v[:,1:2,:]
    p = 2*p[:,0:1,:]-p[:,1:2,:]
    U_bc = jnp.concatenate([rho,rho*u,rho*v,p/(gamma-1)+0.5*rho*(u**2+v**2)],axis=0)
    U_bd_ghost = jnp.tile(U_bc,(1,3,1))
    
    return U_bd_ghost

def bottom(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    return U_bd_ghost

def up(U_bd):
    gamma = thermo.gamma
    U_bd_ghost = jnp.concatenate([U_bd[:,:,-1:],U_bd[:,:,-2:-1],U_bd[:,:,-3:-2]],axis=2)
    rho,u,v,p,a = aux_func.U_to_prim(U_bd_ghost)
    rho = 2*rho[:,:,0:1]-rho[:,:,1:2]
    u = 2*u[:,:,0:1]-u[:,:,1:2]
    v = 2*v[:,:,0:1]-v[:,:,1:2]
    p = 2*p[:,:,0:1]-p[:,:,1:2]
    U_bc = jnp.concatenate([rho,rho*u,rho*v,p/(gamma-1)+0.5*rho*(u**2+v**2)],axis=0)
    U_bd_ghost = jnp.tile(U_bc,(1,1,3))
    return U_bd_ghost

