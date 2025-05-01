import jax.numpy as jnp
from ..solver import aux_func
from ..grid import read_grid
from ..thermodynamics import thermo

def left(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,2:3,:],U_bd[:,1:2,:],U_bd[:,0:1,:]],axis=1)
    u = U_bd_ghost[1:2]/U_bd_ghost[0:1]
    v = U_bd_ghost[2:3]/U_bd_ghost[0:1]
    U_n = u*read_grid.nx_L + v*read_grid.ny_L
    u_n = U_n*read_grid.nx_L
    v_n = U_n*read_grid.ny_L
    u_t = u - u_n
    v_t = v - v_n
    u = u_t - u_n
    v = v_t - v_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([u,v],axis=0))
    return U_bd_ghost

def right(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:,:],U_bd[:,-2:-1,:],U_bd[:,-3:-2,:]],axis=1)
    u = U_bd_ghost[1:2]/U_bd_ghost[0:1]
    v = U_bd_ghost[2:3]/U_bd_ghost[0:1]
    U_n = u*read_grid.nx_R + v*read_grid.ny_R
    u_n = U_n*read_grid.nx_R
    v_n = U_n*read_grid.ny_R
    u_t = u - u_n
    v_t = v - v_n
    u = u_t - u_n
    v = v_t - v_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([u,v],axis=0))
    return U_bd_ghost

def bottom(U_bd):
    gamma = thermo.gamma
    nx_B = read_grid.nx_B
    ny_B = read_grid.ny_B
    rho,u,v,p,a = aux_func.U_to_prim(U_bd[:,:,0:1])
    vn = u*read_grid.nx_B + v*read_grid.ny_B
    rho_cor = (((gamma-1)**2*(vn - 2*a/(gamma-1))**2*(rho**gamma))/(4*gamma*p))**(1/(gamma-1))
    u_cor = u*(ny_B)**2 - v*(nx_B*ny_B)
    v_cor = -u*(nx_B*ny_B) + v*(nx_B)**2
    p_cor = p*(rho_cor**gamma)/(rho**gamma)
    U_bc = jnp.concatenate([rho_cor,rho_cor*u_cor,rho_cor*v_cor,p_cor/((gamma-1))+0.5*rho_cor*(u_cor**2+v_cor**2)],axis=0)
    U_bd_ghost = jnp.concatenate([U_bc,U_bc,U_bc],axis=2)
    return U_bd_ghost

def up(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,-1:],U_bd[:,:,-2:-1],U_bd[:,:,-3:-2]],axis=2)
    u = U_bd_ghost[1:2]/U_bd_ghost[0:1]
    v = U_bd_ghost[2:3]/U_bd_ghost[0:1]
    U_n = u*read_grid.nx_U + v*read_grid.ny_U
    u_n = U_n*read_grid.nx_U
    v_n = U_n*read_grid.ny_U
    u_t = u - u_n
    v_t = v - v_n
    u = u_t - u_n
    v = v_t - v_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([u,v],axis=0))
    return U_bd_ghost