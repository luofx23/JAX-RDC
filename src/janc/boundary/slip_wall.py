import jax.numpy as jnp
from ..grid import read_grid

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
    U_bd_ghost = jnp.concatenate([U_bd[:,:,2:3],U_bd[:,:,1:2],U_bd[:,:,0:1]],axis=2)
    u = U_bd_ghost[1:2]/U_bd_ghost[0:1]
    v = U_bd_ghost[2:3]/U_bd_ghost[0:1]
    U_n = u*read_grid.nx_B + v*read_grid.ny_B
    u_n = U_n*read_grid.nx_B
    v_n = U_n*read_grid.ny_B
    u_t = u - u_n
    v_t = v - v_n
    u = u_t - u_n
    v = v_t - v_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([u,v],axis=0))
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