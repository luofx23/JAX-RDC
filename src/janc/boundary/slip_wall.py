import jax.numpy as jnp
from ..grid import read_grid

def left(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,2:3,:],U_bd[:,1:2,:],U_bd[:,0:1,:]],axis=1)
    U_norm = jnp.sqrt((U_bd_ghost[1:2]/U_bd_ghost[0:1])**2 + (U_bd_ghost[2:3]/U_bd_ghost[0:1])**2)
    Ux_n = U_norm*read_grid.nx_L
    Uy_n = U_norm*read_grid.ny_L
    Ux_t = (U_bd_ghost[1:2]/U_bd_ghost[0:1]) - Ux_n
    Uy_t = (U_bd_ghost[2:3]/U_bd_ghost[0:1]) - Uy_n
    Ux = Ux_t - Ux_n
    Uy = Uy_t - Uy_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([Ux,Uy],axis=0))
    return U_bd_ghost

def right(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:,:],U_bd[:,-2:-1,:],U_bd[:,-3:-2,:]],axis=1)
    U_norm = jnp.sqrt((U_bd_ghost[1:2]/U_bd_ghost[0:1])**2 + (U_bd_ghost[2:3]/U_bd_ghost[0:1])**2)
    Ux_n = U_norm*read_grid.nx_R
    Uy_n = U_norm*read_grid.ny_R
    Ux_t = (U_bd_ghost[1:2]/U_bd_ghost[0:1]) - Ux_n
    Uy_t = (U_bd_ghost[2:3]/U_bd_ghost[0:1]) - Uy_n
    Ux = Ux_t - Ux_n
    Uy = Uy_t - Uy_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([Ux,Uy],axis=0))
    return U_bd_ghost

def bottom(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,2:3],U_bd[:,:,1:2],U_bd[:,:,0:1]],axis=2)
    U_norm = jnp.sqrt((U_bd_ghost[1:2]/U_bd_ghost[0:1])**2 + (U_bd_ghost[2:3]/U_bd_ghost[0:1])**2)
    Ux_n = U_norm*read_grid.nx_B
    Uy_n = U_norm*read_grid.ny_B
    Ux_t = (U_bd_ghost[1:2]/U_bd_ghost[0:1]) - Ux_n
    Uy_t = (U_bd_ghost[2:3]/U_bd_ghost[0:1]) - Uy_n
    Ux = Ux_t - Ux_n
    Uy = Uy_t - Uy_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([Ux,Uy],axis=0))
    return U_bd_ghost

def up(U_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,-1:],U_bd[:,:,-2:-1],U_bd[:,:,-3:-2]],axis=2)
    U_norm = jnp.sqrt((U_bd_ghost[1:2]/U_bd_ghost[0:1])**2 + (U_bd_ghost[2:3]/U_bd_ghost[0:1])**2)
    Ux_n = U_norm*read_grid.nx_U
    Uy_n = U_norm*read_grid.ny_U
    Ux_t = (U_bd_ghost[1:2]/U_bd_ghost[0:1]) - Ux_n
    Uy_t = (U_bd_ghost[2:3]/U_bd_ghost[0:1]) - Uy_n
    Ux = Ux_t - Ux_n
    Uy = Uy_t - Uy_n
    U_bd_ghost = U_bd_ghost.at[1:3].set(U_bd_ghost[0:1]*jnp.concatenate([Ux,Uy],axis=0))
    return U_bd_ghost