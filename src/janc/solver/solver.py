import jax.numpy as jnp
from jax import jit,vmap,pmap
from ..solver import aux_func
from .flux import weno5
from ..thermodynamics import thermo
from ..boundary import boundary
from ..grid import read_grid
from functools import partial


def CFL(field,dx,dy,cfl=0.20):
    U, aux = field[0:-2],field[-2:]
    _,u,v,_,_,a = aux_func.U_to_prim(U,aux)
    cx = jnp.max(abs(u) + a)
    cy = jnp.max(abs(v) + a)
    dt = jnp.minimum(cfl*dx/cx,cfl*dy/cy)
    return dt

def set_solver(thermo_set, boundary_set, grid_file = None, source_set = None, nondim_set = None, solver_mode='base'):
    thermo.set_thermo(thermo_set,nondim_set)
    boundary.set_boundary(boundary_set)
    read_grid.read_CGNS(grid_file)
    aux_func.set_source_terms(source_set)
    

    def rhs(U,theta=None):
        U_with_ghost = boundary.boundary_conditions(U,theta)
        physical_rhs = weno5(U_with_ghost) + aux_func.source_terms(U,theta)
        return physical_rhs

    
    def advance_flux(field,dt,theta=None):
        
        U = field
        U1 = U + dt * rhs(U,theta)
        U2 = 3/4*U + 1/4 * (U1 + dt * rhs(U1,theta))
        U3 = 1/3*U + 2/3 * (U2 + dt * rhs(U2,theta))
        field = U3
        
        return field
    

    @jit    
    def advance_one_step(field,dt,theta=None):
        field = advance_flux(field,dt,theta)
        return field
    
    print('solver is initialized successfully!')
    
    return advance_one_step,rhs
        

    


