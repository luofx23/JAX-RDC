import jax.numpy as jnp
from jax import jit
from aux_func import update_aux,source_terms,U_to_prim,aux_to_thermo
from flux import weno5
from chemical import solve_implicit_rate
from boundary import read_boundary_params,boundary_conditions
from tqdm import tqdm

def rhs(U,aux,dx,dy):
    U_with_ghost,aux_with_ghost = boundary_conditions(U,aux)
    return weno5(U_with_ghost,aux_with_ghost,dx,dy) + source_terms(U, aux)

def CFL(U,aux,dx,cfl=0.20):
    _,u,_,_,_,a = U_to_prim(U,aux)
    c = jnp.max(jnp.abs(u) + a)
    dt = cfl/c*dx
    return dt

@jit
def TVD_RK3(U,aux,dx,dy,dt):
    U1 = U + dt * rhs(U,aux,dx,dy)
    aux1 = update_aux(U1, aux)
    
    U2 = 3/4*U + 1/4 * (U1 + dt * rhs(U1,aux1,dx,dy))
    aux2 = update_aux(U2,aux1)
    
    U3 = 1/3*U + 2/3 * (U2 + dt * rhs(U2,aux2,dx,dy))
    aux3 = update_aux(U3,aux2)
    return U3,aux3

@jit
def implicit_chem_source(U,aux,dt):
    _,T = aux_to_thermo(U,aux)
    rho = U[0:1]
    Y = U[4:]/rho
    drhoY = solve_implicit_rate(T,rho,Y,dt)

    p1 = U[0:4,:,:]
    p2 = U[4:,:,:] + drhoY
    U_new = jnp.concatenate([p1,p2],axis=0)
    aux_new = update_aux(U_new,aux)
    return U_new,aux_new

@jit
def time_step(U,aux,dx,dy,cfl):
    dt = CFL(U,aux,dx,cfl)
    U_adv,aux_adv = TVD_RK3(U,aux,dx,dy,dt)
    U,aux = implicit_chem_source(U_adv,aux_adv,dt)
    return U,aux

class RDC_Simulator:
    def __init__(self,grid,boundary,set):
        read_boundary_params(boundary)
        self.dx = grid['Lx']/grid['nx']
        self.dy = grid['Ly']/grid['ny']
        self.cfl = set['CFL']
    
    def forward(self,U,aux,nt):
        for step in tqdm(range(nt), desc="progress", unit="step"):
              U, aux = time_step(U,aux,self.dx,self.dy,self.cfl)
        return U, aux
        

    


