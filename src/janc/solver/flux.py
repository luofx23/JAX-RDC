import jax.numpy as jnp
from jax import jit
from ..solver import aux_func
from ..grid import read_grid
from ..thermodynamics import thermo

p = 2
eps = 1e-6
C1 = 1 / 10
C2 = 3 / 5
C3 = 3 / 10


@jit
def splitFlux_LF(ixy, U):
    rho,u,v,p,a = aux_func.U_to_prim(U)
    rhoE = U[3:4,:,:]

    zx = (ixy == 1) * 1
    zy = (ixy == 2) * 1
    
    F = jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p)], axis=0)
    G = jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p)], axis=0)
    
    F_J = read_grid.J*(F*read_grid.dxi_dx+G*read_grid.dxi_dy)
    G_J = read_grid.J*(F*read_grid.deta_dx+G*read_grid.deta_dy)

    flux = zx*F_J + zy*G_J
    
    grad_xi = jnp.concatenate([read_grid.dxi_dx,read_grid.dxi_dy],axis=0)
    grad_eta = jnp.concatenate([read_grid.deta_dx,read_grid.deta_dy],axis=0)
    grad_xi_norm = jnp.sqrt(grad_xi[0:1]**2 + grad_xi[1:2]**2)
    grad_eta_norm = jnp.sqrt(grad_eta[0:1]**2 + grad_eta[1:2]**2)
    
    u_J = read_grid.dxi_dx*u + read_grid.dxi_dy*v
    v_J = read_grid.deta_dx*u + read_grid.deta_dy*v
    
    um = jnp.nanmax(abs(u_J) + a*grad_xi_norm)
    vm = jnp.nanmax(abs(v_J) + a*grad_eta_norm)
    theta = zx*um + zy*vm
    Hplus = 0.5 * (flux + theta * read_grid.J * U)
    Hminus = 0.5 * (flux - theta * read_grid.J * U)
    return Hplus, Hminus

@jit
def splitFlux_SW(ixy,U):
    zx_org = (ixy==1)*read_grid.dxi_dx + (ixy==2)*read_grid.deta_dx
    zy_org = (ixy==1)*read_grid.dxi_dy + (ixy==2)*read_grid.deta_dy
    grad_norm = jnp.sqrt(zx_org**2+zy_org**2)
    zx = zx_org/grad_norm
    zy = zy_org/grad_norm
    J = read_grid.J
    gamma = thermo.gamma
    rho,u,v,p,a = aux_func.U_to_prim(U)
    rhoE = U[3:4,:,:]
    theta = zx*u + zy*v
    H1 = J/(2*gamma)*jnp.concatenate([rho,rho*u-rho*a*zx,rho*v-rho*a*zy,rhoE+p-rho*a*theta],axis=0)
    H2 = J*(gamma-1)/gamma*jnp.concatenate([rho,rho*u,rho*v,0.5*rho*(u**2+v**2)],axis=0)
    H4 = J/(2*gamma)*jnp.concatenate([rho,rho*u+rho*a*zx,rho*v+rho*a*zy,rhoE+p+rho*a*theta],axis=0)
    eps = 1e-6
    lambda1 = zx_org*u+zy_org*v-a*grad_norm
    lambda1p = 0.5*(lambda1+jnp.sqrt(lambda1**2+eps**2))
    lambda1m = 0.5*(lambda1-jnp.sqrt(lambda1**2+eps**2))
    lambda2 = zx_org*u+zy_org*v
    lambda2p = 0.5*(lambda2+jnp.sqrt(lambda2**2+eps**2))
    lambda2m = 0.5*(lambda2-jnp.sqrt(lambda2**2+eps**2))
    lambda4 = zx_org*u+zy_org*v+a*grad_norm
    lambda4p = 0.5*(lambda4+jnp.sqrt(lambda4**2+eps**2))
    lambda4m = 0.5*(lambda4-jnp.sqrt(lambda4**2+eps**2))
    Hplus = lambda1p*H1 + lambda2p*H2 + lambda4p*H4
    Hminus = lambda1m*H1 + lambda2m*H2 + lambda4m*H4
    return Hplus,Hminus


@jit
def WENO_plus_x(f):
    fj = f[:,2:-3,3:-3]
    fjp1 = f[:,3:-2,3:-3]
    fjp2 = f[:,4:-1,3:-3]
    fjm1 = f[:,1:-4,3:-3]
    fjm2 = f[:,0:-5,3:-3]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    dfj = fj_halfp[:,1:,:] - fj_halfp[:,0:-1,:]
    return dfj

@jit
def WENO_plus_y(f):

    fj = f[:,3:-3,2:-3]
    fjp1 = f[:,3:-3,3:-2]
    fjp2 = f[:,3:-3,4:-1]
    fjm1 = f[:,3:-3,1:-4]
    fjm2 = f[:,3:-3,0:-5]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    dfj = fj_halfp[:,:,1:] - fj_halfp[:,:,0:-1]

    return dfj

@jit
def WENO_minus_x(f):

    fj = f[:,3:-2,3:-3]
    fjp1 = f[:,4:-1,3:-3]
    fjp2 = f[:,5:,3:-3]
    fjm1 = f[:,2:-3,3:-3]
    fjm2 = f[:,1:-4,3:-3]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    dfj = (fj_halfm[:,1:,:] - fj_halfm[:,0:-1,:])

    return dfj

@jit
def WENO_minus_y(f):

    fj = f[:,3:-3,3:-2]
    fjp1 = f[:,3:-3,4:-1]
    fjp2 = f[:,3:-3,5:]
    fjm1 = f[:,3:-3,2:-3]
    fjm2 = f[:,3:-3,1:-4]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    dfj = (fj_halfm[:,:,1:] - fj_halfm[:,:,0:-1])

    return dfj

@jit
def weno5(U):
    Fplus, Fminus = splitFlux_LF(1, U)
    Gplus, Gminus = splitFlux_LF(2, U)

    dFp = WENO_plus_x(Fplus)
    dFm = WENO_minus_x(Fminus)

    dGp = WENO_plus_y(Gplus)
    dGm = WENO_minus_y(Gminus)

    dF = dFp + dFm
    dG = dGp + dGm

    netflux = dF/(read_grid.dxi) + dG/(read_grid.deta)

    return (-netflux)/read_grid.J[:,3:-3,3:-3]
