import jax.numpy as jnp
from jax import lax

#user_defined imported library#
from thermo import get_thermo,get_R
#finish import#


boundaryParams = {}

def read_boundary_params(boundary:dict):
    global boundaryParams
    boundaryParams = boundary
    #return boundaryParams

#user-defined-functions#
def boundary_conditions(U,aux):
    Pb = boundaryParams['Pb']
    Yinj_cor = boundaryParams['Yinj']
    na = Yinj_cor.shape[0]

    U_periodic = jnp.concatenate([U[:,-4:-3,:],U[:,-3:-2,:],U[:,-2:-1,:],U,U[:,1:2,:],U[:,2:3,:],U[:,3:4,:]],axis=1)
    aux_periodic = jnp.concatenate([aux[:,-4:-3,:],aux[:,-3:-2,:],aux[:,-2:-1,:],aux,aux[:,1:2,:],aux[:,2:3,:],aux[:,3:4,:]],axis=1)
    state_periodic = U_periodic
    gamma_periodic = aux_periodic[0:1,:,:]
    T_periodic = aux_periodic[1:2,:,:]
 
    state_out = state_periodic[:,:,-1:]
    gamma_out = gamma_periodic[:,:,-1:]
    T_out = T_periodic[:,:,-1:]
    rho_out = state_out[0:1,:,:]
    u_out = state_out[1:2,:,:]/rho_out
    v_out = state_out[2:3,:,:]/rho_out
    Y_out = state_out[4:,:,:]/rho_out
    R_out = get_R(Y_out)
    p_out = rho_out*(R_out*T_out)
    a_out = jnp.sqrt(gamma_out*p_out/rho_out)
    mask = (v_out/a_out < 1.0)
    rho_cor_out = lax.select(mask, Pb / (p_out / rho_out),rho_out)
    p_cor_out = lax.select(mask, Pb*jnp.ones_like(p_out),p_out)
    T_cor_out = lax.select(mask, p_cor_out/(rho_cor_out*R_out),T_out)
    _, gamma_out, h_out, _,_ = get_thermo(T_cor_out,Y_out)
    upper_bound_state = jnp.concatenate([rho_cor_out, rho_cor_out * u_out, rho_cor_out * v_out,
                      rho_cor_out*h_out - p_cor_out + 0.5 * rho_cor_out * (u_out ** 2 + v_out ** 2),
                      rho_cor_out * Y_out], axis=0)
    aux_up = jnp.concatenate([gamma_out,T_cor_out],axis=0)

    state_in = state_periodic[:,:,0:1]
    gamma_in = gamma_periodic[:,:,0:1]##
    T_in = T_periodic[:,:,0:1]
    rho_in = state_in[0:1,:,:]
    u_in = state_in[1:2,:,:]/rho_in
    v_in = state_in[2:3,:,:]/rho_in
    Y_in = state_in[4:,:,:]/rho_in
    _, _, h_in, R_in,_ = get_thermo(T_in,Y_in)
    p_in = rho_in*R_in*T_in

    u_temp = jnp.zeros_like(u_in)
    Y_temp = Yinj_cor
    v_temp, T_temp, h_temp, gamma_temp = inj_model(p_in)
    R_temp = get_R(Y_temp)
    rho_temp = p_in/(R_temp*T_temp)

    mask_in = (p_in >= 1.0)
    rho_cor_in = lax.select(mask_in,rho_in,rho_temp)
    u_cor_in = lax.select(mask_in,u_in,u_temp)
    v_cor_in = lax.select(mask_in,-v_in,v_temp)
    T_cor_in = lax.select(mask_in,T_in,T_temp)
    p_cor_in = p_in
    h_cor_in = lax.select(mask_in,h_in,h_temp)
    Y_cor_in = lax.select(jnp.tile(mask_in,(na,1,1)),Y_in,Y_temp)
    gamma_cor_in = lax.select(mask_in,gamma_in,gamma_temp)

    lower_bound_state = jnp.concatenate([rho_cor_in, rho_cor_in * u_cor_in, rho_cor_in * v_cor_in,
                     rho_cor_in*h_cor_in - p_cor_in + 0.5 * rho_cor_in * (u_cor_in ** 2 + v_cor_in ** 2),
                     rho_cor_in * Y_cor_in], axis=0)
    aux_low = jnp.concatenate([gamma_cor_in,T_cor_in],axis=0)
    U_with_ghost_cell = jnp.concatenate([lower_bound_state,lower_bound_state,lower_bound_state,U_periodic,
                        upper_bound_state,upper_bound_state,upper_bound_state],axis=2)
    aux_with_ghost_cell = jnp.concatenate([aux_low,aux_low,aux_low,aux_periodic,
                        aux_up,aux_up,aux_up],axis=2)

    return U_with_ghost_cell.astype(jnp.float32),aux_with_ghost_cell.astype(jnp.float32)


##auxilitary function defined by users## 
def inj_model(p):
    Yinj_cor = boundaryParams['Yinj']
    A1 = 1
    A3 = 5
    A2 = A3-A1
    R = get_R(Yinj_cor)
    gamma = 1.29
    C0 = jnp.sqrt(gamma*R*1.0)

    M = jnp.zeros_like(p)
    P1 = 1.0*(1+(gamma-1)/2*M**2)**(-gamma/(gamma-1))
    V1 = M*(1+(gamma-1)/2*M**2)**(-0.5)*C0
    MFC = A1*1.0/jnp.sqrt(1.0)*jnp.sqrt(gamma/R)*M*(1+(gamma-1)/2*M**2)**(-(gamma+1)/2/(gamma-1))
    A = 0.5
    P3 = p
    B = gamma/(gamma-1)*P3*A3/MFC
    C = -gamma/(gamma-1)*R*1.0
    V3 = (-B+jnp.sqrt(B**2-4*A*C))/(2*A)
    P2 = (MFC*(V3-V1)-P1*A1+P3*A3)/A2

    M1 = jnp.zeros_like(p)
    M2 = jnp.ones_like(p)
    p_cor = p

    for i in range(20):
        M = 0.5*(M1+M2)
        P1 = 1.0*(1+(gamma-1)/2*M**2)**(-gamma/(gamma-1))
        V1 = M*(1+(gamma-1)/2*M**2)**(-0.5)*C0
        MFC = A1*1.0/jnp.sqrt(1.0)*jnp.sqrt(gamma/R)*M*(1+(gamma-1)/2*M**2)**(-(gamma+1)/2/(gamma-1))
        A = 0.5
        B = gamma/(gamma-1)*P3*A3/MFC
        C = -gamma/(gamma-1)*R*1.0
        V3 = (-B+jnp.sqrt(B**2-4*A*C))/(2*A)
        P2 = (MFC*(V3-V1)-P1*A1+P3*A3)/A2

        M2 = lax.select(P2>=P1,M,M2)
        M1 = lax.select(P2<P1,M,M1)

    rho_cor = MFC/V3/A3
    v_cor = V3
    T_cor = p_cor/(R*rho_cor)
    _, gamma, h_cor, _, _ = get_thermo(T_cor,Yinj_cor)


    return v_cor, T_cor, h_cor, gamma