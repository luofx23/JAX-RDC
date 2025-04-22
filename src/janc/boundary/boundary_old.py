import jax
import jax.numpy as jnp

usr_boundary_func = None

def set_boundary(boundary:dict):
    global usr_boundary_func
    assert (('boundary_conditions' in boundary) and (boundary['boundary_conditions'] is not None)),"funtions on boundary conditions must be provided."
    usr_boundary_func = boundary['boundary_conditions']
    

#user-defined-functions#
def boundary_conditions(U,aux):
    field = jnp.concatenate([U,aux],axis=0)
    field_periodic_x = jnp.concatenate([field[:,-4:-3,:],field[:,-3:-2,:],field[:,-2:-1,:],field,field[:,1:2,:],field[:,2:3,:],field[:,3:4,:]],axis=1)
    field_periodic_pad = jnp.concatenate([field_periodic_x[:,:,-4:-3],field_periodic_x[:,:,-3:-2],field_periodic_x[:,:,-2:-1],field_periodic_x,field_periodic_x[:,:,1:2],field_periodic_x[:,:,2:3],field_periodic_x[:,:,3:4]],axis=2)
    U_periodic_pad,aux_periodic_pad = field_periodic_pad[0:-2],field_periodic_pad[-2:]
    U_with_lb,aux_with_lb = usr_boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad)
    U_with_rb,aux_with_rb = usr_boundary_func['right_boundary'](U_with_lb,aux_with_lb)
    U_with_bb,aux_with_bb = usr_boundary_func['bottom_boundary'](U_with_rb,aux_with_rb)
    U_with_ghost_cell,aux_with_ghost_cell = usr_boundary_func['up_boundary'](U_with_bb,aux_with_bb)
  
    return U_with_ghost_cell,aux_with_ghost_cell

##parallel settings##
num_devices = jax.local_device_count()
devices = jax.devices()


def exchange_halo(device_grid):
    _, grid_nx, _ = device_grid.shape
    halo_size = 3

    send_right = device_grid[:,-halo_size:,:] #向右发送的数据
    recv_left = jax.lax.ppermute(send_right,'x',[(i,(i+1)%num_devices) for i in range(num_devices)])

    send_left = device_grid[:,:halo_size,:] #向左发送的数据
    recv_right = jax.lax.ppermute(send_left,'x',[(i,(i-1)%num_devices) for i in range(num_devices)])

    new_grid = jnp.concatenate([recv_left,device_grid,recv_right],axis=1)
    return new_grid


def parallel_boundary_conditions(U,aux):
    device_idx = jax.lax.axis_index('x')
    field = jnp.concatenate([U,aux],axis=0)
    field_periodic_x = exchange_halo(field)
    field_periodic_pad = jnp.concatenate([field_periodic_x[:,:,-4:-3],field_periodic_x[:,:,-3:-2],field_periodic_x[:,:,-2:-1],field_periodic_x,field_periodic_x[:,:,1:2],field_periodic_x[:,:,2:3],field_periodic_x[:,:,3:4]],axis=2)
    U_periodic_pad,aux_periodic_pad = field_periodic_pad[0:-2],field_periodic_pad[-2:]
    U_with_lb,aux_with_lb = jax.lax.cond(device_idx==0,lambda:usr_boundary_func['left_boundary'](U_periodic_pad),lambda:(U_periodic_pad,aux_periodic_pad))
    U_with_rb,aux_with_rb = jax.lax.cond(device_idx==(num_devices-1),lambda:usr_boundary_func['right_boundary'](U_with_lb,aux_with_lb),lambda:(U_with_lb,aux_with_lb))
    U_with_bb,aux_with_bb = usr_boundary_func['bottom_boundary'](U_with_rb,aux_with_rb)
    U_with_ghost_cell,aux_with_ghost_cell = usr_boundary_func['up_boundary'](U_with_bb,aux_with_bb)
    return U_with_ghost_cell,aux_with_ghost_cell


def split_and_distribute_data(grid):
    grid_list = jnp.array(jnp.split(grid,num_devices,axis=1))
    return grid_list