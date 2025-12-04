import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax
# import external libraries

from PyPIC3D.utils import digital_filter, wrap_around

@jit
def J_from_rhov(particles, J, constants, world, grid):
    """
    Compute the current density from the charge density and particle velocities.

    Args:
        particles (list): List of particle species, each with methods to get charge, subcell position, resolution, and index.
        rho (ndarray): Charge density array.
        J (tuple): Current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
        constants (dict): Dictionary containing physical constants.

    Returns:
        tuple: Updated current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
    """

    C = constants['C']
    # speed of light

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    Nx = world['Nx']
    Ny = world['Ny']
    Nz = world['Nz']
    # get the world parameters

    Jx, Jy, Jz = J
    # unpack the values of J
    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)
    # initialize the current arrays as 0
    J = (Jx, Jy, Jz)
    # initialize the current density as a tuple

    if particles:
        # if there are particles in the simulation

        total_x = jnp.concatenate( [species.get_position()[0] for species in particles] )
        total_y = jnp.concatenate( [species.get_position()[1] for species in particles] )
        total_z = jnp.concatenate( [species.get_position()[2] for species in particles] )

        total_dqvx = jnp.concatenate( [species.get_charge() / (dx*dy*dz) * species.get_velocity()[0] for species in particles] )
        total_dqvy = jnp.concatenate( [species.get_charge() / (dx*dy*dz) * species.get_velocity()[1] for species in particles] )
        total_dqvz = jnp.concatenate( [species.get_charge() / (dx*dy*dz) * species.get_velocity()[2] for species in particles] )
        # concatenate all the particle data for easier processing

        shape_factor = particles[0].get_shape()
        # assume all species have the same shape factor

        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (total_x - grid[0][0]) / dx).astype(int),
            lambda _: jnp.round( (total_x - grid[0][0]) / dx).astype(int),
            operand=None
        )

        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (total_y - grid[1][0]) / dy).astype(int),
            lambda _: jnp.round( (total_y - grid[1][0]) / dy).astype(int),
            operand=None
        )

        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (total_z - grid[2][0]) / dz).astype(int),
            lambda _: jnp.round( (total_z - grid[2][0]) / dz).astype(int),
            operand=None
        )
        # calculate the nearest grid point based on shape factor

        x1 = wrap_around(x0+1, Nx)
        y1 = wrap_around(y0+1, Ny)
        z1 = wrap_around(z0+1, Nz)
        # calculate the right grid point

        x_minus1 = x0 - 1
        y_minus1 = y0 - 1
        z_minus1 = z0 - 1
        # calculate the left grid point

        xpts = [x_minus1, x0, x1]
        ypts = [y_minus1, y0, y1]
        zpts = [z_minus1, z0, z1]
        # place all the points in a list

        deltax = (total_x - grid[0][0]) - (x0 * dx)
        deltay = (total_y - grid[1][0]) - (y0 * dy)
        deltaz = (total_z - grid[2][0]) - (z0 * dz)
        # Calculate the difference between the particle position and the nearest grid point

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights( deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )

        for i in range(len(x_weights)):
            for j in range(len(y_weights)):
                for k in range(len(z_weights)):
                    Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add( (total_dqvx) * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
                    Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add( (total_dqvy) * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
                    Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add( (total_dqvz) * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # Add the particle current to the current density arrays

        
        alpha = constants['alpha']
        Jx = digital_filter(Jx, alpha)
        Jy = digital_filter(Jy, alpha)
        Jz = digital_filter(Jz, alpha)
        J = (Jx, Jy, Jz)
        # apply a digital filter to the current density arrays

    return J



def Esirkepov_current(particles, J, constants, world, grid):

    Jx, Jy, Jz = J
    # unpack the values of J
    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)
    # initialize the current arrays as 0

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get the grid spacing parameters
    dt = world['dt']
    # get the time step
    C = constants['C']
    # speed of light
    Nx, Ny, Nz = Jx.shape
    # get the shape of the charge density array

    x_grid, y_grid, z_grid = grid
    # unpack the grid values

    n_dims = 3 - (Nx == 1) - (Ny == 1) - (Nz == 1)
    # determine the number of dimensions in the simulation

    x_dim = (Nx != 1)
    y_dim = (Ny != 1)
    z_dim = (Nz != 1)
    # determine which dimensions are active

    global_Wx = jnp.zeros_like(Jx)
    global_Wy = jnp.zeros_like(Jy)
    global_Wz = jnp.zeros_like(Jz)
    # initialize global weight arrays


    for species in particles:
        q = species.get_charge()
        # get the charge of the species

        old_x, old_y, old_z = species.get_old_position()
        # get the old position of the particles in the species
        x, y, z = species.get_position()
        # get the position of the particles in the species
        vx, vy, vz = species.get_velocity()
        # get the velocity of the particles in the species

        shape_factor = species.get_shape()
        # get the shape factor of the species

        N_particles = species.get_number_of_particles()
        # get the total number of particles in the species


        dJx = jax.lax.cond(
            Nx == 1,
            lambda _: q * vx / (dx*dy*dz),
            lambda _: -q / (dy*dz) / dt * jnp.ones(N_particles),
            operand=None
        )

        dJy = jax.lax.cond(
            Ny == 1,
            lambda _: q * vy / (dx*dy*dz),
            lambda _: -q / (dx*dz) / dt * jnp.ones(N_particles),
            operand=None
        )

        dJz = jax.lax.cond(
            Nz == 1,
            lambda _: q * vz / (dx*dy*dz),
            lambda _: -q / (dx*dy) / dt * jnp.ones(N_particles),
            operand=None
        )
        # calculate the current differential

        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (x - grid[0][0]) / dx).astype(int),
            lambda _: jnp.round( (x - grid[0][0]) / dx).astype(int),
            operand=None
        )

        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (y - grid[1][0]) / dy).astype(int),
            lambda _: jnp.round( (y - grid[1][0]) / dy).astype(int),
            operand=None
        )

        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (z - grid[2][0]) / dz).astype(int),
            lambda _: jnp.round( (z - grid[2][0]) / dz).astype(int),
            operand=None
        )
        # calculate the nearest grid point based on shape factor

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)
        # wrap around the grid indices

        x1 = wrap_around(x0+1, Nx)
        y1 = wrap_around(y0+1, Ny)
        z1 = wrap_around(z0+1, Nz)
        # calculate the right grid point

        x2 = wrap_around(x0+2, Nx)
        y2 = wrap_around(y0+2, Ny)
        z2 = wrap_around(z0+2, Nz)
        # calculate the next right grid point

        x_minus1 = wrap_around(x0-1, Nx)
        y_minus1 = wrap_around(y0-1, Ny)
        z_minus1 = wrap_around(z0-1, Nz)
        # calculate the left grid point

        x_minus2 = wrap_around(x0-2, Nx)
        y_minus2 = wrap_around(y0-2, Ny)
        z_minus2 = wrap_around(z0-2, Nz)
        # calculate the next left grid point

        xpts = [x_minus2, x_minus1, x0, x1, x2]
        ypts = [y_minus2, y_minus1, y0, y1, y2]
        zpts = [z_minus2, z_minus1, z0, z1, z2]
        # place all the points in a list

        deltax = (x - grid[0][0]) - (x0 * dx)
        deltay = (y - grid[1][0]) - (y0 * dy)
        deltaz = (z - grid[2][0]) - (z0 * dz)
        # Calculate the difference between the particle position and the nearest grid point

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )
        # Calculate the weights for the grid points

        old_x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_x - grid[0][0]) / dx).astype(int),
            lambda _: jnp.round( (old_x - grid[0][0]) / dx).astype(int),
            operand=None
        )

        old_y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_y - grid[1][0]) / dy).astype(int),
            lambda _: jnp.round( (old_y - grid[1][0]) / dy).astype(int),
            operand=None
        )

        old_z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_z - grid[2][0]) / dz).astype(int),
            lambda _: jnp.round( (old_z - grid[2][0]) / dz).astype(int),
            operand=None
        )
        # calculate the nearest grid point based on shape factor

        old_x0 = wrap_around(old_x0, Nx)
        old_y0 = wrap_around(old_y0, Ny)
        old_z0 = wrap_around(old_z0, Nz)
        # wrap around the grid indices

        old_deltax = (old_x - grid[0][0]) - (old_x0 * dx)
        old_deltay = (old_y - grid[1][0]) - (old_y0 * dy)
        old_deltaz = (old_z - grid[2][0]) - (old_z0 * dz)
        # Calculate the difference between the old particle position and the grid point nearest the new x

        old_x_weights, old_y_weights, old_z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            operand=None
        )

        x_pad = jnp.zeros_like( x_weights[0] )
        y_pad = jnp.zeros_like( y_weights[0] )
        z_pad = jnp.zeros_like( z_weights[0] )
        # create padding arrays

        x_weights = [x_pad, x_weights[0], x_weights[1], x_weights[2], x_pad]
        y_weights = [y_pad, y_weights[0], y_weights[1], y_weights[2], y_pad]
        z_weights = [z_pad, z_weights[0], z_weights[1], z_weights[2], z_pad]
        old_x_weights = [x_pad, old_x_weights[0], old_x_weights[1], old_x_weights[2], x_pad]
        old_y_weights = [y_pad, old_y_weights[0], old_y_weights[1], old_y_weights[2], y_pad]
        old_z_weights = [z_pad, old_z_weights[0], old_z_weights[1], old_z_weights[2], z_pad]
        # pad the weights to account for shifts


        shift_xi = (old_x0 - x0).astype(int)
        shift_yi = (old_y0 - y0).astype(int)
        shift_zi = (old_z0 - z0).astype(int)
        # calculate the shifts in indices (int)

        def single_particle_roll(old_weights, shift_i):
            return jnp.roll(old_weights, -shift_i, axis=0)
        
        roll_array = jax.vmap(single_particle_roll, in_axes=(1, 0), out_axes=1)

        old_x_weights = roll_array(jnp.array(old_x_weights), shift_xi)
        old_y_weights = roll_array(jnp.array(old_y_weights), shift_yi)
        old_z_weights = roll_array(jnp.array(old_z_weights), shift_zi)
        # vectorized per-particle roll using jax.vmap

        old_x_weights = [old_x_weights[i, ...] for i in range(5)]
        old_y_weights = [old_y_weights[i, ...] for i in range(5)]
        old_z_weights = [old_z_weights[i, ...] for i in range(5)]
        # convert back to lists

        Wx_, Wy_, Wz_ =  get_3D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=None)

        for k in range(len(zpts)):
            for j in range(len(ypts)):
                for i in range(len(xpts)):
                    factor = dJx * Wx_[i, j, k, :]
                    global_Wx = global_Wx.at[xpts[i],ypts[j],zpts[k]].add( factor, mode='drop')
                
                    factor = dJy * Wy_[i, j, k, :]
                    global_Wy = global_Wy.at[xpts[i],ypts[j],zpts[k]].add( factor, mode='drop')

                    factor = dJz * Wz_[i, j, k, :]
                    global_Wz = global_Wz.at[xpts[i],ypts[j],zpts[k]].add( factor, mode='drop')
                    # Add the local weights to the global weight arrays


    for i in range(Nx):
        Jx = Jx.at[i, :, :].add( global_Wx[i, :, :] + Jx.at[i-1, :, :].get(), mode='drop')
    for j in range(Ny):
        Jy = Jy.at[:, j, :].add( global_Wy[:, j, :] + Jy.at[:, j-1, :].get(), mode='drop')
    for k in range(Nz):
        Jz = Jz.at[:, :, k].add( global_Wz[:, :, k] + Jz.at[:, :, k-1].get(), mode='drop')
    # accumulate the global weights into the current density arrays


    J = (Jx, Jy, Jz)

    return J
            
        
        
def get_3D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=None):

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros( (len(y_weights),len(x_weights),len(z_weights), N_particles) )
    Wz_ = jnp.zeros( (len(z_weights),len(x_weights),len(y_weights), N_particles) )


    for i in range(len(x_weights)):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i,j,k,:].set( (x_weights[i] - old_x_weights[i]) * ( 1/3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])     \
                                                    +  1/6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k]) ) )

                Wy_ = Wy_.at[i,j,k,:].set( (y_weights[j] - old_y_weights[j]) * ( 1/3 * (x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k])     \
                                                    +  1/6 * (x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k]) ) )

                Wz_ = Wz_.at[i,j,k,:].set( (z_weights[k] - old_z_weights[k]) * ( 1/3 * (x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j])     \
                                                    +  1/6 * (x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j]) ) )

    return Wx_, Wy_, Wz_

# def get_2D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=2):
#     d_Sx = []
#     d_Sy = []
#     d_Sz = []

#     for i in range(len(x_weights)):
#         d_Sx.append(x_weights[i] - old_x_weights[i])
#         d_Sy.append(y_weights[i] - old_y_weights[i])
#         d_Sz.append(z_weights[i] - old_z_weights[i])
#     # Calculate the difference in weights for the central grid points

#     Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
#     Wy_ = jnp.zeros( (len(y_weights),len(x_weights),len(z_weights), N_particles) )
#     Wz_ = jnp.zeros( (len(z_weights),len(x_weights),len(y_weights), N_particles) )


#     for i in range(len(x_weights)):
#         for j in range(len(y_weights)):
#             for k in range(len(z_weights)):

#                 factor = lax.cond(
#                     null_dim == 0,
#                     # if the 2D plane is in the yz plane
#                     lambda _: 1/3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])     \
#                         +  1/6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k]),

                    
#                     # if the 2D plane is in the xz or xy plane
#                     lambda _: 1/2 * d_Sx[i] * lax.cond(
#                                             null_dim == 1,
#                                             # if the 2D plane is in the xz plane
#                                             lambda _: z_weights[k] + old_z_weights[k],
#                                             # if the 2D plane is in the xy plane
#                                             lambda _: y_weights[j] + old_y_weights[j],
#                                         operand=None
#                                         ),
    
#                     operand=None
#                 )
#                 Wx_ = Wx_.at[i,j,k,:].set( factor )


#                 factor = lax.cond(
#                     null_dim == 1,
#                     # if the 2D plane is in the xz plane
#                     lambda _: 1/3 * (x_weights[j] * z_weights[k] + old_x_weights[j] * old_z_weights[k])     \
#                         +  1/6 * (x_weights[j] * old_z_weights[k] + old_x_weights[j] * z_weights[k]),

#                     # if the 2D plane is in the yz or xy plane
#                     lambda _: 1/2 * d_Sy[i] * lax.cond(
#                                             null_dim == 0,
#                                             # if the 2D plane is in the yz plane
#                                             lambda _: z_weights[k] + old_z_weights[k],
#                                             # if the 2D plane is in the xy plane
#                                             lambda _: x_weights[j] + old_x_weights[j],
#                                         operand=None
#                                         ),
    
#                     operand=None
#                 )
                
#                 Wy_ = Wy_.at[j,i,k,:].set( factor )
                

#                 factor = lax.cond(
#                     null_dim == 2,
#                     # if the 2D plane is in the xy plane
#                     lambda _: 1/3 * (x_weights[j] * y_weights[k] + old_x_weights[j] * old_y_weights[k])     \
#                         +  1/6 * (x_weights[j] * old_y_weights[k] + old_x_weights[j] * y_weights[k]),

#                     # if the 2D plane is in the yz or xz plane
#                     lambda _: 1/2 * d_Sz[i] * lax.cond(
#                                             null_dim == 0,
#                                             # if the 2D plane is in the yz plane
#                                             lambda _: y_weights[k] + old_y_weights[k],
#                                             # if the 2D plane is in the xz plane
#                                             lambda _: x_weights[j] + old_x_weights[j],
#                                         operand=None
#                                         ),
    
#                     operand=None
#                 )

#                 Wz_ = Wz_.at[k,i,j,:].set( factor )

#     return Wx_, Wy_, Wz_


# def get_1D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, dim=0):

#     Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
#     Wy_ = jnp.zeros( (len(y_weights),len(x_weights),len(z_weights), N_particles) )
#     Wz_ = jnp.zeros( (len(z_weights),len(x_weights),len(y_weights), N_particles) )


#     for i in range(len(x_weights)):
#         for j in range(len(y_weights)):
#             for k in range(len(z_weights)):

#                 factor = lax.cond(
#                     dim == 0,
#                     # if the 1D line is in the x direction
#                     lambda _: x_weights[i] - old_x_weights[i],

#                     # if the 1D line is in the y or z direction
#                     lambda _: lax.cond(
#                                     dim == 1,
#                                     # if the 1D line is in the y direction
#                                     lambda _: 1/2 * (y_weights[j] + old_y_weights[j]),
#                                     # if the 1D line is in the z direction
#                                     lambda _: 1/2 * (z_weights[k] + old_z_weights[k]),
#                                     operand=None
#                                         ),
    
#                     operand=None
#                 )
#                 Wx_ = Wx_.at[i,j,k,:].set( factor )


                
#                 factor = lax.cond(
#                     dim == 1,
#                     # if the 1D line is in the y direction
#                     lambda _: y_weights[i] - old_y_weights[i],

#                     # if the 1D line is in the x or z direction
#                     lambda _: lax.cond(
#                                     dim == 0,
#                                     # if the 1D line is in the x direction
#                                     lambda _: 1/2 * (x_weights[j] + old_x_weights[j]),
#                                     # if the 1D line is in the z direction
#                                     lambda _: 1/2 * (z_weights[k] + old_z_weights[k]),
#                                     operand=None
#                                         ),
    
#                     operand=None
#                 )

#                 Wy_ = Wy_.at[j,i,k,:].set( factor )

#                 factor = lax.cond(
#                     dim == 2,
#                     # if the 1D line is in the z direction
#                     lambda _: z_weights[i] - old_z_weights[i],

#                     # if the 1D line is in the x or y direction
#                     lambda _: lax.cond(
#                                     dim == 0,
#                                     # if the 1D line is in the x direction
#                                     lambda _: 1/2 * (x_weights[j] + old_x_weights[j]),
#                                     # if the 1D line is in the y direction
#                                     lambda _: 1/2 * (y_weights[k] + old_y_weights[k]),
#                                     operand=None
#                                         ),
    
#                     operand=None
#                 )

#                 Wz_ = Wz_.at[k,i,j,:].set( factor )

#     return Wx_, Wy_, Wz_

@jit
def get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the second-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = (3/4) - (deltax/dx)**2
    Sy0 = (3/4) - (deltay/dy)**2
    Sz0 = (3/4) - (deltaz/dz)**2

    Sx1 = (1/2) * ((1/2) - (deltax/dx))**2
    Sy1 = (1/2) * ((1/2) - (deltay/dy))**2
    Sz1 = (1/2) * ((1/2) - (deltaz/dz))**2

    Sx_minus1 = (1/2) * ((1/2) + (deltax/dx))**2
    Sy_minus1 = (1/2) * ((1/2) + (deltay/dy))**2
    Sz_minus1 = (1/2) * ((1/2) + (deltaz/dz))**2
    # second order weights

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights

@jit
def get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the first-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = jnp.asarray(1 - deltax / dx)
    Sy0 = jnp.asarray(1 - deltay / dy)
    Sz0 = jnp.asarray(1 - deltaz / dz)

    Sx1 = jnp.asarray(deltax / dx)
    Sy1 = jnp.asarray(deltay / dy)
    Sz1 = jnp.asarray(deltaz / dz)

    Sx_minus1 = jnp.zeros_like(Sx0)
    Sy_minus1 = jnp.zeros_like(Sy0)
    Sz_minus1 = jnp.zeros_like(Sz0)
    # No second-order weights for first-order weighting

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights