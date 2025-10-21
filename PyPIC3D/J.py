import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax
# import external libraries

from PyPIC3D.utils import digital_filter

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

@jit
def wrap_around(ix, size):
    """Wrap around index (scalar or 1D array) to ensure it is within bounds."""
    return jnp.where(ix > size - 1, ix - size, ix)


# def Esirkepov_current(particles, J, constants, world, grid):

#     Jx, Jy, Jz = J
#     # unpack the values of J
#     Jx = Jx.at[:, :, :].set(0)
#     Jy = Jy.at[:, :, :].set(0)
#     Jz = Jz.at[:, :, :].set(0)
#     # initialize the current arrays as 0

#     dx = world['dx']
#     dy = world['dy']
#     dz = world['dz']
#     # get the grid spacing parameters
#     dt = world['dt']
#     # get the time step
#     Nx, Ny, Nz = Jx.shape
#     # get the shape of the charge density array


#     for species in particles:
#         q = species.get_charge()
#         # get the charge of the species

#         old_x, old_y, old_z = species.get_old_position()
#         # get the old position of the particles in the species
#         x, y, z = species.get_position()
#         # get the position of the particles in the species
#         vx, vy, vz = species.get_velocity()
#         # get the velocity of the particles in the species

#         shape_factor = species.get_shape()
#         # get the shape factor of the species

#         N_particles = species.get_number_of_particles()
#         # get the total number of particles in the species


#         dJx = jax.lax.cond(
#             Nx == 1,
#             lambda _: q * vx / (dx*dy*dz),
#             lambda _: q / (dy*dz) / dt * jnp.ones(N_particles),
#             operand=None
#         )

#         dJy = jax.lax.cond(
#             Ny == 1,
#             lambda _: q * vy / (dx*dy*dz),
#             lambda _: q / (dx*dz) / dt * jnp.ones(N_particles),
#             operand=None
#         )

#         dJz = jax.lax.cond(
#             Nz == 1,
#             lambda _: q * vz / (dx*dy*dz),
#             lambda _: q / (dx*dy) / dt * jnp.ones(N_particles),
#             operand=None
#         )
#         # calculate the current differential

#         x0 = jnp.floor( (x - grid[0][0]) / dx).astype(int)
#         y0 = jnp.floor( (y - grid[1][0]) / dy).astype(int)
#         z0 = jnp.floor( (z - grid[2][0]) / dz).astype(int)
#         # calculate the nearest grid point

#         x1 = wrap_around(x0+1, Nx)
#         y1 = wrap_around(y0+1, Ny)
#         z1 = wrap_around(z0+1, Nz)
#         # calculate the right grid point

#         x_minus1 = x0 - 1
#         y_minus1 = y0 - 1
#         z_minus1 = z0 - 1
#         # calculate the left grid point

#         xpts = [x_minus1, x0, x1]
#         ypts = [y_minus1, y0, y1]
#         zpts = [z_minus1, z0, z1]
#         # place all the points in a list

#         old_deltax = old_x - jnp.floor(old_x / dx) * dx
#         old_deltay = old_y - jnp.floor(old_y / dy) * dy
#         old_deltaz = old_z - jnp.floor(old_z / dz) * dz
#         # Calculate the difference between the particle position and the nearest grid point
#         new_deltax = x - jnp.floor(x / dx) * dx
#         new_deltay = y - jnp.floor(y / dy) * dy
#         new_deltaz = z - jnp.floor(z / dz) * dz
#         # Calculate the difference between the particle position and the nearest grid point

#         old_x_weights, old_y_weights, old_z_weights = jax.lax.cond(
#             shape_factor == 1,
#             lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
#             lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
#             operand=None
#         )
#         x_weights, y_weights, z_weights = jax.lax.cond(
#             shape_factor == 1,
#             lambda _: get_first_order_weights(new_deltax, new_deltay, new_deltaz, dx, dy, dz),
#             lambda _: get_second_order_weights(new_deltax, new_deltay, new_deltaz, dx, dy, dz),
#             operand=None
#         )
#         # Calculate the weights for the grid points

#         d_Sx = []
#         d_Sy = []
#         d_Sz = []

#         for i in range(len(x_weights)):
#             d_Sx.append(x_weights[i] - old_x_weights[i])
#             d_Sy.append(y_weights[i] - old_y_weights[i])
#             d_Sz.append(z_weights[i] - old_z_weights[i])
#         # Calculate the difference in weights for the central grid points

#         Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
#         Wy_ = jnp.zeros( (len(y_weights),len(x_weights),len(z_weights), N_particles) )
#         Wz_ = jnp.zeros( (len(z_weights),len(x_weights),len(y_weights), N_particles) )


#         for i in range(len(x_weights)):
#             for j in range(len(y_weights)):
#                 for k in range(len(z_weights)):
#                     Wx_ = Wx_.at[i, j, k, :].set(   d_Sx[i] * (  (old_y_weights[j] * old_z_weights[k]) + (d_Sy[j] * old_z_weights[k] / 2) \
#                         + (old_y_weights[j] * d_Sz[k] / 2)     +      (d_Sy[j] * d_Sz[k] / 3)      ))

#                     Wy_ = Wy_.at[i, j, k, :].set(   d_Sy[j] * (  (old_x_weights[i] * old_z_weights[k]) + (d_Sx[i] * old_z_weights[k] / 2) \
#                         + (old_x_weights[i] * d_Sz[k] / 2)     +      (d_Sx[i] * d_Sz[k] / 3)      ))

#                     Wz_ = Wz_.at[i, j, k, :].set(   d_Sz[k] * (  (old_x_weights[i] * old_y_weights[j]) + (d_Sx[i] * old_y_weights[j] / 2) \
#                         + (old_x_weights[i] * d_Sy[j] / 2)     +      (d_Sx[i] * d_Sy[j] / 3)      ))

#         for j in range(len(ypts)):
#             for k in range(len(zpts)):
#                 Jx = Jx.at[xpts[0],ypts[j],zpts[k]].add( -Wx_[0, j, k,:] * dJx, mode='drop')
#                 Jx = Jx.at[xpts[1],ypts[j],zpts[k]].add( -(Wx_[0,j,k,:] + Wx_[1,j,k,:]) * dJx, mode='drop')
#                 Jx = Jx.at[xpts[2],ypts[j],zpts[k]].add( -(Wx_[0,j,k,:] + Wx_[1,j,k,:] + Wx_[2,j,k,:]) * dJx, mode='drop')

#         for i in range(len(xpts)):
#             for k in range(len(zpts)):
#                 Jy = Jy.at[xpts[i],ypts[0],zpts[k]].add( -Wy_[i, 0, k,:] * dJy, mode='drop')
#                 Jy = Jy.at[xpts[i],ypts[1],zpts[k]].add( -(Wy_[i, 0, k,:] + Wy_[i, 1, k,:]) * dJy, mode='drop')
#                 Jy = Jy.at[xpts[i],ypts[2],zpts[k]].add( -(Wy_[i, 0, k,:] + Wy_[i, 1, k,:] + Wy_[i, 2, k,:]) * dJy, mode='drop')

#         for i in range(len(xpts)):
#             for j in range(len(ypts)):
#                 Jz = Jz.at[xpts[i],ypts[j],zpts[0]].add( -Wz_[i, j, 0,:] * dJz, mode='drop')
#                 Jz = Jz.at[xpts[i],ypts[j],zpts[1]].add( -(Wz_[i, j, 0,:] + Wz_[i, j, 1,:]) * dJz, mode='drop')
#                 Jz = Jz.at[xpts[i],ypts[j],zpts[2]].add( -(Wz_[i, j, 0,:] + Wz_[i, j, 1,:] + Wz_[i, j, 2,:]) * dJz, mode='drop')

#     alpha = constants['alpha']
#     Jx = digital_filter(Jx, alpha)
#     Jy = digital_filter(Jy, alpha)
#     Jz = digital_filter(Jz, alpha)
#     J = (Jx, Jy, Jz)
#     # apply a digital filter to the current density arrays

#     return J

def Esirkepov_current(particles, J, constants, world, grid):
    """
    Esirkepov_current(particles, J, constants, world, grid)
    Compute and deposit charge-conserving currents for a collection of particle species
    onto a staggered grid using the Esirkepov current deposition scheme.
    This function loops over all species and their particles, computes per-particle
    Esirkepov weights (1st- or 2nd-order shape functions supported), and deposits
    the corresponding current density increments into the provided Jx, Jy, Jz
    arrays. The deposition supports 1D, 2D and 3D simulations by treating any
    dimension with size 1 as absent. After deposition a digital filter is applied
    to each current component.
    Parameters
    ----------
    particles : list of ParticleSpecies instances
    J : tuple
        Tuple of three JAX arrays (Jx, Jy, Jz) representing the current density
        on the grid.
    constants : dict
        Dictionary of physical and algorithmic constants. Required keys:
          - 'C'   : speed of light (used only if needed elsewhere; present for completeness)
          - 'alpha': digital filtering coefficient applied to currents after deposition
    world : dict
        Simulation world / grid spacing and timestep parameters. Required keys:
          - 'dx', 'dy', 'dz' : grid spacings in each dimension (scalars)
          - 'dt'             : simulation timestep (scalar)
    grid : tuple
        Tuple (x_grid, y_grid, z_grid) containing the coordinate arrays (1D)
        describing the grid locations along each axis. The code uses grid[axis][0]
        as a reference origin when mapping particle positions to grid indices.
    Returns
    -------
    tuple
        A tuple (Jx_new, Jy_new, Jz_new) containing the updated and filtered
        current density arrays (same shapes as the input arrays in J). Updates
        are returned as new JAX arrays (no in-place mutation).
    """


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
            lambda _: q / (dy*dz) / dt * jnp.ones(N_particles),
            operand=None
        )

        dJy = jax.lax.cond(
            Ny == 1,
            lambda _: q * vy / (dx*dy*dz),
            lambda _: q / (dx*dz) / dt * jnp.ones(N_particles),
            operand=None
        )

        dJz = jax.lax.cond(
            Nz == 1,
            lambda _: q * vz / (dx*dy*dz),
            lambda _: q / (dx*dy) / dt * jnp.ones(N_particles),
            operand=None
        )
        # calculate the current differential

        x0 = wrap_around( jnp.round( (x - grid[0][0]) / dx).astype(int), Nx)
        y0 = wrap_around( jnp.round( (y - grid[1][0]) / dy).astype(int), Ny)
        z0 = wrap_around( jnp.round( (z - grid[2][0]) / dz).astype(int), Nz)
        # calculate the nearest grid point

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

        n_dims = 3 - (Nx == 1) - (Ny == 1) - (Nz == 1)
        # determine the number of dimensions in the simulation


        deltax = x - jnp.round(x / dx) * dx
        deltay = y - jnp.round(y / dy) * dy
        deltaz = z - jnp.round(z / dz) * dz
        # Calculate the difference between the particle position and the nearest grid point

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )
        # Calculate the weights for the grid points

        old_deltax = old_x - jnp.round(old_x / dx) * dx
        old_deltay = old_y - jnp.round(old_y / dy) * dy
        old_deltaz = old_z - jnp.round(old_z / dz) * dz
        # Calculate the difference between the old particle position and the grid point nearest the new x

        old_x_weights, old_y_weights, old_z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            operand=None
        )

        shift_xi = jnp.round( (old_x - x) / dx).astype(int)
        shift_yi = jnp.round( (old_y - y) / dy).astype(int)
        shift_zi = jnp.round( (old_z - z) / dz).astype(int)

        old_x_weights = jnp.roll(jnp.array(old_x_weights), shift=-shift_xi, axis=0)
        old_y_weights = jnp.roll(jnp.array(old_y_weights), shift=-shift_yi, axis=0)
        old_z_weights = jnp.roll(jnp.array(old_z_weights), shift=-shift_zi, axis=0)
        # Assign old shape weights to shifted positions using JAX indexing
        old_x_weights = [old_x_weights[i, ...] for i in range(5)]
        old_y_weights = [old_y_weights[i, ...] for i in range(5)]
        old_z_weights = [old_z_weights[i, ...] for i in range(5)]
        # convert back to lists


        Wx_, Wy_, Wz_ = jax.lax.cond(
            n_dims == 3,
            lambda _: get_3D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=None),
            lambda _: jax.lax.cond(
                n_dims == 2,
                lambda _: get_2D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=(Nx==1)*0 + (Ny==1)*1 + (Nz==1)*2),
                lambda _: get_1D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, dim=(Nx==1)*0 + (Ny==1)*1 + (Nz==1)*2 ),
                operand=None
            ),
            operand=None
        )
        # Calculate the Esirkepov weights for the current deposition

        for k in range(len(zpts)):
            for j in range(len(ypts)):
                factor = 0
                for i in range(len(xpts)):
                    factor += dJx * Wx_[i, j, k, :]
                    Jx = Jx.at[xpts[i],ypts[j],zpts[k]].add( factor, mode='drop')

        for k in range(len(zpts)):
            for i in range(len(xpts)):
                for j in range(len(ypts)):
                    factor = dJy * Wy_[i, j, k, :]
                    Jy = Jy.at[xpts[i],ypts[j],zpts[k]].add( factor, mode='drop')

        for j in range(len(ypts)):
            for i in range(len(xpts)):
                for k in range(len(zpts)):
                    factor = dJz * Wz_[i, j, k, :]
                    Jz = Jz.at[xpts[i],ypts[j],zpts[k]].add( factor, mode='drop')
        # Deposit the currents onto the grid using the Esirkepov weights

    alpha = constants['alpha']
    Jx = digital_filter(Jx, alpha)
    Jy = digital_filter(Jy, alpha)
    Jz = digital_filter(Jz, alpha)
    J = (Jx, Jy, Jz)
    # apply a digital filter to the current density arrays

    return J

def get_3D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=None):
    """
    Compute Esirkepov current-deposition weights for 3D shape functions.

    This function builds the per-grid-node weight contributions for the three Cartesian
    current components (Jx, Jy, Jz) using the Esirkepov scheme (volume-based, charge-
    conserving deposition). The calculation is performed for all combinations of the
    discrete shape-function weights in x, y and z and for all particles, yielding
    arrays that can be directly summed into grid current arrays.

    Parameters
    ----------
    x_weights : sequence or array-like of length Nx
        1D sequence of x-direction shape-function weights. Each element may be a
        scalar or a 1D array of length N_particles representing per-particle weights.
    y_weights : sequence or array-like of length Ny
        1D sequence of y-direction shape-function weights (same conventions as
        x_weights).
    z_weights : sequence or array-like of length Nz
        1D sequence of z-direction shape-function weights (same conventions as
        x_weights).
    old_x_weights : sequence or array-like of length Nx
        The previous timestep x-direction shape-function weights (for computing the
        Esirkepov time-centered deposition).
    old_y_weights : sequence or array-like of length Ny
        The previous timestep y-direction shape-function weights.
    old_z_weights : sequence or array-like of length Nz
        The previous timestep z-direction shape-function weights.
    N_particles : int
        Number of particles. Used to set the trailing dimension of the returned
        weight arrays. If the weight elements are scalars, they will be broadcasted
        to length N_particles; if the weight elements are arrays they should have
        length N_particles.
    null_dim : any, optional
        Placeholder argument retained for API compatibility. Not used by the current
        implementation.

    Returns
    -------
    Wx_, Wy_, Wz_ : tuple of jax.numpy.ndarray
        Three 4D arrays containing the Esirkepov-deposition weights for the x, y
        and z current components, respectively.

        - Wx_ has shape (Nx, Ny, Nz, N_particles) and is indexed as
          Wx_[i, j, k, p] representing the contribution to Jx at grid node (i,j,k)
          from particle p.
        - Wy_ has shape (Ny, Nx, Nz, N_particles) and is indexed as
          Wy_[j, i, k, p] so that the second axis corresponds to the x-index when
          mapping to grid nodes for the y-component.
        - Wz_ has shape (Nz, Nx, Ny, N_particles) and is indexed as
          Wz_[k, i, j, p] so that the second and third axes correspond to the x
          and y indices when mapping to grid nodes for the z-component.
    """

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros( (len(y_weights),len(x_weights),len(z_weights), N_particles) )
    Wz_ = jnp.zeros( (len(z_weights),len(x_weights),len(y_weights), N_particles) )


    for i in range(len(x_weights)):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i,j,k,:].set( (x_weights[i] - old_x_weights[i]) * ( 1/3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])     \
                                                    +  1/6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k]) ) )

                Wy_ = Wy_.at[j,i,k,:].set( (y_weights[i] - old_y_weights[i]) * ( 1/3 * (x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k])     \
                                                    +  1/6 * (x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k]) ) )

                Wz_ = Wz_.at[k,i,j,:].set( (z_weights[i] - old_z_weights[i]) * ( 1/3 * (x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j])     \
                                                    +  1/6 * (x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j]) ) )

    return Wx_, Wy_, Wz_

def get_2D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=2):
    """
    Compute 2D Esirkepov deposition weights from 3D per-axis shape-function weights.
    This function builds per-grid-point deposition weight factors used by the
    Esirkepov current-deposition scheme when collapsing a 3D particle-to-grid
    scatter onto a 2D simulation plane. The returned arrays contain the weight
    contributions for each grid index combination and are replicated across a
    particle axis (size N_particles). The function uses JAX primitives (jnp and
    lax.cond) and is written to be JIT-friendly.
    Parameters
    ----------
    x_weights : array_like, shape (Nx,)
        1D array of current shape-function weights along the x axis for the new
        particle positions.
    y_weights : array_like, shape (Ny,)
        1D array of current shape-function weights along the y axis for the new
        particle positions.
    z_weights : array_like, shape (Nz,)
        1D array of current shape-function weights along the z axis for the new
        particle positions.
    old_x_weights : array_like, shape (Nx,)
        1D array of shape-function weights along x for the particle positions at
        the previous timestep.
    old_y_weights : array_like, shape (Ny,)
        1D array of shape-function weights along y for the particle positions at
        the previous timestep.
    old_z_weights : array_like, shape (Nz,)
        1D array of shape-function weights along z for the particle positions at
        the previous timestep.
    N_particles : int
        Number of particles (final axis length). The per-(i,j,k) scalar factor is
        replicated along this axis, so the returned arrays have a trailing
        dimension of length N_particles.
    null_dim : int, optional (default=2)
        Index of the dimension that is “null” (the coordinate normal to the 2D
        plane). Allowed values:
          - 0: the simulation plane is the yz plane (x is null)
          - 1: the simulation plane is the xz plane (y is null)
          - 2: the simulation plane is the xy plane (z is null)
        The computed formulas for the weight factors change depending on this value.
    Returns
    -------
    Wx_ : jnp.ndarray, shape (Nx, Ny, Nz, N_particles)
        Weight contributions associated with the x-directed current component for
        each (i, j, k) grid index and each particle (replicated along the last axis).
    Wy_ : jnp.ndarray, shape (Ny, Nx, Nz, N_particles)
        Weight contributions associated with the y-directed current component.
        Note the swapped first two spatial axes: indexing is (j, i, k, p) in the
        internal layout used by the routine.
    Wz_ : jnp.ndarray, shape (Nz, Nx, Ny, N_particles)
        Weight contributions associated with the z-directed current component.
        The first axis corresponds to z indices (k), followed by x (i) and y (j).
    """
    

    d_Sx = []
    d_Sy = []
    d_Sz = []

    for i in range(len(x_weights)):
        d_Sx.append(x_weights[i] - old_x_weights[i])
        d_Sy.append(y_weights[i] - old_y_weights[i])
        d_Sz.append(z_weights[i] - old_z_weights[i])
    # Calculate the difference in weights for the central grid points

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros( (len(y_weights),len(x_weights),len(z_weights), N_particles) )
    Wz_ = jnp.zeros( (len(z_weights),len(x_weights),len(y_weights), N_particles) )


    for i in range(len(x_weights)):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):

                factor = lax.cond(
                    null_dim == 0,
                    # if the 2D plane is in the yz plane
                    lambda _: 1/3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])     \
                        +  1/6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k]),

                    
                    # if the 2D plane is in the xz or xy plane
                    lambda _: 1/2 * d_Sx[i] * lax.cond(
                                            null_dim == 1,
                                            # if the 2D plane is in the xz plane
                                            lambda _: z_weights[k] + old_z_weights[k],
                                            # if the 2D plane is in the xy plane
                                            lambda _: y_weights[j] + old_y_weights[j],
                                        operand=None
                                        ),
    
                    operand=None
                )
                Wx_ = Wx_.at[i,j,k,:].set( factor )


                factor = lax.cond(
                    null_dim == 1,
                    # if the 2D plane is in the xz plane
                    lambda _: 1/3 * (x_weights[j] * z_weights[k] + old_x_weights[j] * old_z_weights[k])     \
                        +  1/6 * (x_weights[j] * old_z_weights[k] + old_x_weights[j] * z_weights[k]),

                    # if the 2D plane is in the yz or xy plane
                    lambda _: 1/2 * d_Sy[i] * lax.cond(
                                            null_dim == 0,
                                            # if the 2D plane is in the yz plane
                                            lambda _: z_weights[k] + old_z_weights[k],
                                            # if the 2D plane is in the xy plane
                                            lambda _: x_weights[j] + old_x_weights[j],
                                        operand=None
                                        ),
    
                    operand=None
                )
                
                Wy_ = Wy_.at[j,i,k,:].set( factor )
                

                factor = lax.cond(
                    null_dim == 2,
                    # if the 2D plane is in the xy plane
                    lambda _: 1/3 * (x_weights[j] * y_weights[k] + old_x_weights[j] * old_y_weights[k])     \
                        +  1/6 * (x_weights[j] * old_y_weights[k] + old_x_weights[j] * y_weights[k]),

                    # if the 2D plane is in the yz or xz plane
                    lambda _: 1/2 * d_Sz[i] * lax.cond(
                                            null_dim == 0,
                                            # if the 2D plane is in the yz plane
                                            lambda _: y_weights[k] + old_y_weights[k],
                                            # if the 2D plane is in the xz plane
                                            lambda _: x_weights[j] + old_x_weights[j],
                                        operand=None
                                        ),
    
                    operand=None
                )

                Wz_ = Wz_.at[k,i,j,:].set( factor )

    return Wx_, Wy_, Wz_


def get_1D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, dim=0):
    """
    Compute 1D Esirkepov deposition weights for each grid-line and particle.
    This function builds the three component weight tensors (Wx_, Wy_, Wz_) used
    in Esirkepov-style, charge-conserving current deposition along a 1D line
    aligned with one of the coordinate axes. The weight for the axis parallel to
    the 1D line is computed as the difference between current and old weights;
    the weights for the two transverse axes are computed as the average of the
    current and old weights.
    Parameters
    ----------
    x_weights : array-like (len_x,) or (len_x, N_particles)
        Weight values along the x grid at the current timestep. Each element
        x_weights[i] can be a scalar (same for all particles) or a length-N_particles
        array giving per-particle weights at x-index i.
    old_x_weights : array-like (len_x,) or (len_x, N_particles)
        Weight values along the x grid at the previous timestep (same shape rules
        as x_weights).
    y_weights, old_y_weights : array-like (len_y,) or (len_y, N_particles)
        Current and previous weights along the y grid.
    z_weights, old_z_weights : array-like (len_z,) or (len_z, N_particles)
        Current and previous weights along the z grid.
    N_particles : int
        Number of particles (N_particles). Returned arrays have a particle axis
        of this length. If input weight vectors already include a particle axis
        (shape (len_axis, N_particles)), N_particles should match that axis.
    dim : int, optional (default=0)
        Direction of the 1D line: 0 => x, 1 => y, 2 => z. Determines which axis
        uses the "difference" rule and which use the "average" rule.
    Returns
    -------
    Wx_ : jax.numpy.ndarray, shape (len_x, len_y, len_z, N_particles)
        Weight tensor for the x-component. For the axis matching dim (dim==0),
        Wx_[i,j,k,:] = x_weights[i] - old_x_weights[i]. For the two transverse axes
        the factor is 0.5*(current + old) from the corresponding transverse weight.
    Wy_ : jax.numpy.ndarray, shape (len_y, len_x, len_z, N_particles)
        Weight tensor for the y-component. For dim==1 uses y_weights - old_y_weights
        along the primary axis; otherwise uses averages of the transverse weights.
        Note the axes ordering: the first index of Wy_ corresponds to y grid index.
    Wz_ : jax.numpy.ndarray, shape (len_z, len_x, len_y, N_particles)
        Weight tensor for the z-component. For dim==2 uses z_weights - old_z_weights
        along the primary axis; otherwise uses averages of the transverse weights.
        The first index of Wz_ corresponds to z grid index.
    """


    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros( (len(y_weights),len(x_weights),len(z_weights), N_particles) )
    Wz_ = jnp.zeros( (len(z_weights),len(x_weights),len(y_weights), N_particles) )


    for i in range(len(x_weights)):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):

                factor = lax.cond(
                    dim == 0,
                    # if the 1D line is in the x direction
                    lambda _: old_x_weights[i] - x_weights[i],

                    # if the 1D line is in the y or z direction
                    lambda _: lax.cond(
                                    dim == 1,
                                    # if the 1D line is in the y direction
                                    lambda _: 1/2 * (y_weights[j] + old_y_weights[j]),
                                    # if the 1D line is in the z direction
                                    lambda _: 1/2 * (z_weights[k] + old_z_weights[k]),
                                    operand=None
                                        ),
    
                    operand=None
                )
                Wx_ = Wx_.at[i,j,k,:].set( factor )


                
                factor = lax.cond(
                    dim == 1,
                    # if the 1D line is in the y direction
                    lambda _: old_y_weights[i] - y_weights[i],

                    # if the 1D line is in the x or z direction
                    lambda _: lax.cond(
                                    dim == 0,
                                    # if the 1D line is in the x direction
                                    lambda _: 1/2 * (x_weights[j] + old_x_weights[j]),
                                    # if the 1D line is in the z direction
                                    lambda _: 1/2 * (z_weights[k] + old_z_weights[k]),
                                    operand=None
                                        ),
    
                    operand=None
                )

                Wy_ = Wy_.at[j,i,k,:].set( factor )

                factor = lax.cond(
                    dim == 2,
                    # if the 1D line is in the z direction
                    lambda _: old_z_weights[i] - z_weights[i],

                    # if the 1D line is in the x or y direction
                    lambda _: lax.cond(
                                    dim == 0,
                                    # if the 1D line is in the x direction
                                    lambda _: 1/2 * (x_weights[j] + old_x_weights[j]),
                                    # if the 1D line is in the y direction
                                    lambda _: 1/2 * (y_weights[k] + old_y_weights[k]),
                                    operand=None
                                        ),
    
                    operand=None
                )

                Wz_ = Wz_.at[k,i,j,:].set( factor )

    return Wx_, Wy_, Wz_

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