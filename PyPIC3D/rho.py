import jax
from jax import jit
import jax.numpy as jnp
from jax import lax
# import external libraries

from PyPIC3D.utils import digital_filter, wrap_around
from PyPIC3D.shapes import get_first_order_weights, get_second_order_weights
# import internal libraries

@jit
def compute_rho(particles, rho, world, constants):
    """
    Compute the charge density (rho) for a given set of particles in a simulation world.
    Parameters:
    particles (list): A list of particle species, each containing methods to get the number of particles,
                      their positions, and their charge.
    rho (ndarray): The initial charge density array to be updated.
    world (dict): A dictionary containing the simulation world parameters, including:
                  - 'dx': Grid spacing in the x-direction.
                  - 'dy': Grid spacing in the y-direction.
                  - 'dz': Grid spacing in the z-direction.
                  - 'x_wind': Window size in the x-direction.
                  - 'y_wind': Window size in the y-direction.
                  - 'z_wind': Window size in the z-direction.
    Returns:
    ndarray: The updated charge density array.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    grid = world['grids']['vertex']
    Nx, Ny, Nz = rho.shape
    # get the shape of the charge density array

    rho = jnp.zeros_like(rho)
    # reset rho to zero

    for species in particles:
        shape_factor = species.get_shape()
        # get the shape factor of the species, which determines the weighting function
        N_particles = species.get_number_of_particles()
        q = species.get_charge()
        # get the number of particles and their charge
        dq = q / dx / dy / dz
        # calculate the charge per unit volume
        x, y, z = species.get_position()
        # get the position of the particles in the species


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

        deltax = x - (x0 * dx + grid[0][0])
        deltay = y - (y0 * dy + grid[1][0])
        deltaz = z - (z0 * dz + grid[2][0])
        # calculate the difference based on shape factor

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)
        # ensure indices are within bounds

        x1 = wrap_around(x0 + 1, Nx)
        y1 = wrap_around(y0 + 1, Ny)
        z1 = wrap_around(z0 + 1, Nz)
        # Calculate the index of the next grid point

        x_minus1 = x0 - 1
        y_minus1 = y0 - 1
        z_minus1 = z0 - 1
        # Calculate the index of the previous grid point

        xpts = [x_minus1, x0, x1]
        ypts = [y_minus1, y0, y1]
        zpts = [z_minus1, z0, z1]
        # place all the points in a list

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )
        # get the weighting factors based on the shape factor


        for i in range(3):
            for j in range(3):
                for k in range(3):
                    rho = rho.at[xpts[i], ypts[j], zpts[k]].add( dq * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # distribute the charge of the particles to the grid points using the weighting factors

    alpha = constants['alpha']
    rho = digital_filter(rho, alpha)
    # apply a digital filter to the charge density array

    return rho
