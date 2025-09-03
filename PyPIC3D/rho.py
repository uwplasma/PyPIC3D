import jax
from jax import jit
import jax.numpy as jnp
from jax import lax
# import external libraries

from PyPIC3D.utils import digital_filter
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

        x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
        y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
        z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)
        # Calculate the nearest grid points

        deltax = x - jnp.floor(x / dx) * dx
        deltay = y - jnp.floor(y / dy) * dy
        deltaz = z - jnp.floor(z / dz) * dz
        # Calculate the difference between the particle position and the nearest grid point

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

@jit
def wrap_around(ix, size):
    """Wrap around index (scalar or 1D array) to ensure it is within bounds."""
    return jnp.where(ix > size - 1, ix - size, ix)

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

    Sx1 = jnp.where(jnp.abs(deltax) <= dx/2, (1/2) * ((1/2) - (deltax/dx))**2, 0.0)
    Sy1 = jnp.where(jnp.abs(deltay) <= dy/2, (1/2) * ((1/2) - (deltay/dy))**2, 0.0)
    Sz1 = jnp.where(jnp.abs(deltaz) <= dz/2, (1/2) * ((1/2) - (deltaz/dz))**2, 0.0)

    Sx_minus1 = jnp.where(jnp.abs(deltax) <= dx/2, (1/2) * ((1/2) + (deltax/dx))**2, 0.0)
    Sy_minus1 = jnp.where(jnp.abs(deltay) <= dy/2, (1/2) * ((1/2) + (deltay/dy))**2, 0.0)
    Sz_minus1 = jnp.where(jnp.abs(deltaz) <= dz/2, (1/2) * ((1/2) + (deltaz/dz))**2, 0.0)

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


@jit
def compute_mass_density(particles, rho, world):
    """
    Compute the mass density (rho) for a given set of particles in a simulation world.
    Parameters:
    particles (list): A list of particle species, each containing methods to get the number of particles,
                      their positions, and their mass.
    rho (ndarray): The initial mass density array to be updated.
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
    Nx, Ny, Nz = rho.shape
    # get the shape of the charge density array

    rho = jnp.zeros_like(rho)
    # reset rho to zero

    for species in particles:
        shape_factor = species.get_shape()
        # get the shape factor of the species, which determines the weighting function
        N_particles = species.get_number_of_particles()
        mass = species.get_mass()
        # get the number of particles and their mass
        dm = mass / dx / dy / dz
        # calculate the mass per unit volume
        x, y, z = species.get_position()
        # get the position of the particles in the species

        x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
        y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
        z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)
        # Calculate the nearest grid points

        deltax = x - jnp.floor(x / dx) * dx
        deltay = y - jnp.floor(y / dy) * dy
        deltaz = z - jnp.floor(z / dz) * dz
        # Calculate the difference between the particle position and the nearest grid point

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
                    rho = rho.at[xpts[i], ypts[j], zpts[k]].add( dm * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # distribute the mass of the particles to the grid points using the weighting factors

    return rho