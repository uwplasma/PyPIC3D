import jax
from jax import jit
import jax.numpy as jnp
from jax import lax
# import external libraries

@jit
def compute_rho(particles, rho, world):
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

    rho = jnp.zeros_like(rho)
    # reset rho to zero

    for species in particles:
        shape_factor = species.get_shape()
        # get the shape factor of the species, which determines the weighting function
        N_particles = species.get_number_of_particles()
        charge = species.get_charge()
        # get the number of particles and their charge
        particle_x, particle_y, particle_z = species.get_position()
        # get the position of the particles in the species

        def add_to_rho(particle, rho):
            x = particle_x.at[particle].get()
            y = particle_y.at[particle].get()
            z = particle_z.at[particle].get()
            # get the position of the particle
            rho = lax.cond(
                shape_factor == 1,
                lambda _: first_order_weighting(charge, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind),
                lambda _: second_order_weighting(charge, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind),
                operand=None
            )
            # use the shape factor to determine which weighting function to use and update rho accordingly
            return rho
        # add the particle species to the charge density array
        rho = jax.lax.fori_loop(0, N_particles, add_to_rho, rho)
    # The above loop iterates over each species of particles and updates the charge density array rho

    return rho

@jit
def first_order_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind):
    """
    Distribute the charge of a particle to the surrounding grid points.

    Args:
        q (float): Charge of the particle.
        x (float): x-coordinate of the particle.
        y (float): y-coordinate of the particle.
        z (float): z-coordinate of the particle.
        rho (ndarray): Charge density array.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        x_wind (float): Window in the x-direction.
        y_wind (float): Window in the y-direction.
        z_wind (float): Window in the z-direction.

    Returns:
        ndarray: Updated charge density array.
    """

    Nx, Ny, Nz = rho.shape
    # get the shape of the charge density array

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

    dv = dx * dy * dz
    # Calculate the volume of each grid point

    ##################################### WEIGHTING FACTORS ##########################################################
    Sx0 = 1 - (deltax/dx)
    Sy0 = 1 - (deltay/dy)
    Sz0 = 1 - (deltaz/dz)
    # calculate the weight for the center grid points
    Sx1 = deltax/dx
    Sy1 = deltay/dy
    Sz1 = deltaz/dz
    # calculate the weight for the next grid points
    ####################################################################################################################

    ####################################### CHARGE DISTRIBUTION ########################################################
    rho = rho.at[x0, y0, z0].add((q / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x1, y0, z0].add((q / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x0, y1, z0].add((q / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    rho = rho.at[x0, y0, z1].add((q / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    rho = rho.at[x1, y1, z0].add((q / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    rho = rho.at[x1, y0, z1].add((q / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    rho = rho.at[x0, y1, z1].add((q / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    rho = rho.at[x1, y1, z1].add((q / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    # Distribute the charge of the particle to the grid points
    #####################################################################################################################

    return rho

@jit
def second_order_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind):
    """
    Distribute the charge of a particle to the surrounding grid points using second-order weighting.

    Args:
        q (float): Charge of the particle.
        x (float): x-coordinate of the particle.
        y (float): y-coordinate of the particle.
        z (float): z-coordinate of the particle.
        rho (ndarray): Charge density array.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        x_wind (float): Window in the x-direction.
        y_wind (float): Window in the y-direction.
        z_wind (float): Window in the z-direction.

    Returns:
        ndarray: Updated charge density array.
    """

    Nx, Ny, Nz = rho.shape
    # get the shape of the charge density array

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

    dv = dx * dy * dz
    # Calculate the volume of each grid point


    ####################### WEIGHTING FACTORS ############################################################################
    Sx0 = (3/4) - (deltax/dx)**2
    Sy0 = (3/4) - (deltay/dy)**2
    Sz0 = (3/4) - (deltaz/dz)**2
    # Calculate the weights for the central grid points

    Sx1 = jax.lax.cond(
        deltax <= dx/2,
        lambda _: (1/2) * ((1/2) - (deltax/dx))**2,
        lambda _: jnp.array(0.0, dtype=deltax.dtype),
        operand=None
    )
    Sy1 = jax.lax.cond(
        deltay <= dy/2,
        lambda _: (1/2) * ((1/2) - (deltay/dy))**2,
        lambda _: jnp.array(0.0, dtype=deltay.dtype),
        operand=None
    )
    Sz1 = jax.lax.cond(
        deltaz <= dz/2,
        lambda _: (1/2) * ((1/2) - (deltaz/dz))**2,
        lambda _: jnp.array(0.0, dtype=deltaz.dtype),
        operand=None
    )
    # Calculate the weights for the next grid points

    Sx_minus1 = jax.lax.cond(
        deltax <= dx/2,
        lambda _: (1/2) * ((1/2) + (deltax/dx))**2,
        lambda _: jnp.array(0.0, dtype=deltax.dtype),
        operand=None
    )

    Sy_minus1 = jax.lax.cond(
        deltay <= dy/2,
        lambda _: (1/2) * ((1/2) + (deltay/dy))**2,
        lambda _: jnp.array(0.0, dtype=deltay.dtype),
        operand=None
    )
    Sz_minus1 = jax.lax.cond(
        deltaz <= dz/2,
        lambda _: (1/2) * ((1/2) + (deltaz/dz))**2,
        lambda _: jnp.array(0.0, dtype=deltaz.dtype),
        operand=None
    )
    # Calculate the weights for the previous grid points
    #####################################################################################################################

    ###################################### CHARGE DISTRIBUTION ##########################################################
    rho = rho.at[x0, y0, z0].add((q / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x1, y0, z0].add((q / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x0, y1, z0].add((q / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    rho = rho.at[x0, y0, z1].add((q / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    rho = rho.at[x1, y1, z0].add((q / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    rho = rho.at[x1, y0, z1].add((q / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    rho = rho.at[x0, y1, z1].add((q / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    rho = rho.at[x1, y1, z1].add((q / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    rho = rho.at[x_minus1, y0, z0].add((q / dv) * Sx_minus1 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x0, y_minus1, z0].add((q / dv) * Sx0 * Sy_minus1 * Sz0, mode='drop')
    rho = rho.at[x0, y0, z_minus1].add((q / dv) * Sx0 * Sy0 * Sz_minus1, mode='drop')
    rho = rho.at[x_minus1, y_minus1, z0].add((q / dv) * Sx_minus1 * Sy_minus1 * Sz0, mode='drop')
    rho = rho.at[x_minus1, y0, z_minus1].add((q / dv) * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    rho = rho.at[x0, y_minus1, z_minus1].add((q / dv) * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    rho = rho.at[x_minus1, y_minus1, z_minus1].add((q / dv) * Sx_minus1 * Sy_minus1 * Sz_minus1, mode='drop')
    # Distribute the charge of the particle to the grid points
    #####################################################################################################################


    return rho


@jit
def wrap_around(ix, size):
    """Wrap around index to ensure it is within bounds."""
    return jax.lax.cond(
        ix > size - 1,
        lambda _: ix - size,
        lambda _: ix,
        operand=None
    )