import jax
from jax import jit
import jax.numpy as jnp
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
        N_particles = species.get_number_of_particles()
        charge = species.get_charge()
        particle_x, particle_y, particle_z = species.get_position()
        rho = update_rho(N_particles, particle_x, particle_y, particle_z, dx, dy, dz, charge, x_wind, y_wind, z_wind, rho)
        # add the particle species to the charge density array

    return rho

@jit
def update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, q, x_wind, y_wind, z_wind, rho):
    """
    Update the charge density (rho) based on the positions of particles.

    Args:
        Nparticles (int): Number of particles.
        particlex (array-like): Array containing the x-coordinates of particles.
        particley (array-like): Array containing the y-coordinates of particles.
        particlez (array-like): Array containing the z-coordinates of particles.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        q (float): Charge of a single particle.
        x_wind (array-like): Window function in the x-direction.
        y_wind (array-like): Window function in the y-direction.
        z_wind (array-like): Window in the z-direction.
        rho (array-like): Initial charge density array to be updated.

    Returns:
        array-like: Updated charge density array.
    """

    def addto_rho(particle, rho):
        x = particlex.at[particle].get()
        y = particley.at[particle].get()
        z = particlez.at[particle].get()
        rho = first_order_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind)
        return rho

    rho = jax.lax.fori_loop(0, Nparticles, addto_rho, rho )
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
    # Calculate the nearest grid points
    x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
    y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
    z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)

    # Calculate the difference between the particle position and the nearest grid point
    deltax = (x + x_wind / 2) - x0 * dx
    deltay = (y + y_wind / 2) - y0 * dy
    deltaz = (z + z_wind / 2) - z0 * dz

    # Calculate the index of the next grid point
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Calculate the weights for the surrounding grid points
    wx = deltax / dx
    wy = deltay / dy
    wz = deltaz / dz

    # Calculate the volume of each grid point
    dv = dx * dy * dz


    # Distribute the charge of the particle to the surrounding grid points
    rho = rho.at[x0, y0, z0].add((q / dv) * (1 - wx) * (1 - wy) * (1 - wz), mode='drop')
    rho = rho.at[x1, y0, z0].add((q / dv) *      wx  * (1 - wy) * (1 - wz), mode='drop')
    rho = rho.at[x0, y1, z0].add((q / dv) * (1 - wx) *      wy  * (1 - wz), mode='drop')
    rho = rho.at[x0, y0, z1].add((q / dv) * (1 - wx) * (1 - wy) *      wz, mode='drop')
    rho = rho.at[x1, y1, z0].add((q / dv) *      wx  *      wy  * (1 - wz), mode='drop')
    rho = rho.at[x1, y0, z1].add((q / dv) *      wx  * (1 - wy) *      wz, mode='drop')
    rho = rho.at[x0, y1, z1].add((q / dv) * (1 - wx) *      wy  *      wz, mode='drop')
    rho = rho.at[x1, y1, z1].add((q / dv) *      wx  *      wy  *      wz, mode='drop')

    return rho


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
    # Calculate the nearest grid points
    x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
    y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
    z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)

    # Calculate the difference between the particle position and the nearest grid point
    deltax = (x + x_wind / 2) - x0 * dx
    deltay = (y + y_wind / 2) - y0 * dy
    deltaz = (z + z_wind / 2) - z0 * dz

    # Calculate the index of the next grid point
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x_minus1 = x0 - 1
    y_minus1 = y0 - 1
    z_minus1 = z0 - 1
    # Calculate the index of the previous grid point


    ####################### WEIGHTING FACTORS ##########################################################
    Sx0 = (3/4) - deltax**2
    Sy0 = (3/4) - deltay**2
    Sz0 = (3/4) - deltaz**2
    # Calculate the weights for the central grid points


    Sx1 = jax.lax.cond(
        deltax < 0.5,
        lambda _: (1/2) * ((1/2) - deltax)**2,
        lambda _: 0,
        operand=None
    )
    Sy1 = jax.lax.cond(
        deltay < 0.5,
        lambda _: (1/2) * ((1/2) - deltay)**2,
        lambda _: 0,
        operand=None
    )
    Sz1 = jax.lax.cond(
        deltaz < 0.5,
        lambda _: (1/2) * ((1/2) - deltaz)**2,
        lambda _: 0,
        operand=None
    )
    # Calculate the weights for the next grid points

    Sx_minus1 = jax.lax.cond(
        deltax < 0.5,
        lambda _: (1/2) * ((1/2) + deltax)**2,
        lambda _: 0,
        operand=None
    )

    Sy_minus1 = jax.lax.cond(
        deltay < 0.5,
        lambda _: (1/2) * ((1/2) + deltay)**2,
        lambda _: 0,
        operand=None
    )
    Sz_minus1 = jax.lax.cond(
        deltaz < 0.5,
        lambda _: (1/2) * ((1/2) + deltaz)**2,
        lambda _: 0,
        operand=None
    )
    # Calculate the weights for the previous grid points
    ################################################################################


    dv = dx * dy * dz
    # Calculate the volume of each grid point
    dq = q / dv
    # Calculate the charge per volume

    # Distribute the charge of the particle to the surrounding grid points
    rho = rho.at[x0, y0, z0].add(dq * Sx0 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x1, y0, z0].add(dq * Sx1 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x0, y1, z0].add(dq * Sx0 * Sy1 * Sz0, mode='drop')
    rho = rho.at[x0, y0, z1].add(dq * Sx0 * Sy0 * Sz1, mode='drop')
    rho = rho.at[x1, y1, z0].add(dq * Sx1 * Sy1 * Sz0, mode='drop')
    rho = rho.at[x1, y0, z1].add(dq * Sx1 * Sy0 * Sz1, mode='drop')
    rho = rho.at[x0, y1, z1].add(dq * Sx0 * Sy1 * Sz1, mode='drop')
    rho = rho.at[x1, y1, z1].add(dq * Sx1 * Sy1 * Sz1, mode='drop')
    rho = rho.at[x_minus1, y0, z0].add(dq * Sx_minus1 * Sy0 * Sz0, mode='drop')
    rho = rho.at[x0, y_minus1, z0].add(dq * Sx0 * Sy_minus1 * Sz0, mode='drop')
    rho = rho.at[x0, y0, z_minus1].add(dq * Sx0 * Sy0 * Sz_minus1, mode='drop')
    rho = rho.at[x_minus1, y1, z0].add(dq * Sx_minus1 * Sy1 * Sz0, mode='drop')
    rho = rho.at[x_minus1, y0, z1].add(dq * Sx_minus1 * Sy0 * Sz1, mode='drop')
    rho = rho.at[x0, y_minus1, z1].add(dq * Sx0 * Sy_minus1 * Sz1, mode='drop')
    rho = rho.at[x1, y_minus1, z0].add(dq * Sx1 * Sy_minus1 * Sz0, mode='drop')
    rho = rho.at[x1, y0, z_minus1].add(dq * Sx1 * Sy0 * Sz_minus1, mode='drop')
    rho = rho.at[x_minus1, y1, z1].add(dq * Sx_minus1 * Sy1 * Sz1, mode='drop')
    rho = rho.at[x0, y_minus1, z_minus1].add(dq * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    rho = rho.at[x1, y_minus1, z1].add(dq * Sx1 * Sy_minus1 * Sz1, mode='drop')
    rho = rho.at[x_minus1, y0, z_minus1].add(dq * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    rho = rho.at[x_minus1, y1, z_minus1].add(dq * Sx_minus1 * Sy1 * Sz_minus1, mode='drop')
    rho = rho.at[x1, y1, z_minus1].add(dq * Sx1 * Sy1 * Sz_minus1, mode='drop')
    # Update the charge density array with the distributed charge


    return rho


# def S1(delta):

#     """
#     Calculate the second-order weighting factor for a given delta value.

#     Args:
#         delta (float): The delta value representing the distance from the grid point.

#     Returns:
#         float: The first-order weighting factor.
#     """
#     return jax.lax.cond(
#         delta < 0.5,
#         lambda _: (1/2) * ((1/2) - delta)**2,
#         lambda _: 0,
#         operand=None
#     )

# def S_minus1(delta):
#     """
#     Calculate the second-order weighting factor for a given delta value.

#     Args:
#         delta (float): The delta value representing the distance from the grid point.

#     Returns:
#         float: The first-order weighting factor.
#     """
#     if delta < 0.5:
#         return (1/2) * ( (1/2) + delta )**2
#     else:
#         return 0