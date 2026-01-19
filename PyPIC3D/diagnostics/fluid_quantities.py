
from PyPIC3D.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.utils import wrap_around

import jax
import jax.numpy as jnp
from jax import jit


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

@jit
def compute_velocity_field(particles, field, direction, world):
    """
    Compute the velocity field (v) for a given set of particles in a simulation world.
    Parameters:
    particles (list): A list of particle species, each containing methods to get the number of particles,
                      their positions, and their mass.
    field (ndarray): The initial velocity field array to be updated.
    direction (int): The direction along which to compute the velocity field (0: x, 1: y, 2: z).
    world (dict): A dictionary containing the simulation world parameters, including:
                  - 'dx': Grid spacing in the x-direction.
                  - 'dy': Grid spacing in the y-direction.
                  - 'dz': Grid spacing in the z-direction.
                  - 'x_wind': Window size in the x-direction.
                  - 'y_wind': Window size in the y-direction.
                  - 'z_wind': Window size in the z-direction.
    Returns:
    ndarray: The updated velocity field array.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    Nx, Ny, Nz = field.shape
    # get the shape of the velocity field array

    field = jnp.zeros_like(field)
    # reset field to zero

    for species in particles:
        shape_factor = species.get_shape()
        # get the shape factor of the species, which determines the weighting function
        N_particles = species.get_number_of_particles()
        # get the number of particles
        x, y, z = species.get_position()
        # get the position of the particles in the species
        vx, vy, vz = species.get_velocity()
        # get the velocity of the particles in the species

        dv = jnp.array([vx, vy, vz])[direction] / N_particles
        # select the velocity component based on the direction

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
                    field = field.at[xpts[i], ypts[j], zpts[k]].add( dv * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # distribute the velocity of the particles to the grid points using the weighting factors

    return field




@jit
def compute_pressure_field(particles, field, velocity_field, direction, world):

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    Nx, Ny, Nz = field.shape
    # get the shape of the velocity field array

    field = jnp.zeros_like(field)
    # reset field to zero

    for species in particles:
        shape_factor = species.get_shape()
        # get the shape factor of the species, which determines the weighting function
        x, y, z = species.get_position()
        # get the position of the particles in the species
        vx, vy, vz = species.get_velocity()
        # get the velocity of the particles in the species


        v = jnp.array([vx, vy, vz])[direction]
        # select the velocity component based on the direction

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
                    vbar = v - velocity_field.at[xpts[i], ypts[j], zpts[k]].get()

                    field = field.at[xpts[i], ypts[j], zpts[k]].add( vbar**2 * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # distribute the pressure moment of the particles to the grid points using the weighting factors

    return field