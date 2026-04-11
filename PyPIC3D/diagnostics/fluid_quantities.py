
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.boundary_conditions.grid_and_stencil import collapse_axis_stencil, prepare_particle_axis_stencil
from PyPIC3D.boundary_conditions.boundaryconditions import fold_ghost_cells, update_ghost_cells

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
    grid = world['grids']['vertex']
    Nx, Ny, Nz = rho.shape
    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']
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

        _, _, deltax, xpts = prepare_particle_axis_stencil(
            x, grid[0], Nx, shape_factor, bc_x, wind=world['x_wind'], ghost_cells=True
        )
        _, _, deltay, ypts = prepare_particle_axis_stencil(
            y, grid[1], Ny, shape_factor, bc_y, wind=world['y_wind'], ghost_cells=True
        )
        _, _, deltaz, zpts = prepare_particle_axis_stencil(
            z, grid[2], Nz, shape_factor, bc_z, wind=world['z_wind'], ghost_cells=True
        )

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )
        # get the weighting factors based on the shape factor

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights = jnp.asarray(x_weights)
        y_weights = jnp.asarray(y_weights)
        z_weights = jnp.asarray(z_weights)
        xpts, x_weights = collapse_axis_stencil(xpts, x_weights, Nx, ghost_cells=True)
        ypts, y_weights = collapse_axis_stencil(ypts, y_weights, Ny, ghost_cells=True)
        zpts, z_weights = collapse_axis_stencil(zpts, z_weights, Nz, ghost_cells=True)


        for i in range(3):
            for j in range(3):
                for k in range(3):
                    rho = rho.at[xpts[i], ypts[j], zpts[k]].add( dm * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # distribute the mass of the particles to the grid points using the weighting factors

    rho = fold_ghost_cells(rho, bc_x, bc_y, bc_z)
    rho = update_ghost_cells(rho, bc_x, bc_y, bc_z)
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
    grid = world['grids']['vertex']
    Nx, Ny, Nz = field.shape
    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']
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

        _, _, deltax, xpts = prepare_particle_axis_stencil(
            x, grid[0], Nx, shape_factor, bc_x, wind=world['x_wind'], ghost_cells=True
        )
        _, _, deltay, ypts = prepare_particle_axis_stencil(
            y, grid[1], Ny, shape_factor, bc_y, wind=world['y_wind'], ghost_cells=True
        )
        _, _, deltaz, zpts = prepare_particle_axis_stencil(
            z, grid[2], Nz, shape_factor, bc_z, wind=world['z_wind'], ghost_cells=True
        )

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )
        # get the weighting factors based on the shape factor

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights = jnp.asarray(x_weights)
        y_weights = jnp.asarray(y_weights)
        z_weights = jnp.asarray(z_weights)
        xpts, x_weights = collapse_axis_stencil(xpts, x_weights, Nx, ghost_cells=True)
        ypts, y_weights = collapse_axis_stencil(ypts, y_weights, Ny, ghost_cells=True)
        zpts, z_weights = collapse_axis_stencil(zpts, z_weights, Nz, ghost_cells=True)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    field = field.at[xpts[i], ypts[j], zpts[k]].add( dv * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # distribute the velocity of the particles to the grid points using the weighting factors

    field = fold_ghost_cells(field, bc_x, bc_y, bc_z)
    field = update_ghost_cells(field, bc_x, bc_y, bc_z)
    return field




@jit
def compute_pressure_field(particles, field, velocity_field, direction, world):

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    grid = world['grids']['vertex']
    Nx, Ny, Nz = field.shape
    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']
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

        _, _, deltax, xpts = prepare_particle_axis_stencil(
            x, grid[0], Nx, shape_factor, bc_x, wind=world['x_wind'], ghost_cells=True
        )
        _, _, deltay, ypts = prepare_particle_axis_stencil(
            y, grid[1], Ny, shape_factor, bc_y, wind=world['y_wind'], ghost_cells=True
        )
        _, _, deltaz, zpts = prepare_particle_axis_stencil(
            z, grid[2], Nz, shape_factor, bc_z, wind=world['z_wind'], ghost_cells=True
        )

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )
        # get the weighting factors based on the shape factor

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights = jnp.asarray(x_weights)
        y_weights = jnp.asarray(y_weights)
        z_weights = jnp.asarray(z_weights)
        xpts, x_weights = collapse_axis_stencil(xpts, x_weights, Nx, ghost_cells=True)
        ypts, y_weights = collapse_axis_stencil(ypts, y_weights, Ny, ghost_cells=True)
        zpts, z_weights = collapse_axis_stencil(zpts, z_weights, Nz, ghost_cells=True)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    vbar = v - velocity_field.at[xpts[i], ypts[j], zpts[k]].get()

                    field = field.at[xpts[i], ypts[j], zpts[k]].add( vbar**2 * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # distribute the pressure moment of the particles to the grid points using the weighting factors

    field = fold_ghost_cells(field, bc_x, bc_y, bc_z)
    field = update_ghost_cells(field, bc_x, bc_y, bc_z)
    return field
