import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

from PyPIC3D.pusher.boris import (
    boris_single_particle,
    interpolate_field_to_particles,
    relativistic_boris_single_particle,
)
from PyPIC3D.pusher.higuera_cary import higuera_cary_single_particle


VALID_PARTICLE_PUSHERS = ("boris", "higuera_cary")


def validate_particle_pusher(particle_pusher):
    """
    Validate the particle pusher name before entering jitted simulation code.
    """
    if particle_pusher not in VALID_PARTICLE_PUSHERS:
        raise ValueError(f"particle_pusher must be one of {VALID_PARTICLE_PUSHERS}, got {particle_pusher!r}")


@partial(jit, static_argnames=("periodic", "relativistic", "particle_pusher"))
def particle_push(particles, E, B, grid, staggered_grid, dt, constants, periodic=True, relativistic=True, particle_pusher="boris"):
    """
    Updates particle velocities using the selected particle pusher.

    Args:
        particles (Particles): The particles to be updated.
        E (tuple): Electric field components (Ex, Ey, Ez).
        B (tuple): Magnetic field components (Bx, By, Bz).
        grid (Grid): The grid on which the fields are defined.
        staggered_grid (Grid): The staggered grid for field interpolation.
        dt (float): The time step for the update.
        constants (dict): Dictionary containing physical constants.
        periodic (bool): Kept for compatibility with the existing call signature.
        relativistic (bool): Selects relativistic or non-relativistic Boris.
        particle_pusher (str): Particle pusher name: "boris" or "higuera_cary".

    Returns:
        Particles: The particles with updated velocities.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_forward_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles

    shape_factor = particles.get_shape()
    # get the shape factor of the particles

    ################## INTERPOLATION GRIDS ###################################
    Ex_grid = staggered_grid[0], grid[1], grid[2]
    Ey_grid = grid[0], staggered_grid[1], grid[2]
    Ez_grid = grid[0], grid[1], staggered_grid[2]
    # create the grids for the electric field components
    Bx_grid = grid[0], staggered_grid[1], staggered_grid[2]
    By_grid = staggered_grid[0], grid[1], staggered_grid[2]
    Bz_grid = staggered_grid[0], staggered_grid[1], grid[2]
    # create the staggered grids for the magnetic field components
    ##########################################################################

    ################## INTERPOLATE FIELDS TO PARTICLE POSITIONS ##############
    Ex, Ey, Ez = E
    # unpack the electric field components
    efield_atx = interpolate_field_to_particles(Ex, x, y, z, Ex_grid, shape_factor, ghost_cells=True)
    efield_aty = interpolate_field_to_particles(Ey, x, y, z, Ey_grid, shape_factor, ghost_cells=True)
    efield_atz = interpolate_field_to_particles(Ez, x, y, z, Ez_grid, shape_factor, ghost_cells=True)
    # calculate the electric field at the particle positions on the Yee-staggered component grids
    Bx, By, Bz = B
    # unpack the magnetic field components
    bfield_atx = interpolate_field_to_particles(Bx, x, y, z, Bx_grid, shape_factor, ghost_cells=True)
    bfield_aty = interpolate_field_to_particles(By, x, y, z, By_grid, shape_factor, ghost_cells=True)
    bfield_atz = interpolate_field_to_particles(Bz, x, y, z, Bz_grid, shape_factor, ghost_cells=True)
    # calculate the magnetic field at the particle positions on the Yee-staggered component grids
    #########################################################################

    #################### PARTICLE PUSHER ####################################
    if jnp.ndim(q) == 0:
        boris_vmap = jax.vmap(
            boris_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
        )
        relativistic_boris_vmap = jax.vmap(
            relativistic_boris_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
        )
        higuera_cary_vmap = jax.vmap(
            higuera_cary_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
        )
    else:
        boris_vmap = jax.vmap(
            boris_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )
        relativistic_boris_vmap = jax.vmap(
            relativistic_boris_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )
        higuera_cary_vmap = jax.vmap(
            higuera_cary_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )
    # vectorize the selected particle pusher for batch processing

    if particle_pusher == "boris":
        newvx, newvy, newvz = jax.lax.cond(
            relativistic == True,
            lambda _: relativistic_boris_vmap(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants),
            lambda _: boris_vmap(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants),
            operand=None
        )
        # apply the Boris algorithm to update particle velocities
    elif particle_pusher == "higuera_cary":
        newvx, newvy, newvz = higuera_cary_vmap(vx, vy, vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants)
        # apply the Higuera-Cary algorithm to update particle velocities
    else:
        raise ValueError(f"Unknown particle_pusher: {particle_pusher}")
    #########################################################################

    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles
    return particles
