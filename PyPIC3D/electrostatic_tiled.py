import jax.numpy as jnp

from PyPIC3D.particles.tiled_particle_refresh import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.tiled_pusher import tiled_particle_push
from PyPIC3D.solvers.electrostatic_yee import calculate_tiled_electrostatic_fields
from PyPIC3D.utils import add_external_fields


def time_loop_electrostatic_tiled(
    particles,
    fields,
    world,
    constants,
    curl_func,
    J_func,
    solver,
    tile_shape=None,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Advance a tiled electrostatic PIC system by one time step.

    The particle push and retile use tile-local Yee fields.  Charge density is
    deposited into tiled scalar storage, assembled for the existing global
    Poisson solve, then the solved potential is tiled again before computing
    tile-local electrostatic E.
    """

    del curl_func, J_func

    if len(fields) == 6:
        E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields = fields
        pml_state = None
        overflow_previous = jnp.asarray(False)
    elif len(fields) == 7:
        E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state = fields
        overflow_previous = jnp.asarray(False)
    else:
        E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state, overflow_previous = fields

    if tile_shape is None:
        tile_shape = tuple(int(width) - 2 for width in E_tiles[0].shape[3:])

    push_E_tiles, push_B_tiles = add_external_fields(E_tiles, B_tiles, external_fields)
    # particles see evolved fields plus prescribed external fields

    particles = tiled_particle_push(
        particles,
        push_E_tiles,
        push_B_tiles,
        world,
        constants,
        tile_shape,
        relativistic=relativistic,
        particle_pusher=particle_pusher,
    )
    # push velocities using the selected tiled particle pusher

    particles = update_tiled_particle_positions(particles, world)
    # update particle forward positions before depositing rho

    particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
    overflow = overflow_previous | overflow
    # keep fixed-capacity tile overflow visible to the Python driver

    E_tiles, phi_tiles, rho_tiles = calculate_tiled_electrostatic_fields(
        world,
        particles,
        constants,
        rho_tiles,
        phi_tiles,
        solver,
        "periodic",
        tile_shape,
    )
    # solve electrostatic fields from tiled charge density

    fields = (E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields
