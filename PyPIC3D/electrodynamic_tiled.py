import jax.numpy as jnp

from PyPIC3D.deposition.direct_deposition_tiled import direct_J_from_tiled_particles
from PyPIC3D.particles.tiled_particle_refresh import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.tiled_pusher import tiled_particle_push
from PyPIC3D.solvers.yee_tiled import update_tiled_B, update_tiled_E
from PyPIC3D.utils import add_external_fields


def time_loop_electrodynamic_tiled(
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
    Advance a periodic tiled electrodynamic PIC system by one time step.
    """

    del solver

    if len(fields) == 7:
        E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state = fields
        overflow_previous = jnp.asarray(False)
    else:
        E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state, overflow_previous = fields
    # unpack the tile-major field state.  ``overflow_previous`` carries the
    # particle-retile overflow diagnostic out to the Python driver.

    if tile_shape is None:
        tile_shape = tuple(int(width) - 2 for width in E_tiles[0].shape[3:])

    if not hasattr(particles, "active"):
        if pml_state is None:
            E_tiles = update_tiled_E(E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape)
            B_tiles = update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape)
        else:
            E_tiles, pml_state = update_tiled_E(
                E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape, pml_state
            )
            B_tiles, pml_state = update_tiled_B(
                E_tiles, B_tiles, world, constants, curl_func, tile_shape, pml_state
            )
        fields = (E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state)
        return particles, fields
    # keep the original field-only helper behavior for tests and standalone
    # Maxwell equivalence checks that pass a non-particle sentinel.
    push_E_tiles, push_B_tiles = add_external_fields(E_tiles, B_tiles, external_fields)
    # particles see evolved fields plus external-only fields, as in the
    # standard electrodynamic path.

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
    # use the selected tiled pusher for particle velocities

    particles = update_tiled_particle_positions(particles, world)
    # update particle forward positions before current deposition

    particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
    overflow = overflow_previous | overflow
    # wrap periodic particles and move them into their owning tiles.  Overflow
    # means the fixed tile capacity was exceeded and the Python driver must
    # reject the step rather than silently dropping particles.

    if J_func is None:
        J_tiles = direct_J_from_tiled_particles(
            particles,
            J_tiles,
            constants,
            world,
            filter="none",
        )
    else:
        J_tiles = J_func(particles, J_tiles, constants, world)
    # deposit current directly into tile-local Yee current arrays

    if pml_state is None:
        E_tiles = update_tiled_E(E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape)
    else:
        E_tiles, pml_state = update_tiled_E(
            E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape, pml_state
        )
    # update electric field from B and the supplied tiled current

    if pml_state is None:
        B_tiles = update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape)
    else:
        B_tiles, pml_state = update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape, pml_state)
    # update magnetic field from the newly updated electric field

    fields = (E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields
