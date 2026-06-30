import jax.numpy as jnp
import jax

from PyPIC3D.deposition.direct_deposition_tiled import direct_J_from_tiled_particles
from PyPIC3D.deposition.current_methods import CURRENT_ESIRKEPOV, CURRENT_J_FROM_RHOV
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
    g=None,
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

    if tile_shape is None or g is None:
        raise ValueError("tiled electrodynamic updates require explicit tile_shape and g.")
    g = int(g)

    if not hasattr(particles, "active"):
        if pml_state is None:
            E_tiles = update_tiled_E(E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape, g)
            B_tiles = update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape, g)
        else:
            E_tiles, pml_state = update_tiled_E(
                E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape, g, pml_state
            )
            B_tiles, pml_state = update_tiled_B(
                E_tiles, B_tiles, world, constants, curl_func, tile_shape, g, pml_state
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
        g,
        relativistic=relativistic,
        particle_pusher=particle_pusher,
    )
    # use the selected tiled pusher for particle velocities

    if J_func is None:
        def current_deposition(particles, J_tiles, constants, world, tile_shape=None, g=None):
            return direct_J_from_tiled_particles(
                particles,
                J_tiles,
                constants,
                world,
                filter="none",
                tile_shape=tile_shape,
                g=g,
            )
    else:
        current_deposition = J_func

    def esirkepov_step(state):
        particles, J_tiles, overflow_previous = state
        # Esirkepov needs old and new particle positions.  The deposition kernel
        # predicts the new positions locally, then the actual particle state is
        # advanced and retiled after the current has been computed.
        J_tiles = current_deposition(particles, J_tiles, constants, world, tile_shape=tile_shape, g=g)
        particles = update_tiled_particle_positions(particles, world["dt"])
        particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
        overflow = overflow_previous | overflow
        return particles, J_tiles, overflow

    def direct_current_step(state):
        particles, J_tiles, overflow_previous = state
        particles = update_tiled_particle_positions(particles, world["dt"]/2)
        # update particle positions to the centered direct-current deposition time
        particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
        overflow = overflow_previous | overflow
        # wrap periodic particles and move them into their owning tiles.

        J_tiles = current_deposition(particles, J_tiles, constants, world, tile_shape=tile_shape, g=g)
        # deposit current directly into tile-local Yee current arrays

        particles = update_tiled_particle_positions(particles, world["dt"]/2)
        # complete the full particle position update
        particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
        overflow = overflow_previous | overflow
        # refresh tile ownership after the full position update.
        return particles, J_tiles, overflow

    current_calculation = world.get("current_calculation", CURRENT_J_FROM_RHOV)
    particles, J_tiles, overflow = jax.lax.cond(
        current_calculation == CURRENT_ESIRKEPOV,
        esirkepov_step,
        direct_current_step,
        (particles, J_tiles, overflow_previous),
    )

    if pml_state is None:
        E_tiles = update_tiled_E(E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape, g)
    else:
        E_tiles, pml_state = update_tiled_E(
            E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape, g, pml_state
        )
    # update electric field from B and the supplied tiled current

    if pml_state is None:
        B_tiles = update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape, g)
    else:
        B_tiles, pml_state = update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape, g, pml_state)
    # update magnetic field from the newly updated electric field

    fields = (E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields
