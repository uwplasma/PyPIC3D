# Christopher Woolford Dec 5, 2024
# This contains the public evolution-loop names for the 3D PIC code.

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from PyPIC3D.deposition.current_methods import CURRENT_ESIRKEPOV, CURRENT_J_FROM_RHOV
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.particles.tiled_particle_refresh import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.pusher.tiled_pusher import tiled_particle_push
from PyPIC3D.solvers.electrostatic_yee import calculate_electrostatic_fields
from PyPIC3D.solvers.electrostatic_yee import calculate_tiled_electrostatic_fields
from PyPIC3D.solvers.first_order_yee import _update_B_global, _update_E_global
from PyPIC3D.solvers.yee_tiled import update_tiled_B, update_tiled_E
from PyPIC3D.utils import add_external_fields


__all__ = ["time_loop_electrodynamic", "time_loop_electrostatic"]


def time_loop_electrodynamic(
    particles,
    species_config,
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
    Advance a tiled electrodynamic PIC system by one time step.
    """

    del solver

    if len(fields) == 7:
        E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state = fields
        overflow_previous = jnp.asarray(False)
    else:
        E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state, overflow_previous = fields
    # unpack the tile-major field state. ``overflow_previous`` carries the
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
    # keep field-only helper behavior for standalone Maxwell equivalence checks.

    push_E_tiles, push_B_tiles = add_external_fields(E_tiles, B_tiles, external_fields)
    # particles see evolved fields plus external-only fields, as in the standard
    # electrodynamic path.

    particles = tiled_particle_push(
        particles,
        species_config,
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
        current_deposition = partial(J_from_rhov, filter="none")
    else:
        current_deposition = J_func

    def esirkepov_step(state):
        particles, J_tiles, overflow_previous = state
        # Esirkepov needs old and new particle positions. The deposition kernel
        # predicts the new positions locally, then the actual particle state is
        # advanced and retiled after the current has been computed.
        J_tiles = current_deposition(particles, species_config, J_tiles, constants, world, tile_shape=tile_shape, g=g)
        particles = update_tiled_particle_positions(particles, species_config, world["dt"])
        particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
        overflow = overflow_previous | overflow
        return particles, J_tiles, overflow

    def direct_current_step(state):
        particles, J_tiles, overflow_previous = state
        particles = update_tiled_particle_positions(particles, species_config, world["dt"] / 2)
        # update particle positions to the centered direct-current deposition time
        particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
        overflow = overflow_previous | overflow
        # wrap periodic particles and move them into their owning tiles.

        J_tiles = current_deposition(particles, species_config, J_tiles, constants, world, tile_shape=tile_shape, g=g)
        # deposit current directly into tile-local Yee current arrays

        particles = update_tiled_particle_positions(particles, species_config, world["dt"] / 2)
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


def time_loop_electrostatic(
    particles,
    species_config,
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
    Advance a tiled electrostatic PIC system by one time step.

    The particle push and retile use tile-local fields. Charge density is
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

    if tile_shape is None or g is None:
        raise ValueError("tiled electrostatic updates require explicit tile_shape and g.")
    g = int(g)

    push_E_tiles, push_B_tiles = add_external_fields(E_tiles, B_tiles, external_fields)
    # particles see evolved fields plus prescribed external fields

    particles = tiled_particle_push(
        particles,
        species_config,
        push_E_tiles,
        push_B_tiles,
        world,
        constants,
        tile_shape,
        g,
        relativistic=relativistic,
        particle_pusher=particle_pusher,
    )
    # push velocities using the selected tiled particle pusher

    particles = update_tiled_particle_positions(particles, species_config, world["dt"])
    # update particle forward positions before depositing rho

    particles, overflow = refresh_tiled_particle_tiles(particles, world, tile_shape)
    overflow = overflow_previous | overflow
    # keep fixed-capacity tile overflow visible to the Python driver

    E_tiles, phi_tiles, rho_tiles = calculate_tiled_electrostatic_fields(
        world,
        particles,
        species_config,
        constants,
        rho_tiles,
        phi_tiles,
        solver,
        "periodic",
        tile_shape,
        g,
    )
    # solve electrostatic fields from tiled charge density

    fields = (E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "relativistic", "particle_pusher"))
def _time_loop_electrostatic_global_reference(
    particles,
    fields,
    world,
    constants,
    curl_func,
    J_func,
    solver,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Old global electrostatic loop retained as a reference path for tests.
    """

    E, B, J, rho, phi, external_fields = fields
    center_grid = world["grids"]["center"]
    vertex_grid = world["grids"]["vertex"]
    push_E, push_B = add_external_fields(E, B, external_fields)

    for i in range(len(particles)):
        particles[i] = particle_push(
            particles[i],
            push_E,
            push_B,
            center_grid,
            vertex_grid,
            world,
            constants,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )
        particles[i].update_position()
        particles[i].boundary_conditions(world)

    E, phi, rho = calculate_electrostatic_fields(world, particles, constants, rho, phi, solver, "periodic")

    fields = (E, B, J, rho, phi, external_fields)
    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "relativistic", "particle_pusher"))
def _time_loop_electrodynamic_global_reference(
    particles,
    fields,
    world,
    constants,
    curl_func,
    J_func,
    solver,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Old global electrodynamic loop retained as a reference path for tests.
    """

    E, B, J, rho, phi, external_fields, pml_state = fields
    center_grid = world["grids"]["center"]
    vertex_grid = world["grids"]["vertex"]
    push_E, push_B = add_external_fields(E, B, external_fields)

    for i in range(len(particles)):
        particles[i] = particle_push(
            particles[i],
            push_E,
            push_B,
            center_grid,
            vertex_grid,
            world,
            constants,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )
        particles[i].update_position()

    J = J_func(particles, J, constants, world)
    E, pml_state = _update_E_global(E, B, J, world, constants, curl_func, pml_state)
    B, pml_state = _update_B_global(E, B, world, constants, curl_func, pml_state)

    for i in range(len(particles)):
        particles[i].boundary_conditions(world)

    fields = (E, B, J, rho, phi, external_fields, pml_state)
    return particles, fields
