from PyPIC3D.collisions import apply_collisions
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.utilities.field_helpers import add_external_fields

from .electrostatic_yee import calculate_electrostatic_fields


__all__ = ["time_loop_electrostatic"]


def time_loop_electrostatic(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """
    Advance a tiled electrostatic PIC system by one time step.

    The particle push, charge deposition, local Schwarz Poisson solve, and
    electrostatic gradient all remain in tiled storage.
    """

    E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state, overflow_previous = fields
    # unpack the tiled field state

    dt = dynamic_parameters.dt
    # get the dynamic timestep used by the tiled electrostatic step

    push_E_tiles, push_B_tiles = add_external_fields(E_tiles, B_tiles, external_fields)
    # particles see evolved fields plus prescribed external fields

    particles = particle_push(
        particles,
        species_config,
        push_E_tiles,
        push_B_tiles,
        static_parameters,
        dynamic_parameters,
    )
    # push velocities using the selected tiled particle pusher
    particles, collision_overflow = apply_collisions(particles, species_config, static_parameters, dynamic_parameters)
    overflow_previous = overflow_previous | collision_overflow

    particles = update_tiled_particle_positions(particles, species_config, dt)
    # update particle forward positions before depositing rho

    particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
    overflow = overflow_previous | overflow
    # keep fixed-capacity tile overflow visible to the Python driver

    E_tiles, phi_tiles, rho_tiles = calculate_electrostatic_fields(
        static_parameters,
        dynamic_parameters,
        particles,
        species_config,
        rho_tiles,
        phi_tiles
    )
    # solve electrostatic fields from tiled charge density

    fields = (E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields
