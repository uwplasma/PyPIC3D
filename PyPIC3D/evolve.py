from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.solvers.electrostatic_yee import calculate_tiled_electrostatic_fields
from PyPIC3D.solvers.first_order_yee import update_B, update_E
from PyPIC3D.utils import add_external_fields


__all__ = ["time_loop_electrodynamic", "time_loop_electrostatic"]


def time_loop_electrodynamic(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """
    Advance a tiled electrodynamic PIC system by one time step.
    """

    E, B, J, rho, phi, external_fields, pml_state, overflow_previous = fields
    # unpack the tiled field state

    dt = dynamic_parameters.dt
    # get the dynamic timestep used by the tiled push/deposition sequence

    push_E, push_B = add_external_fields(E, B, external_fields)
    # particles see evolved fields plus external-only fields

    particles = particle_push(
        particles,
        species_config,
        push_E,
        push_B,
        static_parameters,
        dynamic_parameters,
    )
    # use the selected tiled pusher for particle velocities

    def direct_deposition_step(state):
        particles, J_tiles, overflow_previous = state
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
        # update particle positions to the centered direct-current deposition time
        particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        # wrap particles and move them into their owning tiles.
        overflow = overflow_previous | overflow
        # keep fixed-capacity tile overflow visible to the Python driver
        J_tiles = J_from_rhov(
            particles,
            species_config,
            J_tiles,
            static_parameters,
            dynamic_parameters,
        )
        # deposit current directly into tile-local Yee current arrays
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
        # complete the full particle position update
        particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        # refresh tile ownership after the full position update.
        overflow = overflow_previous | overflow
        return particles, J_tiles, overflow
    # if the direct deposition method is selected, first refresh the particle tiles, then deposit current directly into the tiled J arrays

    def esirkepov_deposition_step(state):
        particles, J_tiles, overflow_previous = state
        J_tiles = Esirkepov_current(particles, species_config, J_tiles, static_parameters, dynamic_parameters)
        # deposit current into the tiled J arrays using the Esirkepov method, which requires old and new particle positions
        particles = update_tiled_particle_positions(particles, species_config, dt)
        # update particle positions to the new time step
        particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        # refresh tile ownership after the full position update
        overflow = overflow_previous | overflow
        return particles, J_tiles, overflow
    # if the Esirkepov deposition method is selected, first deposit current into the tiled J arrays, then refresh the particle tiles

    if static_parameters.current_deposition == "esirkepov":
        particles, J, overflow = esirkepov_deposition_step((particles, J, overflow_previous))
    else:
        particles, J, overflow = direct_deposition_step((particles, J, overflow_previous))
    # deposit current into the tiled J arrays using the selected deposition method

    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state)
    # update magnetic field from the previous electric field by half a timestep
    # for no pml, the pml_state is None, and the update_B function returns None for the pml_state

    E, pml_state = update_E(E, B, J, static_parameters, dynamic_parameters, pml_state)
    # update electric field from B and the supplied current
    # for no pml, the pml_state is None, and the update_E function returns None for the pml_state

    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state)
    # update magnetic field from the newly updated electric field by half a timestep
    # for no pml, the pml_state is None, and the update_B function returns None for the pml_state

    fields = (E, B, J, rho, phi, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields


def time_loop_electrostatic(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """
    Advance a tiled electrostatic PIC system by one time step.

    The particle push and retile use tile-local fields. Charge density is
    deposited into tiled scalar storage, assembled for the existing global
    Poisson solve, then the solved potential is tiled again before computing
    tile-local electrostatic E.
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

    particles = update_tiled_particle_positions(particles, species_config, dt)
    # update particle forward positions before depositing rho

    particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
    overflow = overflow_previous | overflow
    # keep fixed-capacity tile overflow visible to the Python driver

    E_tiles, phi_tiles, rho_tiles = calculate_tiled_electrostatic_fields(
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
