"""Coupled Maxwell-Proca PIC with E'/phi' at n and A' at n-1/2."""

from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.utilities.field_helpers import yee_push_fields
from PyPIC3D.solvers.yee.first_order_yee import update_B, update_E
from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import (
    update_dark_A,
    update_dark_E,
    update_dark_phi,
    compute_dark_B,
    synchronized_dark_fields,
)
from PyPIC3D.solvers.dark_matter_yee.pml import advance_dark_pml


__all__ = ["time_loop_dark_photon"]


def dark_photon_push_fields(E, B, dark_fields, external_fields, static_parameters, dynamic_parameters, pml_state=None):
    """Build the integer-time force fields, including adjoint current filtering."""
    dark_E, _A, _phi, dark_B = synchronized_dark_fields(dark_fields, static_parameters, dynamic_parameters, pml_state)
    mixing = static_parameters.sin_chi
    E = tuple(e - mixing * dark_e for e, dark_e in zip(E, dark_E))
    B = tuple(b - mixing * dark_b for b, dark_b in zip(B, dark_B))
    return yee_push_fields(E, B, external_fields, static_parameters, dynamic_parameters)


def time_loop_dark_photon(particles, species_config, fields, static_parameters, dynamic_parameters):
    """Advance particles and both field sectors by one timestep."""
    E, B, J, rho, phi, external_fields, pml_state, dark_fields, overflow = fields
    dt = dynamic_parameters.dt
    maxwell_pml, dark_pml = (None, None) if pml_state is None else pml_state
    push_E, push_B = dark_photon_push_fields(
        E, B, dark_fields, external_fields, static_parameters, dynamic_parameters, dark_pml,
    )
    particles = particle_push(particles, species_config, push_E, push_B, static_parameters, dynamic_parameters)
    # first push the particles with the integer-time force fields, including adjoint current filtering


    if static_parameters.current_deposition == "esirkepov":
        particles_old = particles
        particles = update_tiled_particle_positions(particles, species_config, dt)
        J = Esirkepov_current(
            particles_old, particles, species_config, J,
            static_parameters, dynamic_parameters, coordinate_velocity=particles.u,
        )
    else:
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
        particles, new_overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        overflow = overflow | new_overflow
        J = J_from_rhov(particles, species_config, J, static_parameters, dynamic_parameters)
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
    particles, new_overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
    overflow = overflow | new_overflow
    # compute the current density from the updated particle positions, and refresh the particle tiles to ensure consistency across tile boundaries

    B, maxwell_pml = update_B(E, B, static_parameters, dynamic_parameters, maxwell_pml)
    E, maxwell_pml = update_E(E, B, J, static_parameters, dynamic_parameters, maxwell_pml)
    B, maxwell_pml = update_B(E, B, static_parameters, dynamic_parameters, maxwell_pml)
    # leapfrog integrate the electromagnetic fields with the updated current density, including PML boundary conditions

    if dark_pml is not None:
        dark_fields, dark_pml = advance_dark_pml(
            dark_fields, J, dark_pml, static_parameters, dynamic_parameters,
        )
        pml_state = maxwell_pml, dark_pml
    else:
        E_dark, A_dark, phi_dark = dark_fields
        A_dark = update_dark_A(E_dark, A_dark, phi_dark, static_parameters, dynamic_parameters, dt)
        B_dark = compute_dark_B(A_dark, static_parameters, dynamic_parameters)
        phi_dark = update_dark_phi(A_dark, phi_dark, static_parameters, dynamic_parameters, dt)
        E_dark = update_dark_E(E_dark, B_dark, A_dark, J, static_parameters, dynamic_parameters, dt)
        dark_fields = (E_dark, A_dark, phi_dark)
        # leapfrog integrate the dark photon fields with the updated current density

    return particles, (E, B, J, rho, phi, external_fields, pml_state, dark_fields, overflow)
