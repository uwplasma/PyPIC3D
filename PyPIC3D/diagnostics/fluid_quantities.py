from PyPIC3D.deposition.rho import (
    compute_tiled_mass_density_from_tiled_particles,
    compute_tiled_pressure_field_from_tiled_particles,
    compute_tiled_velocity_field_from_tiled_particles,
)
from PyPIC3D.particles.particle_class import TiledParticles


def compute_mass_density(particles, rho, static_parameters, dynamic_parameters, species_config=None):
    """
    Compute the mass density for tile-major particles.
    Parameters:
    particles (list): A list of particle species, each containing methods to get the number of particles,
                      their positions, and their mass.
    rho (ndarray): The initial mass density array to be updated.
    Returns:
    ndarray: The updated mass density array.
    """
    if isinstance(particles, TiledParticles):
        if getattr(rho, "ndim", 0) == 6:
            return compute_tiled_mass_density_from_tiled_particles(
                particles,
                species_config,
                rho,
                static_parameters,
                dynamic_parameters,
            )
        raise ValueError("compute_mass_density requires tile-major scalar field storage.")

    raise ValueError("Public compute_mass_density requires TiledParticles.")


def compute_velocity_field(particles, field, direction, static_parameters, dynamic_parameters, species_config=None):
    if isinstance(particles, TiledParticles):
        if getattr(field, "ndim", 0) == 6:
            return compute_tiled_velocity_field_from_tiled_particles(
                particles,
                species_config,
                field,
                int(direction),
                static_parameters,
                dynamic_parameters,
            )
        raise ValueError("compute_velocity_field requires tile-major scalar field storage.")

    raise ValueError("Public compute_velocity_field requires TiledParticles.")

def compute_pressure_field(particles, field, velocity_field, direction, static_parameters, dynamic_parameters, species_config=None):
    if isinstance(particles, TiledParticles):
        if getattr(field, "ndim", 0) == 6:
            return compute_tiled_pressure_field_from_tiled_particles(
                particles,
                species_config,
                field,
                velocity_field,
                int(direction),
                static_parameters,
                dynamic_parameters,
            )
        raise ValueError("compute_pressure_field requires tile-major scalar field storage.")

    raise ValueError("Public compute_pressure_field requires TiledParticles.")
