from PyPIC3D.deposition.rho import (
    compute_tiled_mass_density_from_tiled_particles,
    compute_tiled_pressure_field_from_tiled_particles,
    compute_tiled_velocity_field_from_tiled_particles,
)
from PyPIC3D.particles.particle_class import TiledParticles


def compute_mass_density(particles, rho, world, species_config=None):
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
    if isinstance(particles, TiledParticles):
        if getattr(rho, "ndim", 0) == 6:
            return compute_tiled_mass_density_from_tiled_particles(
                particles,
                species_config,
                rho,
                world,
                tile_shape=tuple(int(width) for width in world["tile_shape"]),
                g=int(world["guard_cells"]),
            )
        raise ValueError("compute_mass_density requires tile-major scalar field storage.")

    raise ValueError("Public compute_mass_density requires TiledParticles.")


def compute_velocity_field(particles, field, direction, world, species_config=None):
    if isinstance(particles, TiledParticles):
        if getattr(field, "ndim", 0) == 6:
            return compute_tiled_velocity_field_from_tiled_particles(
                particles,
                species_config,
                field,
                int(direction),
                world,
                tile_shape=tuple(int(width) for width in world["tile_shape"]),
                g=int(world["guard_cells"]),
            )
        raise ValueError("compute_velocity_field requires tile-major scalar field storage.")

    raise ValueError("Public compute_velocity_field requires TiledParticles.")

def compute_pressure_field(particles, field, velocity_field, direction, world, species_config=None):
    if isinstance(particles, TiledParticles):
        if getattr(field, "ndim", 0) == 6:
            return compute_tiled_pressure_field_from_tiled_particles(
                particles,
                species_config,
                field,
                velocity_field,
                int(direction),
                world,
                tile_shape=tuple(int(width) for width in world["tile_shape"]),
                g=int(world["guard_cells"]),
            )
        raise ValueError("compute_pressure_field requires tile-major scalar field storage.")

    raise ValueError("Public compute_pressure_field requires TiledParticles.")
