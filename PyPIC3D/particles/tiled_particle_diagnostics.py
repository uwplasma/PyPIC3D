from typing import NamedTuple

import jax.numpy as jnp

from PyPIC3D.particles.tiled_particles import TiledParticles


class DiagnosticParticleSpecies(NamedTuple):
    name: str
    species_index: int
    x: jnp.ndarray
    x_diagnostic: jnp.ndarray
    u: jnp.ndarray
    charge: jnp.ndarray
    mass: jnp.ndarray
    weight: jnp.ndarray

    def get_name(self):
        return self.name

    def get_number_of_particles(self):
        return self.x.shape[0]

    def get_forward_position(self):
        return self.x[:, 0], self.x[:, 1], self.x[:, 2]

    def get_position(self):
        return self.x_diagnostic[:, 0], self.x_diagnostic[:, 1], self.x_diagnostic[:, 2]

    def get_velocity(self):
        return self.u[:, 0], self.u[:, 1], self.u[:, 2]

    def get_active_mask(self):
        return jnp.ones(self.x.shape[0], dtype=bool)

    def get_charge(self):
        return self.charge * self.weight

    def get_mass(self):
        return self.mass * self.weight

    def get_weight(self):
        return self.weight


def _axis_diagnostic_position(x, u, dt, wind, bc):
    x_diagnostic = x - u * dt / 2

    if int(jnp.asarray(bc).item()) == 0:
        half_wind = wind / 2
        x_diagnostic = jnp.where(
            x_diagnostic > half_wind,
            x_diagnostic - wind,
            jnp.where(x_diagnostic < -half_wind, x_diagnostic + wind, x_diagnostic),
        )

    return x_diagnostic


def _diagnostic_position(x, u, world):
    if world is None:
        return x

    particle_bc = world.get("particle_boundary_conditions", {})
    dt = world["dt"]
    x_diagnostic = _axis_diagnostic_position(x[:, 0], u[:, 0], dt, world["x_wind"], particle_bc.get("x", 0))
    y_diagnostic = _axis_diagnostic_position(x[:, 1], u[:, 1], dt, world["y_wind"], particle_bc.get("y", 0))
    z_diagnostic = _axis_diagnostic_position(x[:, 2], u[:, 2], dt, world["z_wind"], particle_bc.get("z", 0))

    return jnp.stack((x_diagnostic, y_diagnostic, z_diagnostic), axis=-1)


def flatten_tiled_particles_by_species(tiled_particles, species_config=None, species_names=None, world=None):
    """
    Flatten active tile-major particles into diagnostic species arrays.

    This is an output adapter only: it removes inactive slots introduced by the
    fixed-capacity tiled storage and preserves the stored particle position,
    velocity, charge, mass, weight, and species index.
    """

    if not isinstance(tiled_particles, TiledParticles):
        return tiled_particles

    n_species = tiled_particles.active.shape[3]
    flattened_species = []

    for species_index in range(n_species):
        active = tiled_particles.active[:, :, :, species_index, :].reshape(-1)

        x = tiled_particles.x[:, :, :, species_index, :, :].reshape(-1, 3)[active]
        u = tiled_particles.u[:, :, :, species_index, :, :].reshape(-1, 3)[active]
        n_active = int(jnp.sum(active))
        charge = jnp.full((n_active,), species_config.charge[species_index])
        mass = jnp.full((n_active,), species_config.mass[species_index])
        weight = jnp.full((n_active,), species_config.weight[species_index])

        x_diagnostic = _diagnostic_position(x, u, world)

        if species_names is None:
            name = f"species_{species_index}"
        else:
            name = species_names[species_index]

        flattened_species.append(
            DiagnosticParticleSpecies(
                name=name,
                species_index=species_index,
                x=x,
                x_diagnostic=x_diagnostic,
                u=u,
                charge=charge,
                mass=mass,
                weight=weight,
            )
        )

    return flattened_species
