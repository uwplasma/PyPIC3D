import unittest

import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_PERIODIC,
)
from tests.kernel_fixtures import build_tiled_particles, particle_parameters_from_tile_values, particle_species
from tests.kernel_fixtures import kernel_parameters_from_values
from PyPIC3D.particles.particle_tile_communication import (
    _adjacent_tile_offset,
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)


class TiledParticleRefreshFixtures:
    tile_width = 4
    def _build_parameter_values(self):
        return {
            "Nx": 4,
            "Ny": 1,
            "Nz": 1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 1.0,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }

    def _species(self, parameter_set, x1, v1, active_mask=None, update_x=True):
        if active_mask is None:
            active_mask = jnp.ones_like(jnp.asarray(x1), dtype=bool)
        n_particles = len(x1)
        return particle_species(
            name="moving",
            charge=2.0,
            mass=3.0,
            weight=4.0,
            x1=jnp.asarray(x1, dtype=float),
            x2=jnp.zeros(n_particles),
            x3=jnp.zeros(n_particles),
            v1=jnp.asarray(v1, dtype=float),
            v2=jnp.zeros(n_particles),
            v3=jnp.zeros(n_particles),
            active_mask=active_mask,
            update_x=update_x,
        )

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": self.tile_width,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

    def _particle_parameters(self, parameter_set):
        return particle_parameters_from_tile_values(parameter_set, self._simulation_parameters())

    def _tiled_particles(self, species, parameter_set):
        static_parameters, dynamic_parameters = self._particle_parameters(parameter_set)
        return build_tiled_particles(species, static_parameters, dynamic_parameters)

    def _split_parameters(self, parameter_set, tile_shape):
        parameter_set = dict(parameter_set)
        parameter_set["tile_shape"] = tuple(int(width) for width in tile_shape)
        return kernel_parameters_from_values(parameter_set)

    def _active_rows(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.argsort(x[:, 0])
        return x[order], u[order]
