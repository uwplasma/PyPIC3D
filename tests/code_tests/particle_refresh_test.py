import unittest

import jax
import jax.numpy as jnp

from tests.kernel_fixtures import build_tiled_particles, particle_parameters_from_tile_values, particle_species
from tests.kernel_fixtures import kernel_parameters_from_values
from PyPIC3D.particles.particle_tile_communication import (
    _adjacent_tile_offset,
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)


jax.config.update("jax_enable_x64", True)


class TestTiledParticleRefresh(unittest.TestCase):
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
            "particle_tile_nx": 2,
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

    def test_update_tiled_particle_positions_respects_active_and_update_flags(self):
        parameter_set = self._build_parameter_values()
        species = self._species(
            parameter_set,
            x1=[-1.5, -0.5, 0.5],
            v1=[0.25, 0.5, 0.75],
            active_mask=jnp.array([True, False, True]),
        )
        static_parameters, dynamic_parameters = self._particle_parameters(parameter_set)
        tiled_particles, species_config = build_tiled_particles([species], static_parameters, dynamic_parameters)

        moved = update_tiled_particle_positions(tiled_particles, species_config, parameter_set["dt"])

        x, _ = self._active_rows(moved)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-1.25, 1.25])))
        self.assertTrue(jnp.allclose(moved.x[~tiled_particles.active], tiled_particles.x[~tiled_particles.active]))

        fixed_species = self._species(parameter_set, x1=[-1.5], v1=[0.25], update_x=False)
        fixed, fixed_species_config = build_tiled_particles([fixed_species], static_parameters, dynamic_parameters)
        fixed_moved = update_tiled_particle_positions(fixed, fixed_species_config, parameter_set["dt"])
        self.assertTrue(jnp.allclose(fixed_moved.x, fixed.x))

    def test_adjacent_tile_offset_handles_periodic_edges(self):
        source = jnp.array([0, 0, 1, 1])
        dest = jnp.array([0, 1, 0, 1])

        offset = _adjacent_tile_offset(dest, source, tile_count=2)

        self.assertTrue(jnp.array_equal(offset, jnp.array([0, 1, -1, 0])))

    def test_refresh_moves_particles_to_neighbor_tiles_with_static_shape(self):
        parameter_set = self._build_parameter_values()
        species = self._species(parameter_set, x1=[-1.5, -0.25, 0.25], v1=[0.0, 0.0, 0.0])
        tiled_particles, species_config = self._tiled_particles([species], parameter_set)
        moved_x = tiled_particles.x.at[0, 0, 0, 0, 1, 0].set(0.25)
        moved = tiled_particles._replace(x=moved_x)

        refreshed, overflow = refresh_tiled_particle_tiles(moved, *self._split_parameters(parameter_set, (2, 1, 1)))

        self.assertEqual(refreshed.x.shape, tiled_particles.x.shape)
        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active[0, 0, 0, 0])), 1)
        self.assertEqual(int(jnp.sum(refreshed.active[1, 0, 0, 0])), 2)
        x, u = self._active_rows(refreshed)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-1.5, 0.25, 0.25])))
        self.assertTrue(jnp.allclose(u[:, 0], jnp.array([0.0, 0.0, 0.0])))

    def test_refresh_wraps_periodic_particle_to_opposite_tile(self):
        parameter_set = self._build_parameter_values()
        species = self._species(parameter_set, x1=[1.75], v1=[0.0])
        tiled_particles, species_config = self._tiled_particles([species], parameter_set)
        moved = tiled_particles._replace(x=tiled_particles.x.at[1, 0, 0, 0, 0, 0].set(2.25))

        refreshed, overflow = refresh_tiled_particle_tiles(moved, *self._split_parameters(parameter_set, (2, 1, 1)))

        self.assertFalse(bool(overflow))
        self.assertTrue(bool(refreshed.active[0, 0, 0, 0, 0]))
        self.assertTrue(jnp.allclose(refreshed.x[0, 0, 0, 0, 0, 0], -1.75))

    def test_refresh_reflects_particle_from_global_boundary_condition(self):
        parameter_set = self._build_parameter_values()
        parameter_set["particle_boundary_conditions"] = {"x": 1, "y": 0, "z": 0}
        species = self._species(parameter_set, x1=[1.75, -1.75], v1=[0.5, -0.25])
        tiled_particles, species_config = self._tiled_particles([species], parameter_set)
        moved = tiled_particles._replace(
            x=tiled_particles.x
            .at[1, 0, 0, 0, 0, 0].set(2.25)
            .at[0, 0, 0, 0, 0, 0].set(-2.10)
        )

        refreshed, overflow = refresh_tiled_particle_tiles(moved, *self._split_parameters(parameter_set, (2, 1, 1)))

        self.assertFalse(bool(overflow))
        x, u = self._active_rows(refreshed)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-1.90, 1.75])))
        self.assertTrue(jnp.allclose(u[:, 0], jnp.array([0.25, -0.5])))

    def test_refresh_absorbs_particle_from_global_boundary_condition(self):
        parameter_set = self._build_parameter_values()
        parameter_set["particle_boundary_conditions"] = {"x": 2, "y": 0, "z": 0}
        species = self._species(parameter_set, x1=[1.75, -0.25], v1=[0.5, 0.0])
        tiled_particles, species_config = self._tiled_particles([species], parameter_set)
        moved = tiled_particles._replace(x=tiled_particles.x.at[1, 0, 0, 0, 0, 0].set(2.25))

        refreshed, overflow = refresh_tiled_particle_tiles(moved, *self._split_parameters(parameter_set, (2, 1, 1)))

        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active)), 1)
        x, u = self._active_rows(refreshed)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-0.25])))
        self.assertTrue(jnp.allclose(u[:, 0], jnp.array([0.0])))

    def test_refresh_reports_overflow_without_changing_shape(self):
        parameter_set = self._build_parameter_values()
        species = self._species(parameter_set, x1=[-1.5, 0.5, 1.5], v1=[0.0, 0.0, 0.0])
        tiled_particles, species_config = self._tiled_particles([species], parameter_set)
        moved = tiled_particles._replace(x=tiled_particles.x.at[0, 0, 0, 0, 0, 0].set(0.25))

        refreshed, overflow = refresh_tiled_particle_tiles(moved, *self._split_parameters(parameter_set, (2, 1, 1)))

        self.assertEqual(refreshed.x.shape, tiled_particles.x.shape)
        self.assertTrue(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active)), 2)



if __name__ == "__main__":
    unittest.main()
