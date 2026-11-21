"""Focused cross-device regression tests."""

from tests.support.particle_refresh_fixtures import (
    BC_ABSORBING,
    BC_PERIODIC,
    TiledParticleRefreshFixtures,
    jnp,
    refresh_tiled_particle_tiles,
    unittest,
)


class TestTiledParticleRefresh(TiledParticleRefreshFixtures, unittest.TestCase):
    tile_width = 2
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


    def test_refresh_wraps_periodic_particle_using_shifted_grid_bounds(self):
        parameter_set = self._build_parameter_values()
        parameter_set["x_wind"] = 2.0
        parameter_set["dx"] = 0.5
        parameter_set["x_min"] = 1.0
        parameter_set["x_max"] = 3.0
        species = self._species(parameter_set, x1=[2.75], v1=[0.0])
        tiled_particles, species_config = self._tiled_particles([species], parameter_set)
        moved = tiled_particles._replace(x=tiled_particles.x.at[1, 0, 0, 0, 0, 0].set(3.25))

        refreshed, overflow = refresh_tiled_particle_tiles(moved, *self._split_parameters(parameter_set, (2, 1, 1)))

        self.assertFalse(bool(overflow))
        self.assertTrue(bool(refreshed.active[0, 0, 0, 0, 0]))
        self.assertTrue(jnp.allclose(refreshed.x[0, 0, 0, 0, 0, 0], 1.25))


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
        parameter_set["particle_boundary_conditions"] = {
            "x": BC_ABSORBING,
            "y": BC_PERIODIC,
            "z": BC_PERIODIC,
        }
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
