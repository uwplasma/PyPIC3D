"""Single-device numerical tests."""

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh

from tests.support.particle_refresh_fixtures import (
    TiledParticleRefreshFixtures,
    build_tiled_particles,
    refresh_tiled_particle_tiles,
    jnp,
    unittest,
    update_tiled_particle_positions,
)


class TestTiledParticleRefresh(TiledParticleRefreshFixtures, unittest.TestCase):
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


    def test_refresh_wraps_particles_on_one_tile_without_changing_momenta(self):
        parameter_set = self._build_parameter_values()
        species = self._species(parameter_set, x1=[-1.75, 1.75], v1=[0.25, -0.5])
        static_parameters, dynamic_parameters = self._particle_parameters(parameter_set)
        static_parameters = static_parameters._replace(field_mesh=make_field_mesh((1, 1, 1)))
        particles, _ = build_tiled_particles([species], static_parameters, dynamic_parameters)
        particles = particles._replace(x=particles.x.at[0, 0, 0, 0, :, 0].set(jnp.array([2.25, -2.25])))

        refreshed, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)

        x, u = self._active_rows(refreshed)
        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active)), 2)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-1.75, 1.75])))
        self.assertTrue(jnp.array_equal(u[:, 0], jnp.array([0.25, -0.5])))



if __name__ == "__main__":
    unittest.main()
