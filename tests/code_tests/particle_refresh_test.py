"""Single-device numerical tests."""

from tests.support.particle_refresh_fixtures import (
    TiledParticleRefreshFixtures,
    _adjacent_tile_offset,
    build_tiled_particles,
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


    def test_adjacent_tile_offset_handles_periodic_edges(self):
        source = jnp.array([0, 0, 1, 1])
        dest = jnp.array([0, 1, 0, 1])

        offset = _adjacent_tile_offset(dest, source, tile_count=2)

        self.assertTrue(jnp.array_equal(offset, jnp.array([0, 1, -1, 0])))



if __name__ == "__main__":
    unittest.main()
