"""One-tile API contracts."""

from tests.support.pusher_fixtures import (
    SeedLeapfrogVelocityFixtures,
    TiledParticlePusherFixtures,
    build_tiled_particles,
    jnp,
    kernel_parameters_from_values,
    np,
    particle_parameters_from_tile_values,
    particle_push,
    seed_leapfrog_velocity,
    tile_vector_field,
    unittest,
    update_tiled_particle_positions,
)


class TestTiledParticlePusher(TiledParticlePusherFixtures, unittest.TestCase):
    def test_particle_push_respects_active_and_update_flags(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 10.0}
        tile_shape = (8, 6, 4)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        species = self._species(
            parameter_set,
            active_mask=jnp.array([True, False, True, True]),
            update_x=False,
            update_y=True,
            update_z=False,
        )
        particle_static, particle_dynamic = particle_parameters_from_tile_values(
            parameter_set,
            simulation_parameters,
            dynamic_values=dynamic_values,
        )
        tiled_particles, species_config = build_tiled_particles([species], particle_static, particle_dynamic)

        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(relativistic=False, particle_pusher="boris")
        pushed = particle_push(
            tiled_particles,
            species_config,
            tile_vector_field(E, parameter_set, tile_shape, num_guard_cells=1),
            tile_vector_field(B, parameter_set, tile_shape, num_guard_cells=1),
            static_parameters,
            dynamic_parameters,
        )

        self.assertTrue(jnp.allclose(pushed.u[..., 0], tiled_particles.u[..., 0]))
        self.assertTrue(jnp.allclose(pushed.u[..., 2], tiled_particles.u[..., 2]))
        self.assertTrue(jnp.allclose(pushed.u[..., 1][~tiled_particles.active], tiled_particles.u[..., 1][~tiled_particles.active]))
        self.assertFalse(jnp.allclose(pushed.u[..., 1][tiled_particles.active], tiled_particles.u[..., 1][tiled_particles.active]))

        moved = update_tiled_particle_positions(
            pushed,
            species_config,
            dynamic_parameters.dt,
        )
        self.assertTrue(jnp.allclose(moved.x[..., 0], pushed.x[..., 0]))
        self.assertTrue(jnp.allclose(moved.x[..., 2], pushed.x[..., 2]))
        self.assertFalse(jnp.allclose(moved.x[..., 1][pushed.active], pushed.x[..., 1][pushed.active]))


if __name__ == "__main__":
    unittest.main()
