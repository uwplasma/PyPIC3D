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


class TestSeedLeapfrogVelocity(SeedLeapfrogVelocityFixtures, unittest.TestCase):
    def test_gr_seed_requires_a_metric(self):
        static_parameters, dynamic_parameters, D, B, _metric = self._gr_case()
        with self.assertRaises(ValueError):
            seed_leapfrog_velocity(
                self._one_particle((0.1, 0.0, 0.0)),
                self._species(),
                D,
                B,
                static_parameters,
                dynamic_parameters,
            )

    def test_seed_respects_the_per_species_direction_mask(self):
        for solver in ("electrodynamic_yee", "static_metric"):
            with self.subTest(solver=solver):
                if solver == "static_metric":
                    sp, dp, E, B, metric = self._gr_case()
                else:
                    sp, dp, E, B = self._flat_case()
                    metric = None
                E = self._uniform(sp, dp, (0.30, 0.30, 0.30))
                particles = self._one_particle((0.0, 0.0, 0.0))
                seeded = seed_leapfrog_velocity(
                    particles,
                    self._species(update_x=(True, False, True)),
                    E,
                    B,
                    sp,
                    dp,
                    metric=metric,
                )
                u = np.asarray(seeded.u)[0, 0, 0, 0, 0]
                self.assertNotEqual(u[0], 0.0)
                self.assertEqual(u[1], 0.0)
                self.assertNotEqual(u[2], 0.0)


if __name__ == "__main__":
    unittest.main()
