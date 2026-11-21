"""One-tile numerical and physical checks."""

from tests.support.pusher_fixtures import (
    SeedLeapfrogVelocityFixtures,
    hybrid_boris_geodesic_push,
    np,
    particle_push,
    seed_leapfrog_velocity,
    unittest,
)


class TestSeedLeapfrogVelocity(SeedLeapfrogVelocityFixtures, unittest.TestCase):
    def test_flat_seed_matches_a_backward_half_step_of_the_production_pusher(self):
        static_parameters, dynamic_parameters, E, B = self._flat_case()
        particles = self._one_particle((0.10, -0.05, 0.02))
        species = self._species()

        seeded = seed_leapfrog_velocity(
            particles, species, E, B, static_parameters, dynamic_parameters
        )
        expected = particle_push(
            particles,
            species,
            E,
            B,
            static_parameters,
            dynamic_parameters._replace(dt=-0.5 * self.DT),
        )

        np.testing.assert_allclose(
            np.asarray(seeded.u), np.asarray(expected.u), rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(np.asarray(seeded.x), np.asarray(particles.x))
        np.testing.assert_array_equal(
            np.asarray(seeded.active), np.asarray(particles.active)
        )

    def test_flat_seed_moves_the_velocity_backwards_along_the_electric_force(self):
        static_parameters, dynamic_parameters, E, B = self._flat_case()
        charge, mass = 1.0, 2.0
        particles = self._one_particle((0.0, 0.0, 0.0))

        seeded = seed_leapfrog_velocity(
            particles,
            self._species(charge=charge, mass=mass),
            E,
            B,
            static_parameters,
            dynamic_parameters,
        )

        # a particle at rest feels only the electric kick over -dt/2
        expected_vx = -(charge / mass) * 0.30 * 0.5 * self.DT
        self.assertAlmostEqual(
            float(seeded.u[0, 0, 0, 0, 0, 0]), expected_vx, delta=abs(expected_vx) * 1.0e-3
        )

    def test_gr_seed_keeps_the_position_while_offsetting_the_velocity(self):
        static_parameters, dynamic_parameters, D, B, metric = self._gr_case()
        particles = self._one_particle((0.10, -0.05, 0.02))
        species = self._species()

        seeded = seed_leapfrog_velocity(
            particles, species, D, B, static_parameters, dynamic_parameters, metric=metric
        )
        stepped, _centered = hybrid_boris_geodesic_push(
            particles,
            species,
            D,
            B,
            metric,
            static_parameters,
            dynamic_parameters._replace(dt=-0.5 * self.DT),
        )

        np.testing.assert_allclose(
            np.asarray(seeded.u), np.asarray(stepped.u), rtol=0.0, atol=0.0
        )
        # the geodesic push also advances x, so the seed has to discard that
        self.assertFalse(np.allclose(np.asarray(stepped.x), np.asarray(particles.x)))
        np.testing.assert_array_equal(np.asarray(seeded.x), np.asarray(particles.x))


if __name__ == "__main__":
    unittest.main()
