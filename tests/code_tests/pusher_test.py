"""Single-device numerical tests."""

from tests.support.pusher_fixtures import (
    SeedLeapfrogVelocityFixtures,
    TiledParticlePusherFixtures,
    build_tiled_particles,
    hybrid_boris_geodesic_push,
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
