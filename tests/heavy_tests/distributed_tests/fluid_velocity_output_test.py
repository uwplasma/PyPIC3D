from tests.support.distributed_fluid_velocity_output_fixtures import (
    SpeciesConfig,
    TestTiledParticleDiagnosticVelocityFixtures,
    YeeMetric,
    compute_velocity_field,
    contravariant_three_velocity,
    hybrid_boris_geodesic_push,
    interpolate_metric,
    jax,
    jnp,
    particle_three_velocity,
    safe_inactive_positions,
    unittest,
)


class TestTiledParticleDiagnosticVelocity(TestTiledParticleDiagnosticVelocityFixtures, unittest.TestCase):

    def test_nonuniform_metric_matches_analytic_and_serial_particle_velocities(self):
        velocity = particle_three_velocity(self.particles, self.metric, self.static, self.dynamic)
        x, u, active = self.particles.x, self.particles.u, self.particles.active
        diagonal = jnp.stack((2.0 + 0.03*x[..., 0], 1.2 + 0.02*x[..., 1],
                              1.8 + 0.04*x[..., 2]), axis=-1)
        expected = (u/diagonal) / jnp.sqrt(1.0 + jnp.sum(u*u/diagonal, axis=-1))[..., None]
        expected = jnp.where(active[..., None], expected, 0.0)
        self.assertTrue(bool(jnp.allclose(velocity, expected, atol=2.e-14, rtol=2.e-14)))

        for tile in range(2):
            grid = tuple(a[tile, 0, 0] for a in self.dynamic.grids.tiled_center_grid)
            local_active = active[tile, 0, 0]
            local_x = safe_inactive_positions(x[tile, 0, 0], local_active, grid, (True, True, True), 3)
            sampled = interpolate_metric(jax.tree.map(lambda a: a[tile, 0, 0], self.metric),
                                         local_x, grid, "numerical", (True, True, True),
                                         (3, 3, 3), derivatives=False)
            serial = contravariant_three_velocity(
                jnp.where(local_active[..., None], u[tile, 0, 0], 0.0), sampled.gamma_inv)
            serial = jnp.where(local_active[..., None], serial, 0.0)
            self.assertTrue(bool(jnp.allclose(velocity[tile, 0, 0], serial, atol=2.e-14, rtol=2.e-14)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(velocity))))


    def test_velocity_deposition_keeps_local_fields_after_particle_sampling(self):
        species = SpeciesConfig(jnp.ones(2), jnp.ones(2), jnp.asarray([1.0, 2.0]),
                                jnp.ones((2, 3), dtype=bool))

        def output_velocity(particles, metric):
            particles = particles._replace(u=particle_three_velocity(
                particles, metric, self.static, self.dynamic))
            return tuple(compute_velocity_field(
                particles, metric.lapse, direction, self.static, self.dynamic,
                species_config=species) for direction in range(3))

        executable = jax.jit(output_velocity).lower(self.particles, self.metric).compile()
        output = executable(self.particles, self.metric)
        for component in output:
            self.assertEqual(component.sharding.mesh, self.static.field_mesh)
            self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1)
                                for shard in component.addressable_shards))
            self.assertTrue(bool(jnp.all(jnp.isfinite(component))))
        # Deposition folds and refreshes neighbor halos. It must never exchange
        # complete metric or velocity fields across the device mesh.
        hlo = executable.as_text()
        self.assertIn("collective-permute", hlo)
        for collective in ("all-gather", "all-reduce", "all-to-all"):
            self.assertNotIn(collective, hlo)


    def test_hybrid_push_gathers_only_the_metric_and_fields_on_its_device(self):
        metric = YeeMetric((self.metric,)*3, (self.metric,)*3, self.metric, self.metric)
        D = tuple((i+1)*0.02*self.metric.sqrt_gamma for i in range(3))
        B = tuple((i-1)*0.03*self.metric.sqrt_gamma for i in range(3))
        species = SpeciesConfig(jnp.asarray([-1.0, 1.0]), jnp.asarray([1.0, 2.0]),
                                jnp.ones(2), jnp.asarray([[True, True, True], [True, False, True]]))
        particles = self.particles._replace(active=self.particles.active.at[1, ..., 1, :].set(False))
        static = self.static._replace(particle_batch_size=2, particle_pusher="hybrid_boris_geodesic")
        for advance in (False, True):
            with self.subTest(advance_positions=advance):
                push = jax.jit(lambda p, D, B, metric: hybrid_boris_geodesic_push(
                    p, species, D, B, metric, static, self.dynamic, advance_positions=advance))
                executable = push.lower(particles, D, B, metric).compile()
                full, half = executable(particles, D, B, metric)
                for result in (full, half):
                    self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1)
                                        for shard in result.x.addressable_shards))
                    self.assertTrue(bool(jnp.array_equal(result.active, particles.active)))
                    for old, new in ((particles.x, result.x), (particles.u, result.u)):
                        self.assertTrue(bool(jnp.array_equal(old[~particles.active], new[~particles.active],
                                                            equal_nan=True)))
                        self.assertTrue(bool(jnp.all(jnp.isfinite(new[particles.active]))))
                    if not advance:
                        self.assertTrue(bool(jnp.array_equal(result.x, particles.x, equal_nan=True)))
                for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
                    self.assertNotIn(collective, executable.as_text())


if __name__ == "__main__":
    unittest.main()
