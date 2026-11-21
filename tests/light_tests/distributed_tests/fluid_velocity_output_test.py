from tests.support.distributed_fluid_velocity_output_fixtures import (
    TestTiledParticleDiagnosticVelocityFixtures,
    YeeMetric,
    jax,
    jnp,
    particle_lorentz_factor,
    particle_three_velocity,
    unittest,
)


class TestTiledParticleDiagnosticVelocity(TestTiledParticleDiagnosticVelocityFixtures, unittest.TestCase):


    def test_compiled_sampling_has_no_global_metric_communication(self):
        sample = jax.jit(lambda particles, metric: particle_three_velocity(
            particles, metric, self.static, self.dynamic))
        executable = sample.lower(self.particles, self.metric).compile()
        velocity = executable(self.particles, self.metric)
        self.assertEqual(velocity.sharding.mesh, self.static.field_mesh)
        self.assertEqual(velocity.sharding.spec[:3], ("tile_x", "tile_y", "tile_z"))
        self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1) for shard in velocity.addressable_shards))
        hlo = executable.as_text()
        for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
            self.assertNotIn(collective, hlo)


    def test_lorentz_factor_uses_the_local_nonuniform_metric(self):
        metric = YeeMetric((self.metric,)*3, (self.metric,)*3, self.metric, self.metric)
        sample = jax.jit(lambda particles, metric: particle_lorentz_factor(
            particles, metric, self.static, self.dynamic))
        executable = sample.lower(self.particles, metric).compile()
        gamma = executable(self.particles, metric)
        x, u, active = self.particles.x, self.particles.u, self.particles.active
        diagonal = jnp.stack((2.0 + 0.03*x[..., 0], 1.2 + 0.02*x[..., 1],
                              1.8 + 0.04*x[..., 2]), axis=-1)
        expected = jnp.where(active, jnp.sqrt(1.0 + jnp.sum(u*u/diagonal, axis=-1)), 1.0)
        self.assertTrue(bool(jnp.allclose(gamma, expected, atol=2.e-14, rtol=2.e-14)))
        self.assertTrue(all(shard.data.shape[:3] == (1, 1, 1) for shard in gamma.addressable_shards))
        for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
            self.assertNotIn(collective, executable.as_text())


if __name__ == "__main__":
    unittest.main()
