from tests.support.collisions_runtime_fixtures import (
    jnp,
    make_config,
    make_particles,
    run,
    unittest,
)


class TestCollisions(unittest.TestCase):
    def moments(self, particles, config):
        m = config.mass[None, None, None, :, None, None] * particles.active[..., None]
        return jnp.sum(m * particles.u, axis=(0, 1, 2, 3, 4)), jnp.sum(m * particles.u**2)

    def test_binary_conserves_momentum_and_energy(self):
        config = make_config()
        particles = make_particles()
        specs = (("binary", 0, 0, 10.0), ("binary", 0, 1, 10.0), ("binary", 1, 1, 10.0))
        after, _ = run(specs, particles, config, steps=3)
        p0, e0 = self.moments(particles, config)
        p1, e1 = self.moments(after, config)
        self.assertTrue(jnp.allclose(p0, p1, rtol=1e-10, atol=1e-30 + 1e-10 * float(jnp.abs(p0).max())))
        self.assertAlmostEqual(float(e1 / e0), 1.0, places=10)
        self.assertFalse(jnp.allclose(after.u, particles.u))
        self.assertTrue(jnp.array_equal(after.u[..., -7:, :], particles.u[..., -7:, :]))

    def test_mcc_heavy_elastic_preserves_speed_and_charge_exchange_takes_neutral(self):
        config = make_config()
        particles = make_particles(species=1)
        elastic = (("mcc", 0, 1.0e30, 0.0, 1.0e10, (("elastic", 0.0, (0.0, 1e30), (1e-19, 1e-19), -1, -1),)),)
        after, _ = run(elastic, particles, make_config(mass=(9.1e-31,), charge=(-1.6e-19,)))
        speed = lambda p: jnp.linalg.norm(p.u, axis=-1)
        self.assertTrue(jnp.allclose(speed(after), speed(particles), rtol=1e-12))
        self.assertFalse(jnp.allclose(after.u, particles.u))
        cx = (("mcc", 0, 1.0e30, 0.0, 1.0e-26, (("charge_exchange", 0.0, (0.0, 1e30), (1e-19, 1e-19), -1, -1),)),)
        after, _ = run(cx, particles, make_config(mass=(1e-26,), charge=(1.6e-19,)))
        self.assertTrue(jnp.allclose(after.u[after.active], 0.0))


if __name__ == "__main__":
    unittest.main()
