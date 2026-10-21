"""The shared momentum sampler keeps the velocity API and JAX batching contract."""
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.velocity_distribution import maxwell_juttner, maxwell_juttner_momentum


class TestMaxwellJuttnerMomentum(unittest.TestCase):
    def test_velocity_api_uses_same_momentum_draw(self):
        key = jax.random.PRNGKey(42)
        args = (128, 2., .1, .5, 1., 3., key, 4.)
        u = jnp.stack(maxwell_juttner_momentum(*args), axis=-1)
        velocity = jnp.stack(maxwell_juttner(*args), axis=-1)
        expected = 4*u/jnp.sqrt(1+jnp.sum(u*u, axis=-1, keepdims=True))
        self.assertTrue(bool(jnp.allclose(velocity, expected, rtol=1e-13, atol=1e-14, equal_nan=True)))
        self.assertTrue(bool(jnp.all(jnp.linalg.norm(velocity, axis=-1) < 4)))

    def test_dimensionless_temperature_scaling(self):
        key = jax.random.PRNGKey(11)
        actual = jnp.stack(maxwell_juttner_momentum(128, 2., 4., 4., 4., 3., key, 2.))
        expected = jnp.stack(maxwell_juttner_momentum(128, 1., 1.5, 1.5, 1.5, 1., key))
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)))

    def test_cold_invalid_and_hot_temperatures_under_jit_vmap(self):
        keys = jax.random.split(jax.random.PRNGKey(3), 4)
        def sample(key, temperature):
            return jnp.stack(maxwell_juttner_momentum(
                16, 1., temperature, temperature, temperature, 1., key), axis=-1)
        u = jax.jit(jax.vmap(sample))(keys, jnp.array([0., -1., .5, 1e12]))
        self.assertTrue(bool(jnp.all(u[0] == 0)))
        self.assertTrue(bool(jnp.isnan(u[1]).all()))
        self.assertTrue(bool(jnp.isfinite(u[2:]).all()))
        self.assertGreater(float(jnp.linalg.norm(u[3], axis=-1).mean()), 1e11)
        partly_cold = maxwell_juttner_momentum(16, 1., 0., 1., 1., 1., keys[0])
        self.assertTrue(bool(jnp.isnan(jnp.stack(partly_cold)).all()))

    def test_per_candidate_keys_repeat_under_jit(self):
        keys = jax.random.split(jax.random.PRNGKey(25), 32)
        sample = lambda key: jnp.stack(maxwell_juttner_momentum(1, 1., .5, .5, .5, 1., key))[:, 0]
        actual = jax.jit(jax.vmap(sample))(keys)
        expected = jnp.stack([sample(key) for key in keys])
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)))
        self.assertFalse(bool(jnp.all(actual[0] == actual[1])))


if __name__ == '__main__':
    unittest.main()
