"""The shared momentum sampler keeps the velocity API and JAX batching contract."""
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.velocity_distribution import (
    maxwell_juttner_momentum,
)


class TestMaxwellJuttnerMomentum(unittest.TestCase):
    def test_dimensionless_temperature_scaling(self):
        key = jax.random.PRNGKey(11)
        actual = jnp.stack(maxwell_juttner_momentum(128, 2., 4., 4., 4., 3., key, 2.))
        expected = jnp.stack(maxwell_juttner_momentum(128, 1., 1.5, 1.5, 1.5, 1., key))
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)))


if __name__ == "__main__":
    unittest.main()
