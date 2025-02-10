import unittest
import jax
import jax.numpy as jnp
import sys
import os

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.boundaryconditions import apply_supergaussian_boundary_condition, apply_zero_boundary_condition

jax.config.update("jax_enable_x64", True)

class TestBoundaryConditions(unittest.TestCase):

    def test_apply_zero_boundary_condition(self):
        key = jax.random.PRNGKey(0)
        field = jax.random.uniform(key, shape=(10, 10, 10))

        result = apply_zero_boundary_condition(field)

        # Check if the boundary values are set to zero
        self.assertTrue(jnp.all(result[0, :, :] == 0))
        self.assertTrue(jnp.all(result[-1, :, :] == 0))
        self.assertTrue(jnp.all(result[:, 0, :] == 0))
        self.assertTrue(jnp.all(result[:, -1, :] == 0))
        self.assertTrue(jnp.all(result[:, :, 0] == 0))
        self.assertTrue(jnp.all(result[:, :, -1] == 0))

if __name__ == '__main__':
    unittest.main()