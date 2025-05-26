import unittest
import jax
import jax.numpy as jnp
import sys
import os

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.rho import first_order_weighting, second_order_weighting

jax.config.update("jax_enable_x64", True)

class TestRhoFunctions(unittest.TestCase):

    def test_first_order_weighting(self):
        q = 1.0
        x, y, z = 0.5, 0.5, 0.5
        dx, dy, dz = 1.0, 1.0, 1.0
        x_wind, y_wind, z_wind = 2.0, 2.0, 2.0
        rho = jnp.zeros((2, 2, 2))
        expected_rho = jnp.array([[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]])
        updated_rho = first_order_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind)
        self.assertEqual(updated_rho.shape, expected_rho.shape)
        jnp.allclose(updated_rho, expected_rho, rtol=1e-5)
        total_charge = jnp.sum(updated_rho)
        self.assertAlmostEqual(total_charge, q, places=5)

    def test_second_order_weighting(self):
        q = 1.0
        x, y, z = 0.5, 0.5, 0.5
        dx, dy, dz = 1.0, 1.0, 1.0
        x_wind, y_wind, z_wind = 2.0, 2.0, 2.0
        rho = jnp.zeros((2, 2, 2))
        expected_rho = jnp.array([[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]])
        updated_rho = second_order_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind)
        self.assertEqual(updated_rho.shape, expected_rho.shape)
        jnp.allclose(updated_rho, expected_rho, rtol=1e-5)
        total_charge = jnp.sum(updated_rho)
        self.assertAlmostEqual(total_charge, q, places=5)

if __name__ == '__main__':
    unittest.main()