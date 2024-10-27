import unittest
import numpy as np
from src.rho import index_particles, particle_weighting, update_rho

import jax.numpy as jnp

class TestRhoFunctions(unittest.TestCase):

    def test_index_particles(self):
        positions = jnp.array([0.5, 1.5, 2.5, 3.5])
        ds = 1.0
        expected_indices = jnp.array([0, 1, 2, 3])
        for i in range(len(positions)):
            self.assertEqual(index_particles(i, positions, ds), expected_indices[i])

    def test_particle_weighting(self):
        q = 1.0
        x, y, z = 0.5, 0.5, 0.5
        dx, dy, dz = 1.0, 1.0, 1.0
        x_wind, y_wind, z_wind = 1.0, 1.0, 1.0
        rho = jnp.zeros((2, 2, 2))
        expected_rho = jnp.array([[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]])
        updated_rho = particle_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind)
        np.testing.assert_array_almost_equal(updated_rho, expected_rho)

    def test_update_rho(self):
        Nparticles = 2
        particlex = jnp.array([0.5, 1.5])
        particley = jnp.array([0.5, 1.5])
        particlez = jnp.array([0.5, 1.5])
        dx, dy, dz = 1.0, 1.0, 1.0
        q = 1.0
        x_wind, y_wind, z_wind = 1.0, 1.0, 1.0
        rho = jnp.zeros((2, 2, 2))
        expected_rho = jnp.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
        updated_rho = update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, q, x_wind, y_wind, z_wind, rho)
        np.testing.assert_array_almost_equal(updated_rho, expected_rho)

if __name__ == '__main__':
    unittest.main()