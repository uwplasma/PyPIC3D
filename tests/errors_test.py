import unittest
import jax
import jax.numpy as jnp
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.errors import (
    compute_pe, compute_magnetic_divergence_error, compute_electric_divergence_error
)

jax.config.update("jax_enable_x64", True)

class TestErrors(unittest.TestCase):

    def setUp(self):
        self.constants = {'eps': 1.0}
        self.world = {'dx': 1.0, 'dy': 1.0, 'dz': 1.0}

    def test_compute_pe(self):
        x = jnp.linspace(0, 1, 10)
        y = jnp.linspace(0, 1, 10)
        z = jnp.linspace(0, 1, 10)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

        phi = X**2 + Y**2 + Z**2
        eps = 1.0
        rho = -6 * jnp.ones_like(phi)  # Laplacian of phi = -6
        dx, dy, dz = 1.0 / 9, 1.0 / 9, 1.0 / 9

        laplacian = -6 * jnp.ones_like(phi)

        result_spectral = compute_pe(phi, rho, self.constants, self.world, solver='spectral')
        result_fdtd = compute_pe(phi, rho, self.constants, self.world, solver='fdtd')

        self.assertIsInstance(result_spectral, jnp.ndarray)
        self.assertIsInstance(result_fdtd, jnp.ndarray)

        

    def test_compute_magnetic_divergence_error(self):
        key = jax.random.PRNGKey(0)
        Bx_key, By_key, Bz_key = jax.random.split(key, 3)
        Bx = jax.random.uniform(Bx_key, shape=(10, 10, 10))
        By = jax.random.uniform(By_key, shape=(10, 10, 10))
        Bz = jax.random.uniform(Bz_key, shape=(10, 10, 10))
        dx, dy, dz = 1.0, 1.0, 1.0


        result_spectral = compute_magnetic_divergence_error(Bx, By, Bz, self.world, solver='spectral')
        result_fdtd = compute_magnetic_divergence_error(Bx, By, Bz, self.world, solver='fdtd')

        self.assertIsInstance(result_spectral, jnp.ndarray)
        self.assertIsInstance(result_fdtd, jnp.ndarray)

    def test_compute_electric_divergence_error(self):
        key = jax.random.PRNGKey(1)
        Ex_key, Ey_key, Ez_key, rho_key = jax.random.split(key, 4)

        Ex = jax.random.uniform(Ex_key, shape=(10, 10, 10))
        Ey = jax.random.uniform(Ey_key, shape=(10, 10, 10))
        Ez = jax.random.uniform(Ez_key, shape=(10, 10, 10))
        rho = jax.random.uniform(rho_key, shape=(10, 10, 10))
        eps = 1.0
        dx, dy, dz = 1.0, 1.0, 1.0


        result_spectral = compute_electric_divergence_error(Ex, Ey, Ez, rho, self.constants, self.world, solver='spectral')
        result_fdtd = compute_electric_divergence_error(Ex, Ey, Ez, rho, self.constants, self.world, solver='fdtd')

        self.assertIsInstance(result_spectral, jnp.ndarray)
        self.assertIsInstance(result_fdtd, jnp.ndarray)
if __name__ == '__main__':
    unittest.main()