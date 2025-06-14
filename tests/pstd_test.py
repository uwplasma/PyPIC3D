import unittest
import jax
import jax.numpy as jnp
import sys
import os

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.pstd import spectral_gradient, spectral_poisson_solve, spectral_curl, spectral_laplacian, spectral_divergence

jax.config.update("jax_enable_x64", True)

class TestSpectralMethods(unittest.TestCase):

    def setUp(self):
        # Set up common parameters for the tests

        x = jnp.linspace(0, 1, 10)
        y = jnp.linspace(0, 1, 10)
        z = jnp.linspace(0, 1, 10)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')
        self.phi = self.X**2 + self.Y**2 + self.Z**2
        self.constants = {'eps': 1.0}
        self.rho = -6 * jnp.ones_like(self.phi)  # Laplacian of phi = -6

        self.Nx = 10
        self.Ny = 10
        self.Nz = 10
        self.dx = 1.0/9
        self.dy = 1.0/9
        self.dz = 1.0/9
        self.world = {'dx': self.dx, 'dy': self.dy, 'dz': self.dz}
        self.Ex = -2 * self.X
        self.Ey = -2 * self.Y
        self.Ez = -2 * self.Z


    def test_spectral_poisson_solve(self):
        phi = spectral_poisson_solve(self.rho, self.constants, self.world)
        self.assertEqual(phi.shape, (self.Nx, self.Ny, self.Nz))
        jnp.allclose(phi, self.phi, rtol=1e-12)
        # test against analytical solution

    def test_spectral_divergence(self):
        divE = spectral_divergence(self.Ex, self.Ey, self.Ez, self.world)
        self.assertEqual(divE.shape, (self.Nx, self.Ny, self.Nz))
        jnp.allclose(divE, self.rho, rtol=1e-12)

    def test_spectral_curl(self):
        Fx = self.Y
        Fy = -self.X
        Fz = jnp.zeros_like(self.X)
        curlx, curly, curlz = spectral_curl(Fx, Fy, Fz, self.world)
        self.assertEqual(curlx.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(curly.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(curlz.shape, (self.Nx, self.Ny, self.Nz))
        expected_curlx = jnp.zeros_like(self.X)
        expected_curly = jnp.zeros_like(self.Y)
        expected_curlz = -2 * jnp.ones_like(self.Z)
        jnp.allclose(curlx, expected_curlx, rtol=1e-12)
        jnp.allclose(curly, expected_curly, rtol=1e-12)
        jnp.allclose(curlz, expected_curlz, rtol=1e-12)

    def test_spectral_laplacian(self):
        laplacian = spectral_laplacian(self.phi, self.world)
        self.assertEqual(laplacian.shape, (self.Nx, self.Ny, self.Nz))
        jnp.allclose(laplacian, self.rho, rtol=1e-12)

    def test_spectral_gradient(self):
        gradx, grady, gradz = spectral_gradient(self.phi, self.world)
        self.assertEqual(gradx.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(grady.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(gradz.shape, (self.Nx, self.Ny, self.Nz))
        jnp.allclose(gradx, self.Ex, rtol=1e-12)
        jnp.allclose(grady, self.Ey, rtol=1e-12)
        jnp.allclose(gradz, self.Ez, rtol=1e-12)

if __name__ == '__main__':
    unittest.main()