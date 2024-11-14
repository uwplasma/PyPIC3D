import unittest
import numpy as np
from jax import numpy as jnp
from PyPIC3D.spectral import spectral_poisson_solve, spectral_curl, spectralBsolve, spectralEsolve, spectral_laplacian

class TestSpectralMethods(unittest.TestCase):

    def setUp(self):
        # Set up common parameters for the tests
        self.Nx, self.Ny, self.Nz = 16, 16, 16
        self.dx, self.dy, self.dz = 1.0, 1.0, 1.0
        self.dt = 0.1
        self.eps = 1.0
        self.C = 1.0
        self.rho = jnp.ones((self.Nx, self.Ny, self.Nz))
        self.field = jnp.ones((self.Nx, self.Ny, self.Nz))
        self.Ex = jnp.ones((self.Nx, self.Ny, self.Nz))
        self.Ey = jnp.ones((self.Nx, self.Ny, self.Nz))
        self.Ez = jnp.ones((self.Nx, self.Ny, self.Nz))
        self.Bx = jnp.ones((self.Nx, self.Ny, self.Nz))
        self.By = jnp.ones((self.Nx, self.Ny, self.Nz))
        self.Bz = jnp.ones((self.Nx, self.Ny, self.Nz))

    def test_spectral_poisson_solve(self):
        phi = spectral_poisson_solve(self.rho, self.eps, self.dx, self.dy, self.dz)
        self.assertEqual(phi.shape, (self.Nx, self.Ny, self.Nz))
        self.assertTrue(np.all(np.isfinite(phi)))

    def test_spectral_curl(self):
        curlx, curly, curlz = spectral_curl(self.Ex, self.Ey, self.Ez, self.dx, self.dy, self.dz)
        self.assertEqual(curlx.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(curly.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(curlz.shape, (self.Nx, self.Ny, self.Nz))
        self.assertTrue(np.all(np.isfinite(curlx)))
        self.assertTrue(np.all(np.isfinite(curly)))
        self.assertTrue(np.all(np.isfinite(curlz)))

    def test_spectralBsolve(self):
        Bx, By, Bz = spectralBsolve(self.Bx, self.By, self.Bz, self.Ex, self.Ey, self.Ez, self.dx, self.dy, self.dz, self.dt)
        self.assertEqual(Bx.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(By.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(Bz.shape, (self.Nx, self.Ny, self.Nz))
        self.assertTrue(np.all(np.isfinite(Bx)))
        self.assertTrue(np.all(np.isfinite(By)))
        self.assertTrue(np.all(np.isfinite(Bz)))

    def test_spectralEsolve(self):
        Ex, Ey, Ez = spectralEsolve(self.Ex, self.Ey, self.Ez, self.Bx, self.By, self.Bz, self.dx, self.dy, self.dz, self.dt, self.C)
        self.assertEqual(Ex.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(Ey.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(Ez.shape, (self.Nx, self.Ny, self.Nz))
        self.assertTrue(np.all(np.isfinite(Ex)))
        self.assertTrue(np.all(np.isfinite(Ey)))
        self.assertTrue(np.all(np.isfinite(Ez)))

    def test_spectral_laplacian(self):
        laplacian = spectral_laplacian(self.field, self.dx, self.dy, self.dz)
        self.assertEqual(laplacian.shape, (self.Nx, self.Ny, self.Nz))
        self.assertTrue(np.all(np.isfinite(laplacian)))

if __name__ == '__main__':
    unittest.main()