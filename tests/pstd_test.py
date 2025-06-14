import unittest
import jax
import jax.numpy as jnp
from PyPIC3D.pstd import spectral_gradient, spectral_poisson_solve, spectral_curl, spectral_laplacian, spectral_divergence

jax.config.update("jax_enable_x64", True)

class TestSpectralMethods(unittest.TestCase):
    def setUp(self):
        # Use a single Fourier mode for all tests
        self.Nx = 16
        self.Ny = 16
        self.Nz = 16
        self.Lx = 2 * jnp.pi
        self.Ly = 2 * jnp.pi
        self.Lz = 2 * jnp.pi
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        x = jnp.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = jnp.linspace(0, self.Ly, self.Ny, endpoint=False)
        z = jnp.linspace(0, self.Lz, self.Nz, endpoint=False)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')
        # build a grid of points in 3D space
        self.phi = jnp.sin(self.X + self.Y + self.Z)
        # use a simple function for phi = sin(X + Y + Z)
        self.constants = {'eps': 1.0}
        self.world = {'dx': self.dx, 'dy': self.dy, 'dz': self.dz}
        # build constants and world parameters
        self.gradx = jnp.cos(self.X + self.Y + self.Z)
        self.grady = jnp.cos(self.X + self.Y + self.Z)
        self.gradz = jnp.cos(self.X + self.Y + self.Z)
        # Analytical derivatives
        self.laplacian = -3 * self.phi
        # Laplacian: -3 * sin(X + Y + Z)
        self.rho = 3 * self.phi  # For Poisson: Laplacian(phi) = -rho, so rho = 3*phi
        self.Ex = self.gradx
        self.Ey = self.grady
        self.Ez = self.gradz
        # For divergence test: E = grad(phi)
        self.Fx = jnp.sin(self.Y)
        self.Fy = jnp.sin(self.Z)
        self.Fz = jnp.sin(self.X)
        self.curlx = jnp.cos(self.Z) - jnp.cos(self.Y)
        self.curly = jnp.cos(self.X) - jnp.cos(self.Z)
        self.curlz = jnp.cos(self.Y) - jnp.cos(self.X)
        # For curl test: F = (sin(Y), sin(Z), sin(X)), curl(F) = (cos(Z)-cos(Y), cos(X)-cos(Z), cos(Y)-cos(X))

    def test_spectral_curl(self):
        # Use a gradient field, whose curl should be zero
        Fx = self.gradx
        Fy = self.grady
        Fz = self.gradz
        curlx, curly, curlz = spectral_curl(Fx, Fy, Fz, self.world)
        maxerrx = jnp.max(jnp.abs(curlx))
        maxerry = jnp.max(jnp.abs(curly))
        maxerrz = jnp.max(jnp.abs(curlz))
        # print('Curl (grad field) max abs error:', float(maxerrx), float(maxerry), float(maxerrz))
        self.assertTrue(jnp.allclose(curlx, 0, atol=1e-10, rtol=1e-8))
        self.assertTrue(jnp.allclose(curly, 0, atol=1e-10, rtol=1e-8))
        self.assertTrue(jnp.allclose(curlz, 0, atol=1e-10, rtol=1e-8))

    def test_spectral_poisson_solve(self):
        phi_num = spectral_poisson_solve(self.rho, self.constants, self.world)
        phi_num = phi_num - jnp.mean(phi_num)
        phi_true = self.phi - jnp.mean(self.phi)
        maxerr = jnp.max(jnp.abs(phi_num - phi_true))
        # calculate the maximum absolute error
        self.assertEqual(phi_num.shape, self.phi.shape)
        self.assertTrue(jnp.allclose(phi_num, phi_true, atol=1e-10, rtol=1e-8))

    def test_spectral_divergence(self):
        divE = spectral_divergence(self.Ex, self.Ey, self.Ez, self.world)
        maxerr = jnp.max(jnp.abs(divE - self.laplacian))
        # calculate the maximum absolute error
        self.assertEqual(divE.shape, self.phi.shape)
        self.assertTrue(jnp.allclose(divE, self.laplacian, atol=1e-10, rtol=1e-8))

    def test_spectral_laplacian(self):
        laplacian_num = spectral_laplacian(self.phi, self.world)
        maxerr = jnp.max(jnp.abs(laplacian_num - self.laplacian))
        # calculate the maximum absolute error
        self.assertEqual(laplacian_num.shape, self.phi.shape)
        self.assertTrue(jnp.allclose(laplacian_num, self.laplacian, atol=1e-10, rtol=1e-8))

    def test_spectral_gradient(self):
        gradx, grady, gradz = spectral_gradient(self.phi, self.world)
        maxerrx = jnp.max(jnp.abs(gradx - self.gradx))
        maxerry = jnp.max(jnp.abs(grady - self.grady))
        maxerrz = jnp.max(jnp.abs(gradz - self.gradz))
        # calculate the maximum absolute error
        self.assertEqual(gradx.shape, self.phi.shape)
        self.assertEqual(grady.shape, self.phi.shape)
        self.assertEqual(gradz.shape, self.phi.shape)
        self.assertTrue(jnp.allclose(gradx, self.gradx, atol=1e-10, rtol=1e-8))
        self.assertTrue(jnp.allclose(grady, self.grady, atol=1e-10, rtol=1e-8))
        self.assertTrue(jnp.allclose(gradz, self.gradz, atol=1e-10, rtol=1e-8))

if __name__ == '__main__':
    unittest.main()