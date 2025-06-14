import unittest
import jax
import jax.numpy as jnp

from PyPIC3D.fdtd import (
    centered_finite_difference_curl, centered_finite_difference_laplacian,
    centered_finite_difference_gradient, centered_finite_difference_divergence
)

jax.config.update("jax_enable_x64", True)

class TestFDTDMethods(unittest.TestCase):
    def setUp(self):
        x = jnp.linspace(0, 1, 10)
        y = jnp.linspace(0, 1, 10)
        z = jnp.linspace(0, 1, 10)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')
        self.Nx, self.Ny, self.Nz = 10, 10, 10
        self.dx = 1.0/9
        self.dy = 1.0/9
        self.dz = 1.0/9
        self.bc = 'periodic'
        self.slicer = (slice(1, -1), slice(1, -1), slice(1, -1))

    def test_centered_finite_difference_laplacian(self):
        # Scalar field: phi = X**2 + Y**2 + Z**2, Laplacian = 2 + 2 + 2 = 6
        phi = self.X**2 + self.Y**2 + self.Z**2
        expected = 6 * jnp.ones_like(phi)
        laplacian = centered_finite_difference_laplacian(phi, self.dx, self.dy, self.dz, self.bc)
        self.assertEqual(laplacian.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(laplacian.dtype, phi.dtype)
        self.assertTrue(jnp.allclose(laplacian[self.slicer], expected[self.slicer], rtol=1e-2, atol=1e-2))

    def test_centered_finite_difference_gradient(self):
        # Scalar field: phi = X**2 + Y**2 + Z**2, grad = (2X, 2Y, 2Z)
        phi = self.X**2 + self.Y**2 + self.Z**2
        expected_gradx = 2 * self.X
        expected_grady = 2 * self.Y
        expected_gradz = 2 * self.Z
        gradx, grady, gradz = centered_finite_difference_gradient(phi, self.dx, self.dy, self.dz, self.bc)
        self.assertEqual(gradx.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(grady.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(gradz.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(gradx.dtype, phi.dtype)
        self.assertTrue(jnp.allclose(gradx[self.slicer], expected_gradx[self.slicer], rtol=1e-2, atol=1e-2))
        self.assertTrue(jnp.allclose(grady[self.slicer], expected_grady[self.slicer], rtol=1e-2, atol=1e-2))
        self.assertTrue(jnp.allclose(gradz[self.slicer], expected_gradz[self.slicer], rtol=1e-2, atol=1e-2))

    def test_centered_finite_difference_divergence(self):
        # Vector field: F = (X, Y, Z), div F = 1 + 1 + 1 = 3
        Fx = self.X
        Fy = self.Y
        Fz = self.Z
        expected = 3 * jnp.ones_like(self.X)
        divF = centered_finite_difference_divergence(Fx, Fy, Fz, self.dx, self.dy, self.dz, self.bc)
        self.assertEqual(divF.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(divF.dtype, Fx.dtype)
        self.assertTrue(jnp.allclose(divF[self.slicer], expected[self.slicer], rtol=1e-2, atol=1e-2))

    def test_centered_finite_difference_curl(self):
        # Vector field: F = (-Y, X, 0), curl F = (0, 0, 2)
        Fx = -self.Y
        Fy = self.X
        Fz = jnp.zeros_like(self.X)
        expected_curlx = jnp.zeros_like(self.X)
        expected_curly = jnp.zeros_like(self.Y)
        expected_curlz = 2 * jnp.ones_like(self.Z)
        curlx, curly, curlz = centered_finite_difference_curl(Fx, Fy, Fz, self.dx, self.dy, self.dz, self.bc)
        self.assertEqual(curlx.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(curly.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(curlz.shape, (self.Nx, self.Ny, self.Nz))
        self.assertEqual(curlx.dtype, Fx.dtype)
        self.assertTrue(jnp.allclose(curlx[self.slicer], expected_curlx[self.slicer], rtol=1e-2, atol=1e-2))
        self.assertTrue(jnp.allclose(curly[self.slicer], expected_curly[self.slicer], rtol=1e-2, atol=1e-2))
        self.assertTrue(jnp.allclose(curlz[self.slicer], expected_curlz[self.slicer], rtol=1e-2, atol=1e-2))

if __name__ == '__main__':
    unittest.main()