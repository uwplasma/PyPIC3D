import unittest
from unittest import mock
import numpy as np
import jax.numpy as jnp
from errors import compute_pe, compute_magnetic_divergence_error, compute_electric_divergence_error

class TestErrors(unittest.TestCase):

    @mock.patch('errors.spectral_laplacian')
    @mock.patch('errors.centered_finite_difference_laplacian')
    def test_compute_pe(self, mock_fdtd_laplacian, mock_spectral_laplacian):
        phi = np.random.rand(10, 10, 10)
        rho = np.random.rand(10, 10, 10)
        eps = 1.0
        dx, dy, dz = 1.0, 1.0, 1.0

        mock_spectral_laplacian.return_value = np.random.rand(10, 10, 10)
        mock_fdtd_laplacian.return_value = np.random.rand(10, 10, 10)

        result_spectral = compute_pe(phi, rho, eps, dx, dy, dz, solver='spectral')
        result_fdtd = compute_pe(phi, rho, eps, dx, dy, dz, solver='fdtd')
        result_autodiff = compute_pe(phi, rho, eps, dx, dy, dz, solver='autodiff')

        self.assertIsInstance(result_spectral, float)
        self.assertIsInstance(result_fdtd, float)
        self.assertEqual(result_autodiff, 0)

    @mock.patch('errors.spectral_divergence')
    @mock.patch('errors.centered_finite_difference_divergence')
    def test_compute_magnetic_divergence_error(self, mock_fdtd_divergence, mock_spectral_divergence):
        Bx = np.random.rand(10, 10, 10)
        By = np.random.rand(10, 10, 10)
        Bz = np.random.rand(10, 10, 10)
        dx, dy, dz = 1.0, 1.0, 1.0

        mock_spectral_divergence.return_value = np.random.rand(10, 10, 10)
        mock_fdtd_divergence.return_value = np.random.rand(10, 10, 10)

        result_spectral = compute_magnetic_divergence_error(Bx, By, Bz, dx, dy, dz, solver='spectral')
        result_fdtd = compute_magnetic_divergence_error(Bx, By, Bz, dx, dy, dz, solver='fdtd')
        result_autodiff = compute_magnetic_divergence_error(Bx, By, Bz, dx, dy, dz, solver='autodiff')

        self.assertIsInstance(result_spectral, float)
        self.assertIsInstance(result_fdtd, float)
        self.assertEqual(result_autodiff, 0)

    @mock.patch('errors.spectral_divergence')
    @mock.patch('errors.centered_finite_difference_divergence')
    def test_compute_electric_divergence_error(self, mock_fdtd_divergence, mock_spectral_divergence):
        Ex = np.random.rand(10, 10, 10)
        Ey = np.random.rand(10, 10, 10)
        Ez = np.random.rand(10, 10, 10)
        rho = np.random.rand(10, 10, 10)
        eps = 1.0
        dx, dy, dz = 1.0, 1.0, 1.0

        mock_spectral_divergence.return_value = np.random.rand(10, 10, 10)
        mock_fdtd_divergence.return_value = np.random.rand(10, 10, 10)

        result_spectral = compute_electric_divergence_error(Ex, Ey, Ez, rho, eps, dx, dy, dz, solver='spectral')
        result_fdtd = compute_electric_divergence_error(Ex, Ey, Ez, rho, eps, dx, dy, dz, solver='fdtd')
        result_autodiff = compute_electric_divergence_error(Ex, Ey, Ez, rho, eps, dx, dy, dz, solver='autodiff')

        self.assertIsInstance(result_spectral, float)
        self.assertIsInstance(result_fdtd, float)
        self.assertEqual(result_autodiff, 0)

if __name__ == '__main__':
    unittest.main()