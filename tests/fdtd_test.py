import unittest
import jax
import jax.numpy as jnp

from PyPIC3D.fdtd import (
    centered_finite_difference_curl, centered_finite_difference_laplacian,
    centered_finite_difference_gradient, centered_finite_difference_divergence
)

from PyPIC3D.utils import convergence_test, mse

jax.config.update("jax_enable_x64", True)

class TestFDTDMethods(unittest.TestCase):
    def setUp(self):
        x = jnp.linspace(0, 1, 120)
        y = jnp.linspace(0, 1, 120)
        z = jnp.linspace(0, 1, 120)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')
        self.Nx, self.Ny, self.Nz = 120, 120, 120
        self.dx = 1.0/119
        self.dy = 1.0/119
        self.dz = 1.0/119
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
        # Scalar field: phi = sin(2 * pi * X) + sin(2 * pi * Y) + sin(2 * pi * Z)
        # Gradient: grad(phi) = (2 * pi * cos(2 * pi * X), 2 * pi * cos(2 * pi * Y), 2 * pi * cos(2 * pi * Z))
        # Expected values at interior points
        phi = jnp.sin(2 * jnp.pi * self.X) + jnp.sin(2 * jnp.pi * self.Y) + jnp.sin(2 * jnp.pi * self.Z)
        expected_gradx = 2 * jnp.pi * jnp.cos(2 * jnp.pi * self.X)
        expected_grady = 2 * jnp.pi * jnp.cos(2 * jnp.pi * self.Y)
        expected_gradz = 2 * jnp.pi * jnp.cos(2 * jnp.pi * self.Z)
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
        self.assertTrue(jnp.allclose(curlx[self.slicer], expected_curlx[self.slicer], rtol=1e-4, atol=1e-4))
        self.assertTrue(jnp.allclose(curly[self.slicer], expected_curly[self.slicer], rtol=1e-4, atol=1e-4))
        self.assertTrue(jnp.allclose(curlz[self.slicer], expected_curlz[self.slicer], rtol=1e-4, atol=1e-4))


    def test_convergence_fdtd(self):

        def fdtd_laplacian_comparison(nx):
            """
            Computes the mean squared error between the numerical and analytical Laplacian of a test function
            on a 3D periodic grid.

            The test function is phi(x, y, z) = x^2 + y^2 + z^2, whose Laplacian is analytically 6 everywhere.
            The function constructs a cubic grid of size `nx` in each dimension, computes the numerical Laplacian
            using a centered finite difference scheme with periodic boundary conditions, and compares it to the
            analytical solution (6) in the interior of the domain.

            Parameters
            ----------
            nx : int
                Number of grid points in each spatial dimension.

            Returns
            -------
            error : float
                Mean squared error between the numerical and analytical Laplacian in the interior of the grid.
            dx : float
                Grid spacing in the x-direction (equal to y and z spacing).
            """
            x_wind = 0.25
            y_wind = 0.25
            z_wind = 0.25
            # symmetric world volume

            x = jnp.linspace(-x_wind/2, x_wind/2, nx)
            y = jnp.linspace(-y_wind/2, y_wind/2, nx)
            z = jnp.linspace(-z_wind/2, z_wind/2, nx)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            # build x, y, z meshgrid

            dx = x_wind / nx
            dy = y_wind / nx
            dz = z_wind / nx
            # spatial resolution

            phi = X**2 + Y**2 + Z**2
            expected_solution = 6 * jnp.ones_like(phi)
            # define a analytical phi and its expected solution

            laplacian =  centered_finite_difference_laplacian(phi, dx, dy, dz, 'periodic')
            # compute the numerical result

            slicer = (slice(1, -1), slice(1, -1), slice(1, -1))
            # slice for the solution comparison

            error = mse( laplacian[slicer], expected_solution[slicer] )
            # compute the mean squared error of the laplacian against the analytical solution

            return error,  dx

        def fdtd_gradient_comparison(nx):
            """
            Computes the mean squared error between the analytical and numerical gradients
            of a 3D scalar field using centered finite differences with periodic boundary conditions.

            The scalar field is defined as:
                phi = sin(2πX) + sin(2πY) + sin(2πZ)
            over a symmetric cubic domain of size 0.25 in each direction, discretized into `nx` points per axis.

            Parameters
            ----------
            nx : int
                Number of grid points along each spatial dimension.

            Returns
            -------
            error : float
                The sum of mean squared errors between the numerical and analytical gradients
                in the x, y, and z directions.
            dx : float
                The spatial grid spacing along each axis.
            """
            x_wind = 0.25
            y_wind = 0.25
            z_wind = 0.25
            # symmetric world volume

            x = jnp.linspace(-x_wind/2, x_wind/2, nx)
            y = jnp.linspace(-y_wind/2, y_wind/2, nx)
            z = jnp.linspace(-z_wind/2, z_wind/2, nx)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            # build x, y, z meshgrid

            dx = x_wind / nx
            dy = y_wind / nx
            dz = z_wind / nx
            # spatial resolution

            phi = jnp.sin(2 * jnp.pi * X) + jnp.sin(2 * jnp.pi * Y) + jnp.sin(2 * jnp.pi * Z)
            expected_gradx = 2 * jnp.pi * jnp.cos(2 * jnp.pi * X)
            expected_grady = 2 * jnp.pi * jnp.cos(2 * jnp.pi * Y)
            expected_gradz = 2 * jnp.pi * jnp.cos(2 * jnp.pi * Z)
            # build analytical phi with known solution

            gradx, grady, gradz = centered_finite_difference_gradient(phi, dx, dy, dz, 'periodic')
            # compute the numerical result

            x_error = mse(gradx, expected_gradx)
            y_error = mse(grady, expected_grady)
            z_error = mse(gradz, expected_gradz)
            error = x_error + y_error + z_error
            # compute the mean squared error of the gradient function

            return error, dx


        def fdtd_divergence_comparison(nx):
            """
            Computes the mean squared error between the numerical and analytical divergence
            of the vector field F = (X, Y, Z) on a 3D periodic grid.

            The function constructs a symmetric 3D grid of size `nx` in each dimension,
            defines the vector field F = (X, Y, Z), and computes its divergence using a
            centered finite difference scheme with periodic boundary conditions. The analytical
            divergence of F is 3 everywhere. The function returns the mean squared error
            between the computed and expected divergence, along with the grid spacing `dx`.

            Args:
                nx (int): Number of grid points in each spatial dimension.

            Returns:
                tuple:
                    error (float): Mean squared error between numerical and analytical divergence.
                    dx (float): Grid spacing in the x-direction.
            """
            x_wind = 0.25
            y_wind = 0.25
            z_wind = 0.25
            # symmetric world volume

            x = jnp.linspace(-x_wind/2, x_wind/2, nx)
            y = jnp.linspace(-y_wind/2, y_wind/2, nx)
            z = jnp.linspace(-z_wind/2, z_wind/2, nx)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            # build x, y, z meshgrid

            dx = x_wind / nx
            dy = y_wind / nx
            dz = z_wind / nx
            # spatial resolution

            Fx = X
            Fy = Y
            Fz = Z
            # F = (X, Y, Z), div F = 3
            expected_divergence = 3 * jnp.ones_like(X)

            divergence = centered_finite_difference_divergence(Fx, Fy, Fz, dx, dy, dz, 'periodic')
            # compute the numerical divergence

            error = mse(divergence, expected_divergence)
            # compute the mean squared error of the divergence

            return error, dx


        def fdtd_curl_comparison(nx):
            """
            Computes the mean squared error between the analytical and numerical curl of a test vector field
            on a 3D periodic grid, as a function of grid resolution.

            The test vector field is F = (-Y, X, 0), whose analytical curl is (0, 0, 2).
            The numerical curl is computed using a centered finite difference scheme with periodic boundaries.

            Args:
                nx (int): Number of grid points in each spatial dimension.

            Returns:
                tuple:
                    error (float): The total mean squared error between the analytical and numerical curl components.
                    dx (float): The grid spacing in each dimension.
            """

            x_wind = 0.25
            y_wind = 0.25
            z_wind = 0.25
            # symmetric world volume

            x = jnp.linspace(-x_wind/2, x_wind/2, nx)
            y = jnp.linspace(-y_wind/2, y_wind/2, nx)
            z = jnp.linspace(-z_wind/2, z_wind/2, nx)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            # build x, y, z meshgrid

            dx = x_wind / nx
            dy = y_wind / nx
            dz = z_wind / nx
            # spatial resolution

            Fx = -1 * Y
            Fy = X
            Fz = jnp.zeros_like(Z)
            # Vector field: F = (-Y, X, 0), curl F = (0, 0, 2)

            expected_curlx = jnp.zeros_like(X)
            expected_curly = jnp.zeros_like(Y)
            expected_curlz = 2 * jnp.ones_like(Z)
            # compute the analytical curl

            curlx, curly, curlz = centered_finite_difference_curl(Fx, Fy, Fz, dx, dy, dz, 'periodic')
            # compute the numerical curl

            error_x = mse(curlx, expected_curlx)
            error_y = mse(curly, expected_curly)
            error_z = mse(curlz, expected_curlz)
            error = error_x + error_y + error_z + error_z
            # compute the error in the numerical curl

            return error, dx

        order = convergence_test(fdtd_laplacian_comparison)
        self.assertTrue(jnp.isclose(order, 2, rtol=1.5e-2, atol=1.5e-2))
        # compute order of fdtd laplacian

        order = convergence_test(fdtd_gradient_comparison)
        self.assertTrue(jnp.isclose(order, 1, rtol=1.5e-2, atol=1.5e-2))
        # compute order of fdtd gradient

        order = convergence_test(fdtd_divergence_comparison)
        self.assertTrue(jnp.isclose(order, 1, rtol=6.5e-2, atol=6.5e-2))
        # compute order of fdtd divergence

        order = convergence_test(fdtd_curl_comparison)
        self.assertTrue(jnp.isclose(order, 1, rtol=2.5e-2, atol=2.5e-2))
        # compute order of fdtd curl

if __name__ == '__main__':
    unittest.main()