import unittest
import jax
import jax.numpy as jnp
from PyPIC3D.pstd import spectral_gradient, spectral_poisson_solve, spectral_curl, spectral_laplacian, spectral_divergence
from PyPIC3D.utils import mse, convergence_test

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


    def test_convergence_pstd(self):

        def pstd_laplacian_comparison(nx):
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

            phi = jnp.sin(X + Y + Z)
            expected_solution = -3 * phi
            # define a analytical phi and its expected solution

            laplacian =  spectral_laplacian(phi, world={'dx': dx, 'dy': dy, 'dz': dz})
            # compute the numerical result

            slicer = (slice(1, -1), slice(1, -1), slice(1, -1))
            # slice for the solution comparison

            error = mse( laplacian[slicer], expected_solution[slicer] )
            # compute the mean squared error of the laplacian against the analytical solution

            return error,  dx


        def pstd_gradient_comparison(nx):
            """
            Computes the mean squared error (MSE) between the numerical and analytical gradients
            of a scalar field using spectral methods.

            The function constructs a 3D meshgrid over a symmetric cubic domain, defines a scalar
            field `phi = sin(X + Y + Z)`, and computes its gradient both analytically and numerically.
            The numerical gradient is computed using the `spectral_gradient` function. The MSE between
            the numerical and analytical gradients is calculated for each component and summed.

            Args:
                nx (int): Number of grid points along each spatial dimension.

            Returns:
                tuple:
                    error (float): The total mean squared error between the numerical and analytical gradients.
                    dx (float): The spatial resolution along each axis.
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

            phi = jnp.sin(X + Y + Z)
            expected_gradx = jnp.cos(X + Y + Z)
            expected_grady = jnp.cos(X + Y + Z)
            expected_gradz = jnp.cos(X + Y + Z)
            # define a analytical phi and its expected solution

            gradx, grady, gradz = spectral_gradient(phi, world={'dx': dx, 'dy': dy, 'dz': dz})
            # compute the numerical result

            error_x = mse( gradx, expected_gradx )
            error_y = mse( grady, expected_grady )
            error_z = mse( gradz, expected_gradz )
            error = error_x + error_y + error_z
            # compute the mean squared error of the gradient against the analytical solution

            return error, dx


        def pstd_divergence_comparison(nx):
            """
            Computes the mean squared error between the numerical and analytical divergence of a vector field.

            This function constructs a 3D meshgrid over a symmetric cubic domain, defines a vector field
            with components Fx, Fy, Fz = cos(X + Y + Z), and computes its divergence both analytically and
            numerically (using a spectral method). It then returns the mean squared error between the two
            divergence calculations, along with the spatial resolution.

            Args:
                nx (int): Number of grid points along each axis.

            Returns:
                tuple:
                    error (float): Mean squared error between the numerical and analytical divergence.
                    dx (float): Spatial resolution along each axis.
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

            Fx = jnp.cos(X + Y + Z)
            Fy = jnp.cos(X + Y + Z)
            Fz = jnp.cos(X + Y + Z)
            expected_div = -3 * jnp.sin(X + Y + Z)
            # define a analytical phi and its expected solution

            divF = spectral_divergence(Fx, Fy, Fz, world={'dx': dx, 'dy': dy, 'dz': dz})
            # compute the numerical result

            error = mse( divF, expected_div )
            # compute the mean squared error of the divergence against the analytical solution

            return error, dx

        def pstd_curl_comparison(nx):
            """
            Computes the mean squared error (MSE) between the numerical and analytical curl of a vector field
            defined on a 3D grid using spectral methods.

            The vector field is defined as F = (sin(Y), sin(Z), sin(X)), and its analytical curl is
            (cos(Z) - cos(Y), cos(X) - cos(Z), cos(Y) - cos(X)). The function generates a symmetric 3D grid,
            computes the numerical curl using the `spectral_curl` function, and returns the total MSE along
            with the grid spacing.

            Args:
                nx (int): Number of grid points along each spatial dimension.

            Returns:
                tuple:
                    error (float): The sum of mean squared errors for each component of the curl.
                    dx (float): The spatial resolution (grid spacing) in the x-direction.
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

            Fx = jnp.sin(Y)
            Fy = jnp.sin(Z)
            Fz = jnp.sin(X)
            expected_curlx = jnp.cos(Z) - jnp.cos(Y)
            expected_curly = jnp.cos(X) - jnp.cos(Z)
            expected_curlz = jnp.cos(Y) - jnp.cos(X)
            # For curl test: F = (sin(Y), sin(Z), sin(X)), curl(F) = (cos(Z)-cos(Y), cos(X)-cos(Z), cos(Y)-cos(X))
            # define a analytical phi and its expected solution

            curlx, curly, curlz = spectral_curl(Fx, Fy, Fz, world={'dx': dx, 'dy': dy, 'dz': dz})
            # compute the numerical result

            error_x = mse( curlx, expected_curlx )
            error_y = mse( curly, expected_curly )
            error_z = mse( curlz, expected_curlz )
            error = error_x + error_y + error_z
            # compute the mean squared error of the curl against the analytical solution

            return error, dx

        order = convergence_test(pstd_laplacian_comparison)
        self.assertTrue(jnp.isclose(order, 3, rtol=2.5e-2, atol=2.5e-2))
        # compute order of pstd laplacian

        order = convergence_test(pstd_gradient_comparison)
        self.assertTrue(jnp.isclose(order, 1, rtol=1.5e-2, atol=1.5e-2))
        # compute order of pstd gradient

        order = convergence_test(pstd_divergence_comparison)
        self.assertTrue(jnp.isclose(order, 1, rtol=2.5e-2, atol=2.5e-2))
        # compute order of pstd divergence

        order = convergence_test(pstd_curl_comparison)
        self.assertTrue(jnp.isclose(order, 1, rtol=1.5e-2, atol=1.5e-2))
        # compute order of pstd curl

if __name__ == '__main__':
    unittest.main()