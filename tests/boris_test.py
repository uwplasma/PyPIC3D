import unittest
import jax
import jax.numpy as jnp
import sys
import os
from functools import partial


from PyPIC3D.boris import boris_single_particle
from PyPIC3D.boris import create_trilinear_interpolator, create_quadratic_interpolator
from PyPIC3D.utils import mae, convergence_test

jax.config.update("jax_enable_x64", True)

class TestBorisMethods(unittest.TestCase):

    def setUp(self):
        self.q = 1.0
        self.m = 1.0
        self.x = jnp.array([0.0])
        self.y = jnp.array([0.0])
        self.z = jnp.array([0.0])
        # single particle at the origin

        self.vx = jnp.array([1.0])
        self.vy = jnp.array([0.0])
        self.vz = jnp.array([0.0])
        # single particle moving in the x direction

        self.Ex = jnp.zeros( (5, 5, 5) )
        self.Ey = jnp.zeros( (5, 5, 5) )
        self.Ez = jnp.zeros( (5, 5, 5) )
        self.Bx = jnp.zeros( (5, 5, 5) )
        self.By = jnp.zeros( (5, 5, 5) )
        self.Bz = jnp.ones( (5, 5, 5) )
        # grid of 5x5x5 with a uniform magnetic field in the z direction

        self.grid = jnp.arange(-1/2, 1/2, (1/5)), jnp.arange(-1/2, 1/2, (1/5)), jnp.arange(-1/2, 1/2, (1/5))
        self.staggered_grid = jnp.arange(-1/2 + (1/5)/2, 1/2 + (1/5)/2, (1/5)), jnp.arange(-1/2 + (1/5)/2, 1/2 + (1/5)/2, (1/5)), jnp.arange(-1/2 + (1/5)/2, 1/2 + (1/5)/2, (1/5))
        self.dt = 0.1
        # grid and staggered grid for a 5x5x5 grid with a spacing of 1/5 and a timestep of 0.1

    def test_boris(self):

        Ex_interpolate = create_trilinear_interpolator(self.Ex, self.grid)
        Ey_interpolate = create_trilinear_interpolator(self.Ey, self.grid)
        Ez_interpolate = create_trilinear_interpolator(self.Ez, self.grid)
        Bx_interpolate = create_trilinear_interpolator(self.Bx, self.staggered_grid)
        By_interpolate = create_trilinear_interpolator(self.By, self.staggered_grid)
        Bz_interpolate = create_trilinear_interpolator(self.Bz, self.staggered_grid)
        # create interpolators for the electric and magnetic fields

        efield_atx = Ex_interpolate(self.x, self.y, self.z)
        efield_aty = Ey_interpolate(self.x, self.y, self.z)
        efield_atz = Ez_interpolate(self.x, self.y, self.z)
        # calculate the electric field at the particle positions
        bfield_atx = Bx_interpolate(self.x, self.y, self.z)
        bfield_aty = By_interpolate(self.x, self.y, self.z)
        bfield_atz = Bz_interpolate(self.x, self.y, self.z)

        boris_vmap = jax.vmap(boris_single_particle, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))
        newvx, newvy, newvz = boris_vmap(self.vx, self.vy, self.vz, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, self.q, self.m, self.dt, None)

        self.assertIsInstance(newvx, jnp.ndarray)
        self.assertIsInstance(newvy, jnp.ndarray)
        self.assertIsInstance(newvz, jnp.ndarray)
        # make sure the velocities are jax arrays
        self.assertTrue( jnp.allclose(jnp.abs(newvx),  1.0, rtol = 1e-2))
        # make sure the x velocity is unchanged
        self.assertLess( newvy, 0.1 )
        # make sure the y velocity is unchanged
        self.assertLess( newvz, 0.1 )
        # make sure the z velocity is 0.0 from the magnetic field



    def test_boris_single_particle(self):
        vx, vy, vz = 1.0, 0.0, 0.0
        x, y, z = 0.0, 0.0, 0.0

        q = 1.0
        m = 1.0
        E = jnp.array([0.0, 0.0, 0.0])
        B = jnp.array([0.0, 1.0, 0.0])
        dt = 0.1
        n_steps = 5000

        xs = []
        ys = []
        zs = []

        for i in range(n_steps):
            vx, vy, vz = boris_single_particle(vx, vy, vz, E[0], E[1], E[2], B[0], B[1], B[2], q, m, dt, None)
            x += vx * dt
            y += vy * dt
            z += vz * dt

            xs.append(x)
            ys.append(y)
            zs.append(z)


        def measure_xz_radius(xs, zs):
            """
            Measure the radius of the XZ cut by calculating the distance of each point
            in the XZ plane from the origin and returning the average and maximum radius.

            Parameters:
                xs (list): List of x-coordinates.
                zs (list): List of z-coordinates.

            Returns:
                tuple: A tuple containing the average radius and maximum radius.
            """
            xs = jnp.array(xs)
            zs = jnp.array(zs)
            # Calculate the distance of each point from the origin in the XZ plane
            radii = [jnp.sqrt(x**2 + z**2) for x, z in zip(xs, zs)]

            # Compute the average and maximum radius
            avg_radius = jnp.mean(jnp.asarray(radii))

            return avg_radius

        avg_radius  = measure_xz_radius(xs, zs)
        #print(f"Average radius: {avg_radius}")
        self.assertTrue( jnp.isclose(avg_radius, 1.28, atol=0.5) )


    def test_trilinear_convergence(self):

        # Run convergence tests for each interpolation method
        ################### BASIC TRIGONOMETRIC TEST #############################
        slope = convergence_test(partial(interpolation_wave_test, interp_func=create_trilinear_interpolator))
        # measure the convergence rate of the trilinear interpolation
        self.assertTrue(slope > 1.9)
        # assert that the order of the error is at least 2nd order
        ############################################################################

        ################### POLYNOMIAL TEST #######################################
        slope = convergence_test(partial(interpolation_polynomial_test, interp_func=create_trilinear_interpolator))
        # measure the convergence rate of the trilinear interpolation
        self.assertTrue(slope > 1.9)
        # assert that the order of the error is at least 2nd order
        ############################################################################

        ################### HIGH-FREQUENCY TEST ###################################
        slope = convergence_test(partial(interpolation_oscillatory_test, interp_func=create_trilinear_interpolator))
        # measure the convergence rate of the trilinear interpolation
        self.assertTrue(slope > 1.9)
        # assert that the order of the error is at least 2nd order
        ############################################################################

    def test_quadratic_convergence(self):

        ################### BASIC TRIGONOMETRIC TEST ###############################
        slope = convergence_test(partial(interpolation_wave_test, interp_func=create_quadratic_interpolator))
        # measure the convergence rate of the quadratic interpolation
        self.assertTrue(slope > 2.9)
        # assert that the order of the error is at least 3rd order
        ############################################################################

        ################### POLYNOMIAL TEST ########################################
        slope = convergence_test(partial(interpolation_polynomial_test, interp_func=create_quadratic_interpolator))
        # measure the convergence rate of the quadratic interpolation
        self.assertTrue(slope > 2.9)
        # assert that the order of the error is at least 3rd order
        ############################################################################

        ################### HIGH-FREQUENCY TEST ####################################
        slope = convergence_test(partial(interpolation_oscillatory_test, interp_func=create_quadratic_interpolator))
        # measure the convergence rate of the quadratic interpolation
        self.assertTrue(slope > 2.9)
        # assert that the order of the error is at least 3rd order
        ############################################################################


def interpolation_wave_test(nx, interp_func):
    # Define a symmetric domain
    x_wind = 2.0 * jnp.pi
    y_wind = 2.0 * jnp.pi
    z_wind = 2.0 * jnp.pi

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / (nx - 1)
    dy = y_wind / (nx - 1)
    dz = z_wind / (nx - 1)

    # Create meshgrid for field evaluation
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Define analytical test function: smooth trigonometric function
    # f(x,y,z) = sin(x) * cos(y) * sin(z)
    analytical_field = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)

    # Create the trilinear interpolator
    grid = (x_grid, y_grid, z_grid)
    interpolator = interp_func(analytical_field, grid)

    # Create test points that are offset from grid points
    # This tests the interpolation accuracy between grid points
    n_test = nx
    x_test = jnp.linspace(dx/3, x_wind - dx/3, n_test)
    y_test = jnp.linspace(dy/3, y_wind - dy/3, n_test)
    z_test = jnp.linspace(dz/3, z_wind - dz/3, n_test)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten for vectorized interpolation
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Compute analytical values at test points
    analytical_values = jnp.sin(x_test_flat) * jnp.cos(y_test_flat) * jnp.sin(z_test_flat)

    # Interpolate values at test points
    interpolated_values = interpolator(x_test_flat, y_test_flat, z_test_flat)

    # Compute mean squared error
    error = mae(interpolated_values, analytical_values)

    return error, dx


def interpolation_polynomial_test(nx, interp_func):

    # Define domain
    x_wind = 1.0
    y_wind = 1.0
    z_wind = 1.0

    # Create uniform grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / (nx - 1)
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Use a polynomial test function: f(x,y,z) = x³y + y³z + z³x
    # This has continuous derivatives and tests interpolation of curved surfaces
    analytical_field = X**3 * Y + Y**3 * Z + Z**3 * X

    # Create interpolator
    grid = (x_grid, y_grid, z_grid)
    interpolator = interp_func(analytical_field, grid)

    # Test at mid-points between grid points (maximum interpolation error)
    n_test = max(4, nx)  # Ensure we have enough test points
    x_test = jnp.linspace(dx/2, x_wind - dx/2, n_test)
    y_test = jnp.linspace(dx/2, y_wind - dx/2, n_test)
    z_test = jnp.linspace(dx/2, z_wind - dx/2, n_test)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Analytical values
    analytical_values = (x_test_flat**3 * y_test_flat +
                        y_test_flat**3 * z_test_flat +
                        z_test_flat**3 * x_test_flat)

    # Interpolated values
    interpolated_values = interpolator(x_test_flat, y_test_flat, z_test_flat)

    # Compute error
    error = mae(interpolated_values, analytical_values)

    return error, dx


def interpolation_oscillatory_test(nx, interp_func):

    # Define domain
    x_wind = jnp.pi
    y_wind = jnp.pi
    z_wind = jnp.pi

    # Create grid
    x_grid = jnp.linspace(0, x_wind, nx)
    y_grid = jnp.linspace(0, y_wind, nx)
    z_grid = jnp.linspace(0, z_wind, nx)

    dx = x_wind / (nx - 1)
    # create grid spacing

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # High-frequency test function: f(x,y,z) = cos(2x) * sin(3y) * cos(4z)
    analytical_field = jnp.cos(2*X) * jnp.sin(3*Y) * jnp.cos(4*Z)

    # Create interpolator
    grid = (x_grid, y_grid, z_grid)
    interpolator = interp_func(analytical_field, grid)

    # Test points slightly offset from grid
    n_test = max(6, nx)
    x_test = jnp.linspace(dx/4, x_wind - dx/4, n_test)
    y_test = jnp.linspace(dx/4, y_wind - dx/4, n_test)
    z_test = jnp.linspace(dx/4, z_wind - dx/4, n_test)

    X_test, Y_test, Z_test = jnp.meshgrid(x_test, y_test, z_test, indexing='ij')

    # Flatten
    x_test_flat = X_test.flatten()
    y_test_flat = Y_test.flatten()
    z_test_flat = Z_test.flatten()

    # Analytical values
    analytical_values = (jnp.cos(2*x_test_flat) *
                        jnp.sin(3*y_test_flat) *
                        jnp.cos(4*z_test_flat))

    # Interpolated values
    interpolated_values = interpolator(x_test_flat, y_test_flat, z_test_flat)

    # Compute error
    error = mae(interpolated_values, analytical_values)

    return error, dx

if __name__ == '__main__':
    unittest.main()