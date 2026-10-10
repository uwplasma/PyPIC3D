import unittest
import jax
import jax.numpy as jnp

from PyPIC3D.pusher.boris import boris_single_particle, interpolate_field_to_particles
from PyPIC3D.pusher.higuera_cary import higuera_cary_single_particle
from tests.kernel_fixtures import kernel_parameters


def mae(x, y):
    """
    Calculates the root mean squared error between two arrays.
    """
    return jnp.sqrt( jnp.mean( (x-y)**2 ) )


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
        _, self.dynamic_parameters = kernel_parameters(C=10.0)

    def test_boris(self):

        efield_atx = interpolate_field_to_particles(self.Ex, self.x, self.y, self.z, self.grid, shape_factor=1)
        efield_aty = interpolate_field_to_particles(self.Ey, self.x, self.y, self.z, self.grid, shape_factor=1)
        efield_atz = interpolate_field_to_particles(self.Ez, self.x, self.y, self.z, self.grid, shape_factor=1)
        # calculate the electric field at the particle positions
        bfield_atx = interpolate_field_to_particles(self.Bx, self.x, self.y, self.z, self.staggered_grid, shape_factor=1)
        bfield_aty = interpolate_field_to_particles(self.By, self.x, self.y, self.z, self.staggered_grid, shape_factor=1)
        bfield_atz = interpolate_field_to_particles(self.Bz, self.x, self.y, self.z, self.staggered_grid, shape_factor=1)

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

    def test_interpolate_field_to_particles_with_ghost_cells_on_reduced_axes(self):
        field = jnp.full((6, 3, 3), 7.0)
        x = jnp.array([-0.25, 0.0, 0.25])
        y = jnp.zeros_like(x)
        z = jnp.zeros_like(x)
        grid = (
            jnp.linspace(-0.5, 0.5, 6),
            jnp.array([-1.0, 0.0, 1.0]),
            jnp.array([-1.0, 0.0, 1.0]),
        )

        interpolated = interpolate_field_to_particles(
            field,
            x,
            y,
            z,
            grid,
            shape_factor=2,
            ghost_cells=True,
        )

        self.assertTrue(jnp.allclose(interpolated, 7.0))

    def test_interpolate_field_to_particles_accepts_explicit_active_axes(self):
        field = jnp.zeros((7, 5, 5))
        field = field.at[:, 2, 2].set(jnp.linspace(1.0, 2.0, 7))
        field = field.at[:, 1, 2].set(100.0)
        field = field.at[:, 3, 2].set(-50.0)
        field = field.at[:, 2, 1].set(25.0)
        field = field.at[:, 2, 3].set(-10.0)
        x = jnp.array([-0.25, 0.0, 0.25])
        y = jnp.zeros_like(x)
        z = jnp.zeros_like(x)
        grid = (
            jnp.linspace(-0.5, 0.5, 7),
            jnp.linspace(-1.0, 1.0, 5),
            jnp.linspace(-1.0, 1.0, 5),
        )

        interpolated = interpolate_field_to_particles(
            field,
            x,
            y,
            z,
            grid,
            shape_factor=2,
            ghost_cells=True,
            active_axes=(True, False, False),
            inactive_axis_indices=(None, 2, 2),
        )
        reference = interpolate_field_to_particles(
            field[:, 2:3, 2:3],
            x,
            y,
            z,
            (grid[0], grid[1][2:3], grid[2][2:3]),
            shape_factor=2,
            ghost_cells=False,
        )

        self.assertTrue(jnp.allclose(interpolated, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_interpolate_field_to_particles_preserves_trailing_tensor_axes(self):
        x_grid = jnp.linspace(-1.0, 1.0, 7)
        y_grid = jnp.linspace(-1.0, 1.0, 6)
        z_grid = jnp.linspace(-1.0, 1.0, 5)
        X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        scalar_field = 0.5 * X - 0.25 * Y + 0.75 * Z

        component_scale = jnp.arange(1.0, 10.0).reshape((3, 3))
        component_offset = jnp.arange(9.0).reshape((3, 3))
        tensor_field = (
            scalar_field[..., jnp.newaxis, jnp.newaxis] * component_scale
            + component_offset
        )

        x = jnp.asarray((-0.37, 0.18, 0.61))
        y = jnp.asarray((0.22, -0.44, 0.31))
        z = jnp.asarray((-0.15, 0.29, -0.52))
        grid = (x_grid, y_grid, z_grid)

        for shape_factor in (1, 2):
            with self.subTest(shape_factor=shape_factor):
                sampled_scalar = interpolate_field_to_particles(
                    scalar_field,
                    x,
                    y,
                    z,
                    grid,
                    shape_factor,
                )
                sampled_tensor = interpolate_field_to_particles(
                    tensor_field,
                    x,
                    y,
                    z,
                    grid,
                    shape_factor,
                )
                expected_tensor = (
                    sampled_scalar[..., jnp.newaxis, jnp.newaxis] * component_scale
                    + component_offset
                )

                self.assertEqual(sampled_tensor.shape, (x.size, 3, 3))
                self.assertTrue(
                    jnp.allclose(
                        sampled_tensor,
                        expected_tensor,
                        rtol=1.0e-12,
                        atol=1.0e-13,
                    )
                )



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

    def test_higuera_cary_zero_field_preserves_velocity(self):
        vx, vy, vz = higuera_cary_single_particle(
            0.2, -0.1, 0.05,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 0.1,
            self.dynamic_parameters,
        )
        # zero fields should not change the particle velocity
        self.assertTrue(jnp.allclose(jnp.array([vx, vy, vz]), jnp.array([0.2, -0.1, 0.05])))

    def test_higuera_cary_magnetic_field_preserves_speed(self):
        vx0, vy0, vz0 = 0.2, 0.0, 0.0
        vx, vy, vz = higuera_cary_single_particle(
            vx0, vy0, vz0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.1,
            self.dynamic_parameters,
        )
        # magnetic rotation should conserve speed
        old_speed = jnp.sqrt(vx0**2 + vy0**2 + vz0**2)
        new_speed = jnp.sqrt(vx**2 + vy**2 + vz**2)
        self.assertTrue(jnp.allclose(new_speed, old_speed, rtol=1e-12, atol=1e-12))

    def test_higuera_cary_electric_field_accelerates_along_field(self):
        vx, vy, vz = higuera_cary_single_particle(
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 0.1,
            self.dynamic_parameters,
        )
        # electric field in x should accelerate the particle in x only
        self.assertGreater(vx, 0.0)
        self.assertTrue(jnp.allclose(jnp.array([vy, vz]), jnp.array([0.0, 0.0])))

    def test_higuera_cary_vectorizes_with_per_particle_charge_and_mass(self):
        higuera_cary_vmap = jax.vmap(
            higuera_cary_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )
        newvx, newvy, newvz = higuera_cary_vmap(
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([1.0, 1.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, 1.0]),
            0.1,
            self.dynamic_parameters,
        )
        # per-particle charge and mass should work through the vectorized path
        self.assertEqual(newvx.shape, (2,))
        self.assertGreater(newvx[1], newvx[0])
        self.assertTrue(jnp.allclose(newvy, jnp.zeros(2)))
        self.assertTrue(jnp.allclose(newvz, jnp.zeros(2)))


    def test_trilinear_convergence(self):
        convergence_cases = (
            interpolation_wave_test,
            interpolation_polynomial_test,
            interpolation_oscillatory_test,
        )

        for convergence_case in convergence_cases:
            with self.subTest(convergence_case=convergence_case.__name__):
                coarse_error, _ = convergence_case(33, shape_factor=1)
                fine_error, _ = convergence_case(65, shape_factor=1)
                order = jnp.log2(coarse_error / fine_error)

                message = (
                    f"coarse_error={float(coarse_error)}, "
                    f"fine_error={float(fine_error)}, order={float(order)}"
                )
                self.assertGreater(order, 1.9, message)

    def test_quadratic_convergence(self):
        convergence_cases = (
            interpolation_wave_test,
            interpolation_polynomial_test,
            interpolation_oscillatory_test,
        )

        for convergence_case in convergence_cases:
            with self.subTest(convergence_case=convergence_case.__name__):
                coarse_error, _ = convergence_case(33, shape_factor=2)
                fine_error, _ = convergence_case(65, shape_factor=2)
                order = jnp.log2(coarse_error / fine_error)

                message = (
                    f"coarse_error={float(coarse_error)}, "
                    f"fine_error={float(fine_error)}, order={float(order)}"
                )
                self.assertGreater(order, 1.9, message)


def interpolation_wave_test(nx, shape_factor):
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

    grid = (x_grid, y_grid, z_grid)

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
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    # Compute mean squared error
    error = mae(interpolated_values, analytical_values)

    return error, dx


def interpolation_polynomial_test(nx, shape_factor):

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

    grid = (x_grid, y_grid, z_grid)

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
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    # Compute error
    error = mae(interpolated_values, analytical_values)

    return error, dx


def interpolation_oscillatory_test(nx, shape_factor):

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

    grid = (x_grid, y_grid, z_grid)

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
    interpolated_values = interpolate_field_to_particles(
        analytical_field, x_test_flat, y_test_flat, z_test_flat, grid, shape_factor
    )

    # Compute error
    error = mae(interpolated_values, analytical_values)

    return error, dx

if __name__ == '__main__':
    unittest.main()
