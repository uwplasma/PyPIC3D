from tests.support.boris_runtime_fixtures import (
    interpolation_oscillatory_test,
    interpolation_polynomial_test,
    interpolation_wave_test,

    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    boris_single_particle,
    galerkin_ordering,
    higuera_cary_single_particle,
    interpolate_field_to_particles,
    jax,
    jnp,
    kernel_parameters,
    unittest,
)


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

    def test_galerkin_ordering_reduces_the_order_on_staggered_axes(self):
        # particle shapes are anchored on the cell-centered grid, so a component
        # is staggered exactly on the axes its Yee location marks "V"
        self.assertEqual(
            [galerkin_ordering(location, 1) for location in D_FIELD_LOCATIONS],
            [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
        )
        self.assertEqual(
            [galerkin_ordering(location, 2) for location in D_FIELD_LOCATIONS],
            [(1, 2, 2), (2, 1, 2), (2, 2, 1)],
        )
        self.assertEqual(
            [galerkin_ordering(location, 1) for location in B_FIELD_LOCATIONS],
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        self.assertEqual(
            [galerkin_ordering(location, 2) for location in B_FIELD_LOCATIONS],
            [(2, 1, 1), (1, 2, 1), (1, 1, 2)],
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


if __name__ == "__main__":
    unittest.main()
