from tests.support.boris_runtime_fixtures import (
    D_FIELD_LOCATIONS,
    gather_shape_orders,
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

    def test_gather_shape_orders_follows_the_deposition_scheme(self):
        static_parameters, _ = kernel_parameters(shape_factor=2)
        Ex_location = D_FIELD_LOCATIONS[0]

        # only Esirkepov deposition has the reduced-order gather as its adjoint
        for current_deposition in ("direct", "GR_direct"):
            with self.subTest(current_deposition=current_deposition):
                parameters = static_parameters._replace(current_deposition=current_deposition)
                self.assertEqual(gather_shape_orders(Ex_location, parameters), 2)

        esirkepov = static_parameters._replace(current_deposition="esirkepov")
        self.assertEqual(gather_shape_orders(Ex_location, esirkepov), (1, 2, 2))

    def test_interpolate_field_to_particles_accepts_per_axis_shape_orders(self):
        grid_axis = jnp.linspace(-1.0, 1.0, 9)
        X, Y, Z = jnp.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij")
        field = 0.5 * X - 0.25 * Y + 0.75 * Z
        grid = (grid_axis, grid_axis, grid_axis)

        x = jnp.asarray((0.1, -0.4))
        y = jnp.asarray((0.2, 0.3))
        z = jnp.asarray((-0.1, 0.05))
        exact = 0.5 * x - 0.25 * y + 0.75 * z

        # a uniform order reproduces the linear field exactly, on either spelling
        for shape_factor in (1, 2):
            with self.subTest(shape_factor=shape_factor):
                for spelling in (shape_factor, (shape_factor,) * 3):
                    sampled = interpolate_field_to_particles(field, x, y, z, grid, spelling)
                    self.assertTrue(jnp.allclose(sampled, exact, rtol=1.0e-12, atol=1.0e-13))

        # order 0 on x snaps that axis to its nearest node and leaves y, z linear
        nearest_x = jnp.round((x - grid_axis[0]) / (grid_axis[1] - grid_axis[0]))
        snapped = grid_axis[0] + nearest_x * (grid_axis[1] - grid_axis[0])
        sampled = interpolate_field_to_particles(field, x, y, z, grid, (0, 1, 1))
        self.assertTrue(
            jnp.allclose(
                sampled,
                0.5 * snapped - 0.25 * y + 0.75 * z,
                rtol=1.0e-12,
                atol=1.0e-13,
            )
        )

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


if __name__ == "__main__":
    unittest.main()
