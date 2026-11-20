"""One-tile numerical and physical checks."""


from tests.support.static_metric_fixtures import (
    ConstitutiveFieldsFixtures,
    D_FIELD_LOCATIONS,
    GRDirectDepositionFixtures,
    GR_direct_deposition,
    HybridPusherFixtures,
    J_from_rhov,
    Metric,
    ParticleMetric,
    SpeciesConfig,
    StaticMetricInitializationFixtures,
    TiledParticles,
    _constant_tiled_vector,
    _metric_locations_with_grids,
    _replace_lapse_shift,
    _single_particle_state,
    active_interior,
    compute_covariant_E,
    compute_covariant_H,
    coordinate_velocity,
    densitize_vector,
    empty_tiled_vector,
    geodesic_acceleration,
    hybrid_boris_geodesic_push,
    hybrid_pusher,
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
    initialize_kerr_schild_spherical_metric,
    interpolate_metric,
    jax,
    jnp,
    kernel_parameters,
    location_grid,
    magnetic_boris_rotation,
    physical_vector,
    unittest,
)


class TestStaticMetricInitialization(StaticMetricInitializationFixtures, unittest.TestCase):
    def test_flat_cylindrical_particle_metric_derivative_matches_analytic(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=8,
            Ny=2,
            Nz=1,
            x_wind=8.0,
            y_wind=1.0,
            z_wind=1.0,
            x_min=2.5,
            y_min=0.2,
            z_min=0.2,
            tile_shape=(8, 2, 1),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cylindrical_metric(static_parameters, dynamic_parameters)

        g = int(static_parameters.guard_cells)
        grid = tuple(axis[0, 0, 0] for axis in dynamic_parameters.grids.tiled_center_grid)
        r = jnp.linspace(3.2, 9.3, 7)
        position = jnp.stack((r, jnp.full_like(r, 0.7), jnp.full_like(r, 0.7)), axis=-1)
        sampled = interpolate_metric(
            jax.tree.map(lambda array: array[0, 0, 0], metric.center),
            position, grid, "flat_cylindrical", (True, True, False), (g, g, g),
        )

        # gamma^{phi phi} = 1/r^2, so d_r gamma^{phi phi} = -2/r^3.
        self.assertTrue(jnp.allclose(sampled.gamma_inv[:, 1, 1], 1.0 / r**2, rtol=1e-3))
        self.assertTrue(jnp.allclose(sampled.grad_gamma_inv[:, 0, 1, 1], -2.0 / r**3, rtol=2e-2))
        self.assertTrue(jnp.allclose(sampled.grad_gamma_inv[:, 0, 0, 0], 0.0, atol=1e-12))

    def test_flat_cylindrical_metric_stores_signed_sqrt_gamma_at_all_yee_locations(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=8,
            Ny=2,
            Nz=1,
            x_wind=8.0,
            y_wind=1.0,
            z_wind=1.0,
            x_min=-3.75,
            y_min=0.2,
            z_min=0.2,
            tile_shape=(8, 2, 1),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
            metric="flat_cylindrical",
        )
        metric = initialize_flat_cylindrical_metric(static_parameters, dynamic_parameters)

        for metric_at_location, grid in _metric_locations_with_grids(metric, dynamic_parameters):
            R = grid[0][..., :, jnp.newaxis, jnp.newaxis]
            expected = jnp.broadcast_to(R, metric_at_location.sqrt_gamma.shape)

            self.assertTrue(jnp.allclose(metric_at_location.sqrt_gamma, expected))
            self.assertTrue(jnp.any(metric_at_location.sqrt_gamma < 0.0))
            self.assertTrue(jnp.any(metric_at_location.sqrt_gamma > 0.0))

    def test_spherical_metrics_store_signed_sqrt_gamma_at_all_yee_locations(self):
        ntheta = 8
        dtheta = 2.0 * jnp.pi / ntheta
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=4,
            Ny=ntheta,
            Nz=1,
            x_wind=1.0,
            y_wind=2.0 * jnp.pi,
            z_wind=1.0,
            x_min=2.0,
            y_min=0.25 * dtheta,
            z_min=0.2,
            tile_shape=(4, ntheta, 1),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        mass = 0.1
        spin = 0.2
        flat_metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
        kerr_metric = initialize_kerr_schild_spherical_metric(
            static_parameters,
            dynamic_parameters,
            mass=mass,
            spin=spin,
        )

        for metric_at_location, grid in _metric_locations_with_grids(flat_metric, dynamic_parameters):
            R = grid[0][..., :, jnp.newaxis, jnp.newaxis]
            theta = grid[1][..., jnp.newaxis, :, jnp.newaxis]
            expected = jnp.broadcast_to(
                R**2 * jnp.sin(theta),
                metric_at_location.sqrt_gamma.shape,
            )

            self.assertTrue(jnp.allclose(metric_at_location.sqrt_gamma, expected))
            self.assertTrue(jnp.any(metric_at_location.sqrt_gamma < 0.0))
            self.assertTrue(jnp.any(metric_at_location.sqrt_gamma > 0.0))

        for metric_at_location, grid in _metric_locations_with_grids(kerr_metric, dynamic_parameters):
            R = grid[0][..., :, jnp.newaxis, jnp.newaxis]
            theta = grid[1][..., jnp.newaxis, :, jnp.newaxis]
            rho_squared = R**2 + spin**2 * jnp.cos(theta) ** 2
            xi = 1.0 + 2.0 * mass * R / rho_squared
            expected = jnp.broadcast_to(
                rho_squared * jnp.sqrt(xi) * jnp.sin(theta),
                metric_at_location.sqrt_gamma.shape,
            )

            self.assertTrue(jnp.allclose(metric_at_location.sqrt_gamma, expected))
            self.assertTrue(jnp.any(metric_at_location.sqrt_gamma < 0.0))
            self.assertTrue(jnp.any(metric_at_location.sqrt_gamma > 0.0))


class TestConstitutiveFields(ConstitutiveFieldsFixtures, unittest.TestCase):
    def test_static_metric_constitutive_fields_include_lapse_and_shift_terms(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=4,
            Ny=4,
            Nz=4,
            x_wind=4.0,
            y_wind=4.0,
            z_wind=4.0,
            tile_shape=(4, 4, 4),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        metric = _replace_lapse_shift(metric, lapse=0.7, shift=(0.2, -0.1, 0.15))
        D = _constant_tiled_vector(static_parameters, dynamic_parameters, (1.1, -0.7, 0.3))
        B = _constant_tiled_vector(static_parameters, dynamic_parameters, (0.4, 0.9, -1.2))
        D, B = densitize_vector(D, metric.D), densitize_vector(B, metric.B)

        E = compute_covariant_E(D, B, metric)
        H = compute_covariant_H(D, B, metric)

        g = int(static_parameters.guard_cells)
        active = (slice(None), slice(None), slice(None), slice(g, -g), slice(g, -g), slice(g, -g))
        beta = jnp.asarray((0.2, -0.1, 0.15))
        D0 = jnp.asarray((1.1, -0.7, 0.3))
        B0 = jnp.asarray((0.4, 0.9, -1.2))
        expected_E = 0.7 * D0 + jnp.cross(beta, B0)
        expected_H = 0.7 * B0 - jnp.cross(beta, D0)

        for i in range(3):
            self.assertTrue(jnp.allclose(E[i][active], expected_E[i]))
            self.assertTrue(jnp.allclose(H[i][active], expected_H[i]))


class TestHybridPusher(HybridPusherFixtures, unittest.TestCase):
    def test_magnetic_boris_rotation_raises_covariant_momentum_in_cross_product(self):
        gamma = jnp.asarray(
            (
                (1.0, 0.2, 0.0),
                (0.2, 2.0, 0.1),
                (0.0, 0.1, 3.0),
            )
        )
        gamma_inv = jnp.linalg.inv(gamma)
        metric = Metric(
            lapse=jnp.asarray(0.8),
            shift=jnp.asarray((0.0, 0.0, 0.0)),
            gamma=gamma,
            gamma_inv=gamma_inv,
            sqrt_gamma=jnp.sqrt(jnp.linalg.det(gamma)),
        )
        u_minus = jnp.asarray((0.3, -0.4, 0.2))
        B_con = jnp.asarray((0.1, 0.5, -0.2))
        q_over_m = 1.7
        dt = 0.31

        u_plus = magnetic_boris_rotation(u_minus, B_con, metric, q_over_m, dt)

        Gamma_minus = jnp.sqrt(1.0 + jnp.einsum("i,ij,j->", u_minus, gamma_inv, u_minus))
        u0_bar = Gamma_minus / metric.lapse
        t_con = q_over_m * B_con * 0.5 * dt / u0_bar
        t_cov = gamma @ t_con
        t_norm = jnp.dot(t_con, t_cov)
        u_minus_con = gamma_inv @ u_minus
        u_prime = u_minus + metric.sqrt_gamma * jnp.cross(u_minus_con, t_con)
        s_con = 2.0 * t_con / (1.0 + t_norm)
        u_prime_con = gamma_inv @ u_prime
        expected = u_minus + metric.sqrt_gamma * jnp.cross(u_prime_con, s_con)

        self.assertAllClose(u_plus, expected, rtol=0.0, atol=1.0e-12)

    def test_coordinate_velocity_uses_lapse_scaled_contravariant_velocity_minus_shift(self):
        gamma = jnp.asarray(
            (
                (1.0, 0.2, 0.0),
                (0.2, 1.5, 0.1),
                (0.0, 0.1, 2.0),
            )
        )
        gamma_inv = jnp.linalg.inv(gamma)
        metric = Metric(
            lapse=jnp.asarray(0.7),
            shift=jnp.asarray((0.2, -0.1, 0.15)),
            gamma=gamma,
            gamma_inv=gamma_inv,
            sqrt_gamma=jnp.sqrt(jnp.linalg.det(gamma)),
        )
        u_cov = jnp.asarray((0.4, -0.2, 0.1))

        dx_dt = coordinate_velocity(u_cov, metric)

        Gamma = jnp.sqrt(1.0 + jnp.einsum("i,ij,j->", u_cov, gamma_inv, u_cov))
        expected = metric.lapse * (gamma_inv @ u_cov) / Gamma - metric.shift
        self.assertAllClose(dx_dt, expected, rtol=0.0, atol=1.0e-12)

    def test_geodesic_acceleration_returns_zero_for_flat_constant_metric(self):
        metric = ParticleMetric(
            lapse=jnp.asarray(1.0),
            shift=jnp.asarray((0.0, 0.0, 0.0)),
            gamma=jnp.eye(3),
            gamma_inv=jnp.eye(3),
            sqrt_gamma=jnp.asarray(1.0),
            grad_lapse=jnp.zeros(3),
            grad_shift=jnp.zeros((3, 3)),
            grad_gamma_inv=jnp.zeros((3, 3, 3)),
        )
        u_cov = jnp.asarray((0.4, -0.2, 0.1))

        du_dt = geodesic_acceleration(u_cov, metric)

        self.assertEqual(du_dt.shape, u_cov.shape)
        self.assertAllClose(du_dt, jnp.zeros(3), rtol=0.0, atol=1.0e-12)

    def test_hybrid_boris_geodesic_push_advances_flat_neutral_particle_with_u_over_gamma(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=4,
            Ny=4,
            Nz=4,
            x_wind=4.0,
            y_wind=4.0,
            z_wind=4.0,
            dt=0.2,
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.3, 0.4, 0.0))
        species = species._replace(charge=jnp.asarray([0.0]))
        zeros = empty_tiled_vector(static_parameters, dynamic_parameters)

        pushed, centered = hybrid_boris_geodesic_push(
            particles,
            species,
            zeros,
            zeros,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        gamma = jnp.sqrt(1.0 + 0.3**2 + 0.4**2)
        expected_x = jnp.asarray((0.3, 0.4, 0.0)) * dynamic_parameters.dt / gamma
        self.assertTrue(jnp.allclose(pushed.x[0, 0, 0, 0, 0], expected_x))
        self.assertTrue(jnp.allclose(centered.x[0, 0, 0, 0, 0], 0.5 * expected_x))
        self.assertTrue(jnp.allclose(pushed.u[0, 0, 0, 0, 0], jnp.asarray((0.3, 0.4, 0.0))))

    def test_hybrid_boris_geodesic_push_uses_current_position_for_both_electric_half_steps(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=8,
            Ny=2,
            Nz=2,
            x_wind=8.0,
            y_wind=2.0,
            z_wind=2.0,
            dt=0.2,
            tile_shape=(8, 2, 2),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)

        D_grid_x = dynamic_parameters.grids.tiled_center_grid[0][:, :, :, :, jnp.newaxis, jnp.newaxis]
        D = (D[0].at[:, :, :, :, :, :].set(D_grid_x), D[1], D[2])

        x_n = jnp.asarray((2.0, 0.0, 0.0))
        u_n_minushalf = jnp.asarray((0.3, 0.0, 0.0))
        particles = TiledParticles(
            x=x_n.reshape((1, 1, 1, 1, 1, 3)),
            u=u_n_minushalf.reshape((1, 1, 1, 1, 1, 3)),
            active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
        )
        species = SpeciesConfig(
            charge=jnp.asarray([1.0]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.0]),
            update_x=jnp.asarray([[True, True, True]]),
        )

        pushed, centered = hybrid_boris_geodesic_push(
            particles,
            species,
            D,
            B,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        D_at_x_n = hybrid_pusher.gather_vector(
            tuple(component[0, 0, 0] for component in D),
            D_FIELD_LOCATIONS,
            x_n[None, :],
            tuple(axis[0, 0, 0] for axis in dynamic_parameters.grids.tiled_center_grid),
            tuple(axis[0, 0, 0] for axis in dynamic_parameters.grids.tiled_vertex_grid),
            static_parameters,
            (True, True, True),
            (static_parameters.guard_cells,) * 3,
        )[0, 0]
        expected_u = u_n_minushalf + dynamic_parameters.dt * jnp.asarray((D_at_x_n, 0.0, 0.0))
        expected_dx_dt = expected_u / jnp.sqrt(1.0 + jnp.dot(expected_u, expected_u))

        self.assertAllClose(
            pushed.u[0, 0, 0, 0, 0],
            expected_u,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertTrue(jnp.allclose(centered.u[0, 0, 0, 0, 0], pushed.u[0, 0, 0, 0, 0]))
        self.assertAllClose(
            centered.x[0, 0, 0, 0, 0],
            x_n + 0.5 * dynamic_parameters.dt * expected_dx_dt,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertAllClose(
            pushed.x[0, 0, 0, 0, 0],
            x_n + dynamic_parameters.dt * expected_dx_dt,
            rtol=0.0,
            atol=1.0e-6,
        )


class TestGRDirectDeposition(GRDirectDepositionFixtures, unittest.TestCase):
    def test_GR_direct_deposition_uses_lapse_scaled_contravariant_three_velocity(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=1,
            Ny=1,
            Nz=1,
            x_wind=1.0,
            y_wind=1.0,
            z_wind=1.0,
            dt=0.1,
            tile_shape=(1, 1, 1),
            shape_factor=1,
            current_filter="none",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        metric = metric._replace(
            center=metric.center._replace(lapse=jnp.full_like(metric.center.lapse, 2.0))
        )
        particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.5, 0.0, 0.0))
        J = empty_tiled_vector(static_parameters, dynamic_parameters)

        J = GR_direct_deposition(
            particles,
            species,
            J,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        interior = active_interior(static_parameters, dynamic_parameters)
        gamma = jnp.sqrt(1.0 + 0.5**2)
        expected = 2.0 * 0.5 / gamma
        self.assertTrue(jnp.allclose(J[0][interior], expected))
        self.assertTrue(jnp.allclose(J[1][interior], 0.0))
        self.assertTrue(jnp.allclose(J[2][interior], 0.0))

    def test_GR_direct_deposition_returns_fpic_shifted_source_current(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=1,
            Ny=1,
            Nz=1,
            x_wind=1.0,
            y_wind=1.0,
            z_wind=1.0,
            dt=0.1,
            tile_shape=(1, 1, 1),
            shape_factor=1,
            current_filter="none",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        metric = _replace_lapse_shift(metric, lapse=0.7, shift=(0.2, -0.1, 0.15))
        particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.5, 0.0, 0.0))
        J = empty_tiled_vector(static_parameters, dynamic_parameters)

        J = GR_direct_deposition(
            particles,
            species,
            J,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        interior = active_interior(static_parameters, dynamic_parameters)
        gamma = jnp.sqrt(1.0 + 0.5**2)
        expected = jnp.asarray((0.7 * 0.5 / gamma, 0.0, 0.0)) - jnp.asarray((0.2, -0.1, 0.15))
        self.assertTrue(jnp.allclose(J[0][interior], expected[0]))
        self.assertTrue(jnp.allclose(J[1][interior], expected[1]))
        self.assertTrue(jnp.allclose(J[2][interior], expected[2]))

    def test_GR_direct_deposition_is_adjoint_to_staggered_field_gather(self):
        positions = jnp.asarray(
            (
                ((0.1, 0.2, 0.3), (3.9, 1.7, 2.2)),
                ((4.1, 2.8, 1.4), (7.9, 3.8, 3.7)),
            )
        ).reshape((1, 1, 1, 1, 4, 3))
        u_cov = jnp.asarray(
            (
                ((0.31, -0.17, 0.09), (-0.22, 0.28, -0.13)),
                ((0.19, 0.11, -0.24), (-0.27, -0.16, 0.21)),
            )
        ).reshape((1, 1, 1, 1, 4, 3))
        active = jnp.ones((1, 1, 1, 1, 4), dtype=bool)
        species = SpeciesConfig(
            charge=jnp.asarray([-0.7]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.3]),
            update_x=jnp.asarray([[True, True, True]]),
        )

        for shape_factor in (1, 2):
            static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
                Nx=8,
                Ny=4,
                Nz=4,
                x_wind=8.0,
                y_wind=4.0,
                z_wind=4.0,
                x_min=0.0,
                y_min=0.0,
                z_min=0.0,
                dt=0.1,
                tile_shape=(8, 4, 4),
                shape_factor=shape_factor,
                current_filter="none",
                solver="static_metric",
                current_deposition="GR_direct",
                particle_pusher="hybrid_boris_geodesic",
            )
            metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
            metric = _replace_lapse_shift(metric, lapse=0.73, shift=(0.12, -0.08, 0.05))
            particles = TiledParticles(x=positions, u=u_cov, active=active)

            center_grid = dynamic_parameters.grids.tiled_center_grid
            vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
            D_grids = tuple(
                location_grid(center_grid, vertex_grid, location)
                for location in D_FIELD_LOCATIONS
            )
            D = []
            for component, grid in enumerate(D_grids):
                x_grid, y_grid, z_grid = grid
                field = (
                    jnp.sin(2.0 * jnp.pi * x_grid[..., :, jnp.newaxis, jnp.newaxis] / 8.0)
                    + (0.3 + 0.1 * component)
                    * jnp.cos(2.0 * jnp.pi * y_grid[..., jnp.newaxis, :, jnp.newaxis] / 4.0)
                    + (0.2 - 0.05 * component)
                    * jnp.sin(2.0 * jnp.pi * z_grid[..., jnp.newaxis, jnp.newaxis, :] / 4.0)
                )
                D.append(field)
            D = tuple(D)

            J = GR_direct_deposition(
                particles,
                species,
                empty_tiled_vector(static_parameters, dynamic_parameters),
                metric,
                static_parameters,
                dynamic_parameters,
            )

            g = int(static_parameters.guard_cells)
            interior = (
                slice(None),
                slice(None),
                slice(None),
                slice(g, -g),
                slice(g, -g),
                slice(g, -g),
            )
            # J is the native density sqrt(gamma) J^i, so it already carries the volume factor.
            grid_work = sum(jnp.sum(D[i][interior] * J[i][interior]) for i in range(3))
            grid_work *= dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz

            gathered_D = []
            for tx in range(1):
                gathered_D.append(
                    hybrid_pusher.gather_vector(
                        tuple(D[i][tx, 0, 0] for i in range(3)),
                        D_FIELD_LOCATIONS,
                        particles.x[tx, 0, 0],
                        tuple(axis[tx, 0, 0] for axis in center_grid),
                        tuple(axis[tx, 0, 0] for axis in vertex_grid),
                        static_parameters,
                        (True, True, True),
                        (g, g, g),
                    )
                )
            gathered_D = jnp.stack(gathered_D, axis=0).reshape(particles.x.shape)

            Gamma = jnp.sqrt(1.0 + jnp.sum(particles.u**2, axis=-1))
            source_velocity = 0.73 * particles.u / Gamma[..., jnp.newaxis]
            source_velocity -= jnp.asarray((0.12, -0.08, 0.05))
            weighted_charge = species.charge[0] * species.weight[0]
            particle_work = weighted_charge * jnp.sum(
                particles.active[..., jnp.newaxis]
                * source_velocity
                * gathered_D
            )

            scale = jnp.maximum(jnp.abs(grid_work), jnp.abs(particle_work))
            relative_residual = jnp.abs(grid_work - particle_work) / scale
            self.assertLess(relative_residual, 1.0e-12)

    def test_flat_GR_direct_deposition_matches_standard_stencil_on_reduced_axes(self):
        positions = jnp.asarray(
            ((0.1, 0.0, 0.0), (0.9, 0.0, 0.0), (7.1, 0.0, 0.0), (7.9, 0.0, 0.0))
        ).reshape((1, 1, 1, 1, 4, 3))
        u_cov = jnp.asarray(
            ((0.31, -0.17, 0.09), (-0.22, 0.28, -0.13), (0.19, 0.11, -0.24), (-0.27, -0.16, 0.21))
        ).reshape((1, 1, 1, 1, 4, 3))
        active = jnp.ones((1, 1, 1, 1, 4), dtype=bool)
        species = SpeciesConfig(
            charge=jnp.asarray([-0.7]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.3]),
            update_x=jnp.asarray([[True, True, True]]),
        )

        for shape_factor in (1, 2):
            static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
                Nx=8,
                Ny=1,
                Nz=1,
                x_wind=8.0,
                y_wind=1.0,
                z_wind=1.0,
                x_min=0.0,
                y_min=-0.5,
                z_min=-0.5,
                tile_shape=(8, 1, 1),
                shape_factor=shape_factor,
                current_filter="none",
                solver="static_metric",
                current_deposition="GR_direct",
                particle_pusher="hybrid_boris_geodesic",
            )
            metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
            GR_particles = TiledParticles(x=positions, u=u_cov, active=active)
            Gamma = jnp.sqrt(1.0 + jnp.sum(u_cov**2, axis=-1))
            standard_particles = GR_particles._replace(u=u_cov / Gamma[..., jnp.newaxis])
            J_template = empty_tiled_vector(static_parameters, dynamic_parameters)

            GR_J = GR_direct_deposition(
                GR_particles,
                species,
                J_template,
                metric,
                static_parameters,
                dynamic_parameters,
            )
            standard_J = J_from_rhov(
                standard_particles,
                species,
                J_template,
                static_parameters,
                dynamic_parameters,
            )

            for GR_component, standard_component in zip(GR_J, standard_J):
                self.assertAllClose(
                    GR_component,
                    standard_component,
                    rtol=0.0,
                    atol=1.0e-12,
                )

    def test_GR_direct_deposition_returns_conformal_spherical_current(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=8,
            Ny=1,
            Nz=1,
            x_wind=8.0,
            y_wind=1.0,
            z_wind=1.0,
            x_min=2.0,
            y_min=0.4,
            z_min=0.2,
            dt=0.1,
            tile_shape=(8, 1, 1),
            shape_factor=1,
            current_filter="none",
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
            metric="flat_spherical",
        )
        metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
        J_template = empty_tiled_vector(static_parameters, dynamic_parameters)
        species = SpeciesConfig(
            charge=jnp.asarray([1.0]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.0]),
            update_x=jnp.asarray([[True, True, True]]),
        )

        def deposit_radial_particle(radius):
            particles = TiledParticles(
                x=jnp.asarray((radius, 0.4, 0.2)).reshape((1, 1, 1, 1, 1, 3)),
                u=jnp.asarray((0.5, 0.0, 0.0)).reshape((1, 1, 1, 1, 1, 3)),
                active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
            )
            J = GR_direct_deposition(
                particles,
                species,
                J_template,
                metric,
                static_parameters,
                dynamic_parameters,
            )

            g = int(static_parameters.guard_cells)
            active = (
                slice(None),
                slice(None),
                slice(None),
                slice(g, -g),
                slice(g, -g),
                slice(g, -g),
            )
            physical_current_sum = jnp.sum(physical_vector(J, metric.D)[0][active])
            conformal_flux = jnp.sum(
                J[0][active]
                * dynamic_parameters.dx
                * dynamic_parameters.dy
                * dynamic_parameters.dz
            )
            return physical_current_sum, conformal_flux

        inner_current, inner_flux = deposit_radial_particle(2.5)
        outer_current, outer_flux = deposit_radial_particle(6.5)
        expected_flux = 0.5 / jnp.sqrt(1.0 + 0.5**2)

        self.assertTrue(jnp.allclose(inner_flux, expected_flux, rtol=1.0e-5, atol=1.0e-6))
        self.assertTrue(jnp.allclose(outer_flux, expected_flux, rtol=1.0e-5, atol=1.0e-6))
        self.assertGreater(inner_current, outer_current)


if __name__ == "__main__":
    unittest.main()
