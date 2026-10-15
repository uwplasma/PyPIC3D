"""Single-device numerical tests."""

from contextlib import ExitStack
from unittest.mock import Mock, patch

from tests.support.static_metric_fixtures import (
    BC_TYPE_PARTICLE,
    ConstitutiveFieldsFixtures,
    D_FIELD_LOCATIONS,
    GRDirectDepositionFixtures,
    GR_direct_deposition,
    HybridPusherFixtures,
    J_from_rhov,
    Metric,
    ParticleMetric,
    SpeciesConfig,
    StaticMetricDispatchFixtures,
    StaticMetricInitializationFixtures,
    StaticMetricTimeLoopFixtures,
    TiledParticles,
    _constant_tiled_vector,
    _encode_current_calculation,
    _metric_locations_with_grids,
    _replace_lapse_shift,
    _single_particle_state,
    _validate_tiled_yee_configuration,
    active_interior,
    compute_covariant_E,
    compute_covariant_H,
    coordinate_velocity,
    densitize_fields,
    densitize_vector,
    empty_tiled_vector,
    geodesic_acceleration,
    hybrid_boris_geodesic_push,
    hybrid_pusher,
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
    interpolate_metric,
    itertools,
    jax,
    jnp,
    kernel_parameters,
    location_grid,
    magnetic_boris_rotation,
    physical_vector,
    shard_tiled_particles,
    tiled_bilinear_filter_vector,
    tiled_digital_filter_vector,
    time_loop_static_metric,
    unittest,
    update_D,
    validate_field_solver,
)


class TestStaticMetricInitialization(StaticMetricInitializationFixtures, unittest.TestCase):
    def test_flat_cartesian_metric_matches_center_grid_shape(self):
        static_parameters, dynamic_parameters = kernel_parameters(Nx=4, Ny=3, Nz=2, guard_cells=3)

        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        g = int(static_parameters.guard_cells)
        shape = (1, 1, 1, 4 + 2 * g, 3 + 2 * g, 2 + 2 * g)

        self.assertEqual(metric.center.lapse.shape, shape)
        self.assertEqual(metric.center.gamma_inv.shape, shape + (3, 3))
        self.assertTrue(jnp.allclose(metric.center.lapse, 1.0))
        self.assertTrue(jnp.allclose(metric.center.gamma_inv[..., 0, 0], 1.0))
        self.assertTrue(jnp.allclose(metric.center.sqrt_gamma, 1.0))


    def test_kerr_schild_metric_initializers_build_finite_metrics(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=4,
            Ny=4,
            Nz=4,
            x_wind=1.0,
            y_wind=0.8,
            z_wind=1.0,
            x_min=2.0,
            y_min=0.17,
            z_min=0.5,
            tile_shape=(4, 4, 4),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )

        cartesian = initialize_kerr_schild_cartesian_metric(
            static_parameters,
            dynamic_parameters,
            mass=0.1,
            spin=0.2,
        )
        spherical = initialize_kerr_schild_spherical_metric(
            static_parameters,
            dynamic_parameters,
            mass=0.1,
            spin=0.2,
        )

        for metric in (cartesian, spherical):
            for array in metric.center:
                self.assertTrue(jnp.all(jnp.isfinite(array)))
            self.assertTrue(jnp.all(metric.center.lapse < 1.0))
            self.assertTrue(jnp.allclose(metric.center.gamma @ metric.center.gamma_inv, jnp.eye(3), atol=1e-12))


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
            static_parameters.shape_factor,
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


    def test_hybrid_boris_geodesic_push_accepts_multiple_species_in_one_tile(self):
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
        zeros = empty_tiled_vector(static_parameters, dynamic_parameters)
        x = jnp.zeros((1, 1, 1, 2, 3, 3))
        u = jnp.asarray(
            (
                ((0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ((-0.1, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ),
            dtype=float,
        ).reshape((1, 1, 1, 2, 3, 3))
        active = jnp.asarray(
            (
                (True, True, False),
                (True, True, False),
            ),
            dtype=bool,
        ).reshape((1, 1, 1, 2, 3))
        particles = TiledParticles(x=x, u=u, active=active)
        species = SpeciesConfig(
            charge=jnp.asarray([0.0, 0.0]),
            mass=jnp.asarray([1.0, 2.0]),
            weight=jnp.asarray([1.0, 1.0]),
            update_x=jnp.asarray([[True, True, True], [True, True, True]]),
        )

        pushed, centered = hybrid_boris_geodesic_push(
            particles,
            species,
            zeros,
            zeros,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        self.assertEqual(pushed.x.shape, x.shape)
        self.assertEqual(centered.x.shape, x.shape)
        self.assertTrue(jnp.all(jnp.isfinite(pushed.x)))


    def test_hybrid_boris_geodesic_push_masks_position_and_velocity_by_species_direction(self):
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
        D = _constant_tiled_vector(static_parameters, dynamic_parameters, (0.2, 0.2, 0.2))
        B = empty_tiled_vector(static_parameters, dynamic_parameters)
        particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 2, 1, 3)),
            u=jnp.zeros((1, 1, 1, 2, 1, 3)),
            active=jnp.ones((1, 1, 1, 2, 1), dtype=bool),
        )
        species = SpeciesConfig(
            charge=jnp.asarray([1.0, 1.0]),
            mass=jnp.asarray([1.0, 1.0]),
            weight=jnp.asarray([1.0, 1.0]),
            update_x=jnp.asarray(
                (
                    (False, True, False),
                    (True, False, True),
                )
            ),
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

        update_mask = species.update_x.reshape((1, 1, 1, 2, 1, 3))
        self.assertTrue(jnp.allclose(jnp.where(update_mask, 0.0, pushed.u), 0.0))
        self.assertTrue(jnp.allclose(jnp.where(update_mask, 0.0, pushed.x), 0.0))
        self.assertTrue(jnp.allclose(jnp.where(update_mask, 0.0, centered.x), 0.0))
        self.assertTrue(jnp.all(jnp.abs(pushed.u[update_mask]) > 0.0))
        self.assertTrue(jnp.all(jnp.abs(pushed.x[update_mask]) > 0.0))
        self.assertTrue(jnp.all(jnp.abs(centered.x[update_mask]) > 0.0))



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


    def test_GR_direct_deposition_uses_distributed_filters(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=8,
            Ny=1,
            Nz=1,
            x_wind=8.0,
            y_wind=1.0,
            z_wind=1.0,
            dt=0.1,
            tile_shape=(8, 1, 1),
            shape_factor=1,
            current_filter="none",
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)

        x = jnp.zeros((1, 1, 1, 1, 1, 3)).at[0, 0, 0, 0, 0, 0].set(-2.0)
        u = jnp.zeros_like(x).at[0, 0, 0, 0, 0, 0].set(0.5)
        active = jnp.zeros((1, 1, 1, 1, 1), dtype=bool).at[0, 0, 0, 0, 0].set(True)
        particles = TiledParticles(x=x, u=u, active=active)
        species = SpeciesConfig(
            charge=jnp.asarray([1.0]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.0]),
            update_x=jnp.asarray([[True, True, True]]),
        )
        J_template = empty_tiled_vector(static_parameters, dynamic_parameters)

        raw_J = GR_direct_deposition(
            particles,
            species,
            J_template,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        filter_cases = (
            (
                "digital",
                lambda J, parameters: tiled_digital_filter_vector(
                    J,
                    dynamic_parameters.alpha,
                    parameters,
                    bc_type=BC_TYPE_PARTICLE,
                ),
            ),
            (
                "bilinear",
                lambda J, parameters: tiled_bilinear_filter_vector(
                    J,
                    parameters,
                    bc_type=BC_TYPE_PARTICLE,
                ),
            ),
        )
        for filter_name, filter_function in filter_cases:
            filtered_parameters = static_parameters._replace(current_filter=filter_name)
            filtered_J = GR_direct_deposition(
                particles,
                species,
                J_template,
                metric,
                filtered_parameters,
                dynamic_parameters,
            )
            expected_J = filter_function(raw_J, filtered_parameters)

            for actual_component, expected_component in zip(filtered_J, expected_J):
                self.assertTrue(jnp.allclose(actual_component, expected_component))


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


    def test_GR_direct_deposition_masks_complete_shifted_current_by_direction(self):
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
        particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.5, 0.25, -0.125))
        J_template = empty_tiled_vector(static_parameters, dynamic_parameters)

        full_current = GR_direct_deposition(
            particles,
            species,
            J_template,
            metric,
            static_parameters,
            dynamic_parameters,
        )
        masked_current = GR_direct_deposition(
            particles,
            species._replace(update_x=jnp.asarray([[False, True, False]])),
            J_template,
            metric,
            static_parameters,
            dynamic_parameters,
        )
        disabled_current = GR_direct_deposition(
            particles,
            species._replace(update_x=jnp.zeros((1, 3), dtype=bool)),
            J_template,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        self.assertTrue(jnp.allclose(masked_current[0], 0.0))
        self.assertTrue(jnp.allclose(masked_current[1], full_current[1]))
        self.assertTrue(jnp.allclose(masked_current[2], 0.0))
        for component in disabled_current:
            self.assertTrue(jnp.allclose(component, 0.0))


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
                        shape_factor,
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


    def test_update_D_consumes_densitized_current_without_metric_rescaling(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=4,
            Ny=4,
            Nz=1,
            x_wind=1.0,
            y_wind=0.8,
            z_wind=1.0,
            x_min=2.0,
            y_min=0.17,
            z_min=0.2,
            dt=0.1,
            tile_shape=(4, 4, 1),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
            metric="flat_spherical",
        )
        metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        H = empty_tiled_vector(static_parameters, dynamic_parameters)
        J = _constant_tiled_vector(static_parameters, dynamic_parameters, (1.0, 0.0, 0.0))

        # D and J are native densities, so the source enters with no sqrt(gamma) factor.
        D_next = update_D(
            D,
            H,
            J,
            metric,
            static_parameters,
            dynamic_parameters,
            dynamic_parameters.dt,
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
        self.assertTrue(jnp.allclose(
            D_next[0][active],
            -4.0 * jnp.pi * dynamic_parameters.dt,
        ))



class TestStaticMetricTimeLoop(StaticMetricTimeLoopFixtures, unittest.TestCase):
    def test_empty_storage_evolves_vacuum_without_particle_kernels(self):
        s, d = kernel_parameters(
            Nx=4, Ny=1, Nz=1, x_wind=4., tile_shape=(4, 1, 1), dt=.05,
            solver="static_metric", current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(s, d)
        zero = empty_tiled_vector(s, d)
        wave = jnp.broadcast_to(
            jnp.sin(jnp.arange(zero[1].shape[3]) * jnp.pi / 2)[None, None, None, :, None, None],
            zero[1].shape,
        )
        D = (zero[0], wave, zero[2])
        fields = (D, zero, zero, zero[0], zero[0], (zero, zero), metric,
                  (D, zero), jnp.asarray(True))
        # A neutral particle provides an independent zero-current reference
        # through the populated path, with nontrivial vacuum field evolution.
        reference_particles, species = _single_particle_state(s, d, (0., 0., 0.))
        species = species._replace(charge=jnp.zeros_like(species.charge))
        _, expected = time_loop_static_metric(reference_particles, species, fields, s, d)
        self.assertFalse(jnp.allclose(expected[1][2], zero[2]))
        stale_fields = fields[:2] + (tuple(jnp.ones_like(c) for c in zero),) + fields[3:]

        for scheme, nspecies, checked in itertools.product(
                ("GR_direct", "esirkepov"), (0, 1), (False, True)):
            with self.subTest(scheme=scheme, nspecies=nspecies, checked=checked):
                static = s._replace(current_deposition=scheme)
                particles = TiledParticles(
                    jnp.zeros((1, 1, 1, nspecies, 0, 3)),
                    jnp.zeros((1, 1, 1, nspecies, 0, 3)),
                    jnp.zeros((1, 1, 1, nspecies, 0), dtype=bool),
                )
                empty_species = SpeciesConfig(*(value[:nspecies] for value in species))
                transform = Mock(side_effect=AssertionError("vacuum current transform"))
                with ExitStack() as stack:
                    for name in ("hybrid_boris_geodesic_push", "GR_direct_deposition",
                                 "Esirkepov_current", "refresh_tiled_particle_tiles"):
                        stack.enter_context(patch(
                            "PyPIC3D.solvers.GR_yee.time_loop." + name,
                            side_effect=AssertionError("vacuum called " + name),
                        ))
                    result = jax.jit(lambda p, f: time_loop_static_metric(
                        p, empty_species, f, static, d,
                        return_diagnostics=checked, current_transform=transform,
                    ))(particles, stale_fields)
                if checked:
                    new, updated, diagnostics = result
                    self.assertEqual(diagnostics.absorbed_count.shape, (2, nspecies))
                    for value in diagnostics:
                        self.assertTrue(jnp.all(value == 0))
                else:
                    new, updated = result
                for actual, original in zip(new, particles):
                    self.assertTrue(jnp.array_equal(actual, original))
                for actual_vector, expected_vector in zip(updated[:3], expected[:3]):
                    for actual, reference in zip(actual_vector, expected_vector):
                        self.assertAllClose(actual, reference, atol=1e-12)
                self.assertTrue(bool(updated[-1]))
                transform.assert_not_called()

    def test_particle_boundaries_and_prior_overflow_for_both_schemes(self):
        for scheme, bc in itertools.product(('GR_direct', 'esirkepov'), (1, 2)):
            with self.subTest(scheme=scheme, boundary=bc):
                s, d = kernel_parameters(
                    Nx=8, Ny=1, Nz=1, x_wind=8., tile_shape=(8, 1, 1), dt=.2,
                    solver='static_metric', particle_pusher='hybrid_boris_geodesic',
                    current_deposition=scheme, particle_boundary_conditions=(bc, 0, 0))
                metric = initialize_flat_cartesian_metric(s, d)
                x = jnp.array([[[[[[3.98, 0., 0.]]]]]])
                u = jnp.array([[[[[[1., 0., 0.]]]]]])
                particles = shard_tiled_particles(
                    TiledParticles(x, u, jnp.ones(x.shape[:-1], dtype=bool)), s)
                species = SpeciesConfig(jnp.zeros(1), jnp.ones(1), jnp.ones(1),
                                        jnp.ones((1, 3), dtype=bool))
                zero = empty_tiled_vector(s, d)
                fields = densitize_fields((zero, zero, zero, jnp.zeros_like(zero[0]),
                          jnp.zeros_like(zero[0]), (zero, zero), metric,
                          (zero, zero), jnp.asarray(True)))
                new, updated = jax.jit(lambda p, f: time_loop_static_metric(
                    p, species, f, s, d))(particles, fields)
                self.assertTrue(bool(updated[-1]))  # Previous overflow remains sticky.
                self.assertEqual(int(new.active.sum()), 1 if bc == 1 else 0)
                if bc == 1:
                    self.assertLess(float(new.x[..., 0].reshape(-1)[0]), 4.)
                    self.assertLess(float(new.u[..., 0].reshape(-1)[0]), 0.)


    def test_static_metric_time_loop_keeps_metric_state_tail(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=4,
            Ny=4,
            Nz=4,
            x_wind=4.0,
            y_wind=4.0,
            z_wind=4.0,
            dt=0.05,
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.2, 0.0, 0.0))
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)
        J = empty_tiled_vector(static_parameters, dynamic_parameters)
        rho = jnp.zeros_like(J[0])
        phi = jnp.zeros_like(J[0])
        external_fields = (D, B)
        static_metric_state = (D, B)
        fields = densitize_fields((D, B, J, rho, phi, external_fields, metric, static_metric_state, jnp.asarray(False)))

        particles, fields = time_loop_static_metric(
            particles,
            species,
            fields,
            static_parameters,
            dynamic_parameters,
        )

        self.assertEqual(len(fields), 9)
        self.assertIs(fields[6], metric)
        self.assertEqual(len(fields[7]), 2)
        self.assertFalse(bool(jax.device_get(fields[-1])))
        self.assertTrue(jnp.all(jnp.isfinite(particles.x)))
        self.assertTrue(jnp.all(jnp.isfinite(particles.u)))


    def test_static_metric_time_loop_accepts_empty_particle_storage(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            Nx=4,
            Ny=4,
            Nz=1,
            x_wind=4.0,
            y_wind=4.0,
            z_wind=1.0,
            dt=0.05,
            tile_shape=(4, 4, 1),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 0, 0, 3)),
            u=jnp.zeros((1, 1, 1, 0, 0, 3)),
            active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
        )
        species = SpeciesConfig(
            charge=jnp.zeros((0,)),
            mass=jnp.zeros((0,)),
            weight=jnp.zeros((0,)),
            update_x=jnp.zeros((0, 3), dtype=bool),
        )
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)
        J = empty_tiled_vector(static_parameters, dynamic_parameters)
        rho = jnp.zeros_like(J[0])
        phi = jnp.zeros_like(J[0])
        external_fields = (D, B)
        static_metric_state = (D, B)
        fields = densitize_fields((D, B, J, rho, phi, external_fields, metric, static_metric_state, jnp.asarray(False)))

        particles, fields = time_loop_static_metric(
            particles,
            species,
            fields,
            static_parameters,
            dynamic_parameters,
        )

        D_next, B_next, J_next = fields[:3]
        self.assertEqual(particles.x.shape, (1, 1, 1, 0, 0, 3))
        self.assertEqual(len(fields), 9)
        self.assertFalse(bool(jax.device_get(fields[-1])))
        for vector in (D_next, B_next, J_next):
            for component in vector:
                self.assertTrue(jnp.all(jnp.isfinite(component)))



class TestStaticMetricDispatch(StaticMetricDispatchFixtures, unittest.TestCase):
    def test_static_metric_dispatch_contract_accepts_hybrid_gr_direct_path(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        static_config = {
            "solver": "static_metric",
            "current_calculation": "GR_direct_deposition",
            "particle_pusher": "hybrid_boris_geodesic",
            "filter_j": "none",
            "particle_tile_nx": static_parameters.tile_shape[0],
            "particle_tile_ny": static_parameters.tile_shape[1],
            "particle_tile_nz": static_parameters.tile_shape[2],
        }
        dynamic_config = {
            "Nx": int(dynamic_parameters.Nx),
            "Ny": int(dynamic_parameters.Ny),
            "Nz": int(dynamic_parameters.Nz),
        }

        validate_field_solver("static_metric")

        self.assertEqual(_encode_current_calculation("GR_direct_deposition"), "GR_direct")
        _validate_tiled_yee_configuration(static_config, dynamic_config)



if __name__ == "__main__":
    unittest.main()
