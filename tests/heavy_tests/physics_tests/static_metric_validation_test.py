from tests.support.physics_static_metric_validation_fixtures import (
    D_FIELD_LOCATIONS,
    FIELD_CASES,
    GR_direct_deposition,
    PERIOD,
    PERIODIC_LOOP_CASES,
    PUSHER_CASES,
    TestCurrentDepositionConvergenceFixtures,
    TestFieldSolverConvergenceFixtures,
    TiledParticles,
    _advance_time_loop,
    _flat_cartesian_metric_at_position,
    _flat_cylindrical_metric_at_position,
    _flat_spherical_metric_at_position,
    build_center_metric,
    build_full_metric,
    compute_covariant_E,
    consistent_initial_field_state,
    contravariant_three_velocity,
    convergence_order,
    coordinate_lattice,
    densitize_vector,
    deposition_velocity_field,
    divergence,
    empty_particle_state,
    empty_tiled_scalar,
    empty_tiled_vector,
    evolve_vacuum,
    exact_source_velocity,
    generic_lapse_shift_metric_at_position,
    hamiltonian,
    hybrid_boris_geodesic_push,
    interior_rms,
    interpolate_metric,
    jax,
    jnp,
    kerr_cartesian_metric_at_position,
    kerr_spherical_metric_at_position,
    location_positions,
    math,
    one_particle,
    one_species,
    parameters,
    periodic_B_field,
    periodic_D_field,
    physical_fields,
    physical_vector,
    reference_trajectory,
    run_production_pusher,
    sample_center_metric,
    schwarzschild_cartesian_metric_at_position,
    tearDownModule,
    unittest,
    update_B,
)


class TestAnalyticMetricsAgainstClosedForm(unittest.TestCase):


    def test_flat_curvilinear_charts_reproduce_a_straight_line(self):
        """
        A force-free particle in flat space travels in a straight line.  Running
        the *production pusher* in cylindrical and spherical charts must return
        the image of that Cartesian straight line.
        """
        velocity = jnp.array([0.25, 0.30, -0.15])
        lorentz = 1.0 / math.sqrt(1.0 - float(velocity @ velocity))
        u_cartesian = lorentz * velocity
        start_cartesian = jnp.array([3.0, 0.55, 0.30])
        T = 1.2
        end_cartesian = start_cartesian + velocity * T

        def to_cylindrical(x):
            return jnp.array([jnp.hypot(x[0], x[1]), jnp.arctan2(x[1], x[0]), x[2]])

        def from_cylindrical(q):
            return jnp.array([q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1]), q[2]])

        def to_spherical(x):
            radius = jnp.linalg.norm(x)
            return jnp.array([radius, jnp.arccos(x[2] / radius), jnp.arctan2(x[1], x[0])])

        def from_spherical(q):
            return jnp.array(
                [
                    q[0] * jnp.sin(q[1]) * jnp.cos(q[2]),
                    q[0] * jnp.sin(q[1]) * jnp.sin(q[2]),
                    q[0] * jnp.cos(q[1]),
                ]
            )

        def lower_into_chart(inverse_map, q, u_cart):
            return jax.jacfwd(inverse_map)(q).T @ u_cart

        charts = (
            ("flat_cylindrical", _flat_cylindrical_metric_at_position, to_cylindrical, from_cylindrical),
            ("flat_spherical", _flat_spherical_metric_at_position, to_spherical, from_spherical),
        )
        for metric_name, metric_at_position, forward, inverse in charts:
            q0 = forward(start_cartesian)
            u0 = lower_into_chart(inverse, q0, u_cartesian)
            q_expected = forward(end_cartesian)
            u_expected = lower_into_chart(inverse, q_expected, u_cartesian)
            mins = tuple(float(q0[axis]) - 0.6 for axis in range(3))
            wind = (2.0, 1.6, 1.6)
            q_final, u_final = run_production_pusher(
                metric_at_position,
                tuple(float(value) for value in q0),
                tuple(float(value) for value in u0),
                T,
                T / 600.0,
                48,
                wind,
                mins,
                metric_name=metric_name,
            )
            position_error = float(jnp.linalg.norm(q_final - q_expected))
            momentum_error = float(jnp.linalg.norm(u_final - u_expected))
            # The residual is the O(h^2) grid sampling of the metric, not the
            # scheme; TestPusherConvergence measures that order directly.
            self.assertLess(position_error, 1.0e-4, f"{metric_name}: {position_error}")
            self.assertLess(momentum_error, 1.0e-4, f"{metric_name}: {momentum_error}")

    def test_schwarzschild_and_kerr_circular_orbits_have_the_right_frequency(self):
        """
        A Kerr equatorial circular orbit at ``r`` has ``Omega = M^(1/2)/(r^(3/2) + a M^(1/2))``
        in Kerr-Schild time.  The production pusher must keep ``r`` fixed and
        return to the starting azimuth after ``2 pi / Omega``.
        """
        mass, radius = 1.0, 8.0
        for spin in (0.0, 0.9):
            root = math.sqrt(mass * radius)
            denominator = radius * math.sqrt(radius**2 - 3.0 * mass * radius + 2.0 * spin * root)
            energy = (radius**2 - 2.0 * mass * radius + spin * root) / denominator
            angular_momentum = root * (radius**2 - 2.0 * spin * root + spin**2) / denominator
            delta = radius**2 - 2.0 * mass * radius + spin**2
            u_r = (2.0 * mass * radius * energy - spin * angular_momentum) / delta
            omega = math.sqrt(mass) / (radius**1.5 + spin * math.sqrt(mass))
            period = 2.0 * math.pi / omega

            x_final, u_final = run_production_pusher(
                kerr_spherical_metric_at_position(spin),
                (radius, 0.5 * math.pi, 0.0),
                (u_r, 0.0, angular_momentum),
                0.98 * period,
                0.05,
                48,
                (6.0, 0.8, 7.3),
                (5.0, 1.15, -0.3),
                metric_name="kerr_schild_spherical",
            )
            radius_drift = abs(float(x_final[0]) - radius) / radius
            theta_drift = abs(float(x_final[1]) - 0.5 * math.pi)
            # run_production_pusher rounds the step count; compare at that
            # actual time, not the unachieved requested endpoint.
            actual_time = round((0.98 * period) / 0.05) * 0.05
            azimuth_error = abs(float(x_final[2]) - omega * actual_time)
            momentum_error = abs(float(u_final[2]) - angular_momentum) / abs(angular_momentum)
            self.assertLess(radius_drift, 2.0e-3, f"spin={spin} radius drift {radius_drift}")
            self.assertLess(theta_drift, 1.0e-4, f"spin={spin} theta drift {theta_drift}")
            self.assertLess(azimuth_error, 5.0e-3, f"spin={spin} azimuth error {azimuth_error}")
            self.assertLess(momentum_error, 1.0e-12, f"spin={spin} u_phi drift {momentum_error}")


class TestPusherConvergence(unittest.TestCase):
    SPATIAL_LEVELS = (16, 32, 64)
    TEMPORAL_STEPS = (200, 400, 800, 1600)

    def test_pusher_accuracy_for_exactly_reconstructed_cylindrical_metric(self):
        """Cylindrical gamma is quadratic: spatial refinement has no error to remove."""
        case = PUSHER_CASES[0]
        reference = reference_trajectory(case["metric"], case["x0"], case["u0"], case["T"])
        positions = []
        for N in (16, 32):
            x, _ = run_production_pusher(
                case["metric"], case["x0"], case["u0"], case["T"], case["T"]/4000,
                N, case["wind"], case["mins"], metric_name=case["metric_name"])
            self.assertLess(float(jnp.linalg.norm(x-reference[:3])), 2e-9)
            positions.append(x)
        self.assertLess(float(jnp.linalg.norm(positions[0]-positions[1])), 1e-12)

    def test_pusher_spatial_convergence_above_temporal_and_reference_floors(self):
        """
        Hermite values are generally third order and derivatives second order.
        Integrated trajectory errors can converge faster through cancellation;
        require at least second order without imposing a false upper bound.
        Explicit dt and RK4 refinement keep the finest spatial error resolved.
        """
        report = []
        for case in PUSHER_CASES[1:]:
            reference = reference_trajectory(
                case["metric"],
                case["x0"],
                case["u0"],
                case["T"],
                case.get("charge", 0.0),
                case.get("D_values", (0, 0, 0)),
                case.get("B_values", (0, 0, 0)),
                n=20000,
            )
            # Spherical trajectory cancellation resolves a much smaller spatial error.
            steps = 16000 if case["metric_name"] == "flat_spherical" else 4000
            dt = case["T"] / steps
            errors = []
            for N in self.SPATIAL_LEVELS:
                x_final, _u_final = run_production_pusher(
                    case["metric"],
                    case["x0"],
                    case["u0"],
                    case["T"],
                    dt,
                    N,
                    case["wind"],
                    case["mins"],
                    metric_name=case["metric_name"],
                    charge=case.get("charge", 0.0),
                    D_values=case.get("D_values", (0, 0, 0)),
                    B_values=case.get("B_values", (0, 0, 0)),
                )
                errors.append(float(jnp.linalg.norm(x_final - reference[:3])))
            refined, _ = run_production_pusher(
                case["metric"], case["x0"], case["u0"], case["T"], dt/2,
                self.SPATIAL_LEVELS[-1], case["wind"], case["mins"],
                metric_name=case["metric_name"], charge=case.get("charge", 0.0),
                D_values=case.get("D_values", (0, 0, 0)),
                B_values=case.get("B_values", (0, 0, 0)))
            coarse_reference = reference_trajectory(
                case["metric"], case["x0"], case["u0"], case["T"],
                case.get("charge", 0.0), case.get("D_values", (0, 0, 0)),
                case.get("B_values", (0, 0, 0)), n=10000)
            temporal_delta = float(jnp.linalg.norm(refined-x_final))
            reference_delta = float(jnp.linalg.norm(coarse_reference[:3]-reference[:3]))
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            report.append(f"{case['name']}: errors={errors} orders={orders} "
                          f"temporal_delta={temporal_delta} reference_delta={reference_delta}")
            self.assertLess(temporal_delta, .05*errors[-1], "\n".join(report))
            self.assertLess(reference_delta, .01*errors[-1], "\n".join(report))
            for order in orders:
                self.assertGreater(order, 1.8, "\n".join(report))
            jax.clear_caches()

    def test_pusher_converges_second_order_in_time_for_every_chart(self):
        """
        Time-step self-convergence on a fixed reconstructed metric isolates
        Strang/Boris/leapfrog temporal truncation from spatial reconstruction.
        """
        report = []
        for case in PUSHER_CASES:
            N = 24
            solutions = []
            for steps in self.TEMPORAL_STEPS:
                solutions.append(
                    run_production_pusher(
                        case["metric"],
                        case["x0"],
                        case["u0"],
                        case["T"],
                        case["T"] / steps,
                        N,
                        case["wind"],
                        case["mins"],
                        metric_name=case["metric_name"],
                        charge=case.get("charge", 0.0),
                        D_values=case.get("D_values", (0, 0, 0)),
                        B_values=case.get("B_values", (0, 0, 0)),
                    )
                )
            finest = solutions[-1]
            errors = [
                float(jnp.linalg.norm(solution[0] - finest[0]))
                + float(jnp.linalg.norm(solution[1] - finest[1]))
                for solution in solutions[:-1]
            ]
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            report.append(f"{case['name']}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 1.7, "\n".join(report))

    def test_pusher_is_second_order_with_second_order_particle_shapes(self):
        """Second-order electromagnetic particle shapes retain GR spatial accuracy."""
        case = PUSHER_CASES[5]  # Gaussian-normal chart
        reference = reference_trajectory(case["metric"], case["x0"], case["u0"], case["T"])
        errors = []
        for N in self.SPATIAL_LEVELS:
            x_final, _u_final = run_production_pusher(
                case["metric"],
                case["x0"],
                case["u0"],
                case["T"],
                case["T"] / 4000.0,
                N,
                case["wind"],
                case["mins"],
                metric_name=case["metric_name"],
                shape_factor=2,
            )
            errors.append(float(jnp.linalg.norm(x_final - reference[:3])))
        orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
        for order in orders:
            self.assertGreater(order, 1.8, f"errors={errors} orders={orders}")


class TestFieldSolverConvergence(TestFieldSolverConvergenceFixtures, unittest.TestCase):


    def test_constitutive_relations_and_curl_updates_are_second_order(self):
        """
        ``E_i = alpha gamma_ij D^j + sqrt(gamma) eps_ijk beta^j B^k`` (FPIC Eq. 10),
        ``H_i = alpha gamma_ij B^j - sqrt(gamma) eps_ijk beta^j D^k`` (FPIC Eq. 9),
        ``dt B^i = -curl(E)^i / sqrt(gamma)`` (Eq. 8) and
        ``dt D^i = curl(H)^i / sqrt(gamma) - 4 pi J^i`` (Eq. 7).
        """
        report = []
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            measurements = [
                self._field_errors(metric_at_position, wind, mins, metric_name, N)
                for N in self.LEVELS
            ]
            for index, label in enumerate(("E_i", "H_i", "dB/dt", "dD/dt")):
                errors = [measurement[index] for measurement in measurements]
                if max(errors) < 1.0e-13:
                    continue  # exact for this chart (diagonal metric, zero shift)
                orders = [
                    convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
                ]
                report.append(f"{name} {label}: errors={errors} orders={orders}")
                for order in orders:
                    self.assertGreater(order, 1.85, "\n".join(report))
                    self.assertLess(order, 2.3, "\n".join(report))


class TestCurrentDepositionConvergence(TestCurrentDepositionConvergenceFixtures, unittest.TestCase):

    def test_deposited_current_converges_second_order_for_every_chart(self):
        report = []
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            errors = []
            for N in self.LEVELS:
                static_parameters, dynamic_parameters = parameters(
                    (N, N, N), wind, mins, 0.01, metric_name=metric_name
                )
                metric = build_full_metric(dynamic_parameters, metric_at_position)
                guard = int(static_parameters.guard_cells)
                positions = coordinate_lattice(mins, wind, N, self.PER_CELL)
                momenta = jax.vmap(deposition_velocity_field)(positions)
                count = positions.shape[0]
                particles = TiledParticles(
                    x=positions.reshape(1, 1, 1, 1, count, 3),
                    u=momenta.reshape(1, 1, 1, 1, count, 3),
                    active=jnp.ones((1, 1, 1, 1, count), dtype=bool),
                )
                J = physical_vector(GR_direct_deposition(
                    particles,
                    one_species(1.0),
                    empty_tiled_vector(static_parameters, dynamic_parameters),
                    metric,
                    static_parameters,
                    dynamic_parameters,
                ), metric.D)
                cell_volume = float(dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz)
                lattice_density = self.PER_CELL**3 / cell_volume
                component_errors = []
                for i in range(3):
                    grid_positions = location_positions(D_FIELD_LOCATIONS[i], dynamic_parameters)
                    shape = grid_positions.shape[:-1]
                    flat = grid_positions.reshape(-1, 3)
                    source = jax.vmap(
                        lambda point: exact_source_velocity(metric_at_position, point)
                    )(flat)[:, i].reshape(shape)
                    sqrt_gamma = jax.vmap(lambda point: metric_at_position(point)[4])(flat).reshape(shape)
                    exact = lattice_density * source / sqrt_gamma
                    component_errors.append(
                        interior_rms(J[i] - exact, guard, skip=3)
                        / interior_rms(exact, guard, skip=3)
                    )
                errors.append(max(component_errors))
            orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
            report.append(f"{name}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 1.55, "\n".join(report))

    def test_total_deposited_current_equals_the_single_particle_source(self):
        """
        The shape function is a partition of unity, so
        ``sum_g sqrt(gamma)_g J^i_g dV`` must equal ``q w (alpha v^i - beta^i)``
        *exactly*, with the metric taken where the kernel itself takes it -- from
        the grid.  Any deviation is a lost or double-counted contribution, not a
        discretisation error.
        """
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            N = 16
            static_parameters, dynamic_parameters = parameters(
                (N, N, N), wind, mins, 0.01, metric_name=metric_name
            )
            metric = build_full_metric(dynamic_parameters, metric_at_position)
            guard = int(static_parameters.guard_cells)
            position = jnp.asarray(
                [mins[axis] + (0.5137, 0.4231, 0.6117)[axis] * wind[axis] for axis in range(3)]
            )
            momentum = deposition_velocity_field(position)
            particles = TiledParticles(
                x=position.reshape(1, 1, 1, 1, 1, 3),
                u=momentum.reshape(1, 1, 1, 1, 1, 3),
                active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
            )
            J = GR_direct_deposition(
                particles,
                one_species(2.0, weight=3.0),
                empty_tiled_vector(static_parameters, dynamic_parameters),
                metric,
                static_parameters,
                dynamic_parameters,
            )
            cell_volume = dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
            window = (slice(None),) * 3 + (slice(guard, -guard),) * 3
            totals = jnp.asarray(
                [
                    jnp.sum(J[i][window]) * cell_volume
                    for i in range(3)
                ]
            )
            sampled = sample_center_metric(
                metric, position.reshape(1, 3), static_parameters, dynamic_parameters
            )
            velocity = contravariant_three_velocity(momentum.reshape(1, 3), sampled.gamma_inv)
            expected = 6.0 * (sampled.lapse[:, None] * velocity - sampled.shift)[0]
            error = float(jnp.max(jnp.abs(totals - expected)) / jnp.max(jnp.abs(expected)))
            self.assertLess(error, 1.0e-12, f"{name}: relative error {error}")


    def test_hermite_metric_values_and_derivatives_have_distinct_orders(self):
        """Smooth supplied data exercise lapse, shift and non-diagonal gamma in 3D."""
        provider = generic_lapse_shift_metric_at_position
        positions = jax.random.uniform(jax.random.PRNGKey(913), (400, 3), minval=-1.2, maxval=1.2)
        exact = jax.vmap(provider)(positions)
        gradients = jax.vmap(jax.jacfwd(lambda q: (provider(q)[0], provider(q)[1], provider(q)[3])))(positions)
        errors = {name: [] for name in ("lapse", "shift", "gamma", "inverse", "grad_lapse", "grad_shift", "grad_inverse")}
        for N in (12, 24, 48):
            s, d = parameters((N, N, N), (4., 4., 4.), (-2., -2., -2.), .01)
            m = build_center_metric(d, provider)
            tile = jax.tree.map(lambda a: a[0, 0, 0], m.center)
            grid = tuple(a[0, 0, 0] for a in d.grids.tiled_center_grid)
            sampled = interpolate_metric(tile, positions, grid, "numerical", (True, True, True), (3, 3, 3))
            computed = sampled[:4] + sampled[5:]
            expected = exact[:4] + (gradients[0], gradients[1], jnp.moveaxis(gradients[2], -1, -3))
            for name, a, b in zip(errors, computed, expected):
                errors[name].append(float(jnp.sqrt(jnp.mean((a-b)**2))))
        for name, values in errors.items():
            expected_order = 2 if name.startswith("grad_") else 3
            orders = [convergence_order(a, b) for a, b in zip(values, values[1:])]
            with self.subTest(quantity=name):
                for order in orders:
                    self.assertGreater(order, expected_order-.35, (values, orders))
                    self.assertLess(order, expected_order+.35, (values, orders))


class TestTimeLoopConvergence(unittest.TestCase):
    def test_flat_vacuum_plane_wave_is_second_order_in_space(self):
        """
        ``D^z = sin(k(x - t))``, ``B^y = -D^z`` is an exact vacuum solution of
        Eqs. (7)-(8) with ``alpha = 1``, ``beta = 0``.  Propagating it a full
        transit through the complete time loop pins down every sign in the
        Maxwell update.
        """
        L = 2.0 * math.pi
        T = 2.0 * math.pi
        errors = []
        for N in (16, 32, 64):
            dt = T / 512.0
            static_parameters, dynamic_parameters = parameters(
                (N, 4, 4), (L, L, L), (0.0, 0.0, 0.0), dt, metric_name="flat_cartesian"
            )
            metric = build_full_metric(dynamic_parameters, _flat_cartesian_metric_at_position)
            guard = int(static_parameters.guard_cells)

            def travelling_D(position, t):
                return jnp.array([0.0, 0.0, jnp.sin(position[0] - t)])

            def travelling_B(position, t):
                return jnp.array([0.0, -jnp.sin(position[0] - t), 0.0])

            fields = consistent_initial_field_state(
                static_parameters,
                dynamic_parameters,
                metric,
                lambda p: travelling_D(p, 0.0),
                lambda p: travelling_B(p, 0.0),
                dt,
            )
            particles, species = empty_particle_state()

            steps = int(round(T / dt))
            particles, fields = _advance_time_loop(
                particles, species, fields, static_parameters, dynamic_parameters, steps)
            positions = location_positions(D_FIELD_LOCATIONS[2], dynamic_parameters)
            exact = jnp.sin(positions[..., 0] - steps * dt)
            errors.append(interior_rms(physical_fields(fields)[0][2] - exact, guard))
        orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
        for order in orders:
            self.assertGreater(order, 1.85, f"errors={errors} orders={orders}")
            self.assertLess(order, 2.2, f"errors={errors} orders={orders}")

    def test_time_loop_is_second_order_in_time_for_curved_charts(self):
        report = []
        T = 1.0
        for name, metric_at_position, metric_name in PERIODIC_LOOP_CASES:
            solutions = []
            step_counts = (50, 100, 200, 400)
            for steps in step_counts:
                _sp, _dp, fields = evolve_vacuum(
                    metric_at_position,
                    metric_name,
                    16,
                    (PERIOD,) * 3,
                    (0.0,) * 3,
                    T / steps,
                    T,
                    periodic_D_field,
                    periodic_B_field,
                )
                solutions.append(fields[0])
            finest = solutions[-1]
            errors = [
                max(
                    interior_rms(solution[component] - finest[component], 2)
                    for component in range(3)
                )
                for solution in solutions[:-1]
            ]
            orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
            report.append(f"{name}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 1.85, "\n".join(report))


class TestInvariantsAndConstraints(unittest.TestCase):

    def test_magnetic_constraint_is_preserved_to_round_off(self):
        """
        ``d_i (sqrt(gamma)_i B^i)`` is the divergence the Yee update annihilates
        identically on the native densities; the metric weighting in ``E`` must
        not break that.
        """
        key = jax.random.PRNGKey(0)
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            N = 16
            static_parameters, dynamic_parameters = parameters(
                (N, N, N), wind, mins, 0.01, metric_name=metric_name
            )
            metric = build_full_metric(dynamic_parameters, metric_at_position)
            guard = int(static_parameters.guard_cells)
            shape = empty_tiled_scalar(static_parameters, dynamic_parameters).shape
            B_tiles = densitize_vector(tuple(
                jax.random.normal(jax.random.fold_in(key, i), shape) for i in range(3)
            ), metric.B)
            D_tiles = densitize_vector(tuple(
                jax.random.normal(jax.random.fold_in(key, 10 + i), shape) for i in range(3)
            ), metric.D)
            E_tiles = compute_covariant_E(D_tiles, B_tiles, metric)

            before = divergence(B_tiles, dynamic_parameters, forward=True)
            evolved = B_tiles
            for _ in range(20):
                evolved = update_B(
                    E_tiles, evolved, metric, static_parameters, dynamic_parameters,
                    dynamic_parameters.dt,
                )
            after = divergence(evolved, dynamic_parameters, forward=True)
            window = (slice(None),) * 3 + (slice(guard + 1, -guard - 1),) * 3
            drift = float(jnp.max(jnp.abs((after - before)[window])))
            scale = float(jnp.max(jnp.abs(before[window])))
            self.assertLess(drift / scale, 1.0e-11, f"{name}: drift={drift} scale={scale}")

    def test_geodesic_invariants_converge_second_order_in_grid_spacing(self):
        """
        ``H = alpha Gamma - beta^i u_i`` (stationarity) and ``L_z = x u_y - y u_x``
        (axisymmetry) must be conserved.  They are evaluated at matching times,
        ``(x^{n+1/2}, u^{n+1/2})``, because the pusher stores ``u`` on half steps.
        """
        cases = (
            (
                "Schwarzschild KS-cart",
                schwarzschild_cartesian_metric_at_position,
                "kerr_schild_cartesian",
                (10.0, 0.0, 0.3),
                (0.09539392014169457, 0.3651483716701107, 0.0),
                10.0,
                (8.0, 9.0, 4.0),
                (4.0, -2.0, -2.0),
            ),
            (
                "Kerr a=0.9 KS-cart",
                kerr_cartesian_metric_at_position(0.9),
                "kerr_schild_cartesian",
                (8.0, 0.0, 0.4),
                (0.10, 0.42, 0.02),
                10.0,
                (8.0, 9.0, 4.0),
                (1.5, -2.0, -2.0),
            ),
        )
        report = []
        for name, metric_at_position, metric_name, x0, u0, T, wind, mins in cases:
            state0 = jnp.concatenate((jnp.asarray(x0), jnp.asarray(u0)))
            H0 = float(hamiltonian(metric_at_position, state0))
            L0 = x0[0] * u0[1] - x0[1] * u0[0]
            energy_errors = []
            momentum_errors = []
            for N in (12, 24, 48):
                dt = 0.01
                static_parameters, dynamic_parameters = parameters(
                    (N, N, N), wind, mins, dt, metric_name=metric_name
                )
                metric = build_center_metric(dynamic_parameters, metric_at_position)
                D = empty_tiled_vector(static_parameters, dynamic_parameters)
                B = empty_tiled_vector(static_parameters, dynamic_parameters)
                species = one_species(0.0)
                started, _ = hybrid_boris_geodesic_push(
                    one_particle(x0, u0),
                    species,
                    D,
                    B,
                    metric,
                    static_parameters,
                    dynamic_parameters._replace(dt=jnp.asarray(-0.5 * dt)),
                )
                particles = one_particle(x0, started.u[0, 0, 0, 0, 0])

                def body(current, _):
                    advanced, centered = hybrid_boris_geodesic_push(
                        current, species, D, B, metric, static_parameters, dynamic_parameters
                    )
                    return advanced, (centered.x[0, 0, 0, 0, 0], centered.u[0, 0, 0, 0, 0])

                _final, (X, U) = jax.lax.scan(body, particles, None, length=int(round(T / dt)))
                H = jax.vmap(lambda s: hamiltonian(metric_at_position, s))(
                    jnp.concatenate((X, U), axis=1)
                )
                Lz = X[:, 0] * U[:, 1] - X[:, 1] * U[:, 0]
                energy_errors.append(float(jnp.max(jnp.abs(H - H0))) / abs(H0))
                momentum_errors.append(float(jnp.max(jnp.abs(Lz - L0))) / abs(L0))
            energy_orders = [
                convergence_order(energy_errors[i], energy_errors[i + 1])
                for i in range(len(energy_errors) - 1)
            ]
            momentum_orders = [
                convergence_order(momentum_errors[i], momentum_errors[i + 1])
                for i in range(len(momentum_errors) - 1)
            ]
            report.append(
                f"{name}: dH={energy_errors} p={energy_orders} dL={momentum_errors} p={momentum_orders}"
            )
            self.assertGreater(energy_orders[-1], 1.7, "\n".join(report))
            self.assertGreater(momentum_orders[-1], 1.7, "\n".join(report))


class TestLeapfrogVelocitySeed(unittest.TestCase):
    """
    ``seed_leapfrog_velocity`` is what makes the staggered scheme second order.
    Without it the initial state carries an O(dt) error that the pusher cannot
    recover from, and the measured rate collapses to one.
    """

    STEP_COUNTS = (200, 400, 800, 1600)

    def _final_positions(self, case, seeded, N):
        positions = []
        for steps in self.STEP_COUNTS:
            x_final, _u_final = run_production_pusher(
                case["metric"],
                case["x0"],
                case["u0"],
                case["T"],
                case["T"] / steps,
                N,
                case["wind"],
                case["mins"],
                metric_name=case["metric_name"],
                charge=case.get("charge", 0.0),
                D_values=case.get("D_values", (0, 0, 0)),
                B_values=case.get("B_values", (0, 0, 0)),
                offset_initial_half_step=seeded,
            )
            positions.append(x_final)
        return positions

    def test_seeding_restores_second_order_in_a_curved_chart(self):
        """
        Generic lapse+shift chart.  Time-step self-convergence on a fixed grid,
        so the O(h^2) metric-sampling floor cancels and only the effect of the
        initial state is left.
        """
        case = PUSHER_CASES[6]  # generic lapse+shift chart, charged, with a B field
        report = []
        rates = {}
        for seeded in (True, False):
            positions = self._final_positions(case, seeded, 32)
            finest = positions[-1]
            errors = [float(jnp.linalg.norm(x - finest)) for x in positions[:-1]]
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            rates[seeded] = (errors, orders)
            report.append(f"seeded={seeded}: errors={errors} orders={orders}")

        seeded_errors, seeded_orders = rates[True]
        unseeded_errors, unseeded_orders = rates[False]
        for order in seeded_orders:
            self.assertGreater(order, 1.8, "\n".join(report))
        # self-convergence against the finest run biases the estimator upwards,
        # so the unseeded rate is only required to stay clearly short of second
        # order -- the decisive statement is the size of the error itself
        self.assertLess(max(unseeded_orders), 1.8, "\n".join(report))
        self.assertGreater(
            unseeded_errors[0] / seeded_errors[0], 100.0, "\n".join(report)
        )


if __name__ == "__main__":
    unittest.main()
