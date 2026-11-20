"""One-tile API contracts."""

from contextlib import ExitStack
from unittest.mock import Mock, patch

from tests.support.static_metric_fixtures import (
    BC_TYPE_PARTICLE,
    GRDirectDepositionFixtures,
    GR_direct_deposition,
    HybridPusherFixtures,
    SpeciesConfig,
    StaticMetricDispatchFixtures,
    StaticMetricInitializationFixtures,
    StaticMetricTimeLoopFixtures,
    TiledParticles,
    _constant_tiled_vector,
    _encode_current_calculation,
    _replace_lapse_shift,
    _single_particle_state,
    _validate_tiled_yee_configuration,
    densitize_fields,
    empty_tiled_vector,
    hybrid_boris_geodesic_push,
    initialize_flat_cartesian_metric,
    initialize_flat_spherical_metric,
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
    itertools,
    jax,
    jnp,
    kernel_parameters,
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


class TestHybridPusher(HybridPusherFixtures, unittest.TestCase):
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
                for actual, original in zip(jax.tree.leaves(new), jax.tree.leaves(particles)):
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
