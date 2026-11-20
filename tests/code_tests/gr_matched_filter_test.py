from tests.support.gr_matched_filter_runtime_fixtures import (
    _validate_current_filter_contract,
    build_yee_metric,
    empty_tiled_scalar,
    empty_tiled_vector,
    filter_gr_grid,
    hybrid_boris_geodesic_push,
    initialize_flat_cartesian_metric,
    initialize_simulation,
    jax,
    jnp,
    parameters,
    particle_metric,
    particles_at,
    species,
    time_loop_static_metric,
    unittest,
    validate_gr_filter,
    varying_metric,
)


class TestGRMatchedFilter(unittest.TestCase):
    def assertClose(self, actual, expected, atol=2.e-12, rtol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=atol, rtol=rtol)),
                        f"max difference = {float(jnp.max(jnp.abs(actual - expected)))}")

    def test_none_preserves_arrays_and_metric_exactly(self):
        static, dynamic = parameters("none")
        metric = build_yee_metric(dynamic, varying_metric)
        raw = jax.random.normal(jax.random.PRNGKey(4), (3, 3, 8, 1, 1), dtype=jnp.float64)
        self.assertTrue(bool(jnp.array_equal(filter_gr_grid(raw, static, dynamic), raw)))
        for actual, expected in zip(particle_metric(metric, static, dynamic), metric.center):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))

    def test_push_filters_physical_fields_after_volume_conversion(self):
        static, dynamic = parameters()
        metric = build_yee_metric(dynamic, varying_metric)
        physical = jnp.array([.3, -.2, .1])
        density = tuple(value * sample.sqrt_gamma for value, sample in zip(physical, metric.D))
        zero = empty_tiled_vector(static, dynamic)
        old = particles_at([2.37, 0., 0.])
        old = old._replace(u=jnp.broadcast_to(jnp.array([.1, -.07, .02]), old.u.shape))
        actual, _ = hybrid_boris_geodesic_push(old, species(), density, zero, metric,
                                               static, dynamic, advance_positions=False)
        # A constant physical field remains constant. Only the particle metric
        # changes; Maxwell's node volumes still convert the supplied density.
        reference_metric = metric._replace(center=particle_metric(metric, static, dynamic))
        expected, _ = hybrid_boris_geodesic_push(old, species(), density, zero, reference_metric,
                                                static._replace(current_filter="none"), dynamic,
                                                advance_positions=False)
        self.assertClose(actual.u, expected.u)
        wrong_order = filter_gr_grid(density[0], static, dynamic, tiled=True) / metric.D[0].sqrt_gamma
        self.assertGreater(float(jnp.max(jnp.abs(wrong_order - physical[0]))), 1.e-4)

    def test_config_acceptance_and_rejection(self):
        for solver in ("static_metric", "dynamic_metric"):
            for choice in ("bilinear", "digital"):
                config = dict(solver=solver, metric="numerical", filter_j=choice,
                              current_calculation="esirkepov")
                _validate_current_filter_contract(config)
                for bad in (dict(metric="flat_spherical"), dict(x_bc="conducting"),
                            dict(particle_y_bc="absorbing")):
                    with self.assertRaisesRegex(ValueError, "Cartesian|periodic"):
                        _validate_current_filter_contract(dict(config, **bad))
        for solver in ("electrodynamic_yee", "dark_matter_yee"):
            with self.assertRaisesRegex(ValueError, "use filter_j='none'"):
                _validate_current_filter_contract(dict(solver=solver, filter_j="bilinear",
                                                       current_calculation="esirkepov"))
        static, _ = parameters()
        for bad in (dict(metric="flat_spherical"), dict(boundary_conditions=(0, 1, 0)),
                    dict(particle_boundary_conditions=(2, 0, 0))):
            with self.assertRaisesRegex(ValueError, "Cartesian|periodic"):
                validate_gr_filter(static._replace(**bad))
        validate_gr_filter(static._replace(current_filter="none", metric="flat_spherical"))
        # Global periodic defaults cannot override contradictory per-species
        # declarations silently when selecting the matched periodic method.
        with self.assertRaisesRegex(ValueError, "periodic particle species"):
            initialize_simulation({
                "simulation_parameters": dict(solver="static_metric", filter_j="bilinear"),
                "particle1": dict(x_bc="absorbing"),
            })

    def test_current_transform_cannot_double_filter_the_current(self):
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice)
            metric = initialize_flat_cartesian_metric(static, dynamic)
            zero = empty_tiled_vector(static, dynamic)
            rho = empty_tiled_scalar(static, dynamic)
            fields = (zero, zero, zero, rho, rho, (zero, zero), metric, (zero, zero), False)
            with self.assertRaisesRegex(ValueError, "filter_j or current_transform, not both"):
                time_loop_static_metric(particles_at([2.3, 0., 0.]), species(), fields,
                                        static, dynamic, current_transform=lambda current: current)

    def test_filter_requires_all_six_periodic_bssn_faces(self):
        try:
            from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
        except ModuleNotFoundError as error:
            if error.name == "JAX_BSSN":
                self.skipTest("Optional JAX_BSSN dependency is not installed")
            raise

        faces = ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")
        periodic = dict.fromkeys(faces, 0)
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice, solver="dynamic_metric")
            build_bssn_parameters(static, dynamic, evolution_parameters=periodic)
            for face in faces:
                with self.subTest(filter=choice, face=face):
                    controls = dict(periodic, **{face: 1})
                    with self.assertRaisesRegex(ValueError, "all six BSSN boundaries"):
                        build_bssn_parameters(static, dynamic, evolution_parameters=controls)
                    # The existing unfiltered solver still permits independent
                    # gravitational boundaries, including Sommerfeld faces.
                    unfiltered, _ = build_bssn_parameters(static._replace(current_filter="none"),
                                                         dynamic, evolution_parameters=controls)
                    self.assertEqual(int(getattr(unfiltered.bssn_evolution_parameters, face)), 1)


if __name__ == "__main__":
    unittest.main()
