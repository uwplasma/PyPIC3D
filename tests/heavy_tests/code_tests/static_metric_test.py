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


if __name__ == "__main__":
    unittest.main()
