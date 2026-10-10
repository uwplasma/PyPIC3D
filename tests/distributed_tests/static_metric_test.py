"""Focused cross-device regression tests."""

from tests.support.static_metric_fixtures import (
    SpeciesConfig,
    StaticMetricTimeLoopFixtures,
    TiledParticles,
    densitize_fields,
    empty_tiled_vector,
    importlib,
    initialize_flat_cartesian_metric,
    itertools,
    jax,
    jnp,
    kernel_parameters,
    patch,
    shard_tiled_particles,
    time_loop_static_metric,
    unittest,
)


class TestStaticMetricTimeLoop(StaticMetricTimeLoopFixtures, unittest.TestCase):
    def test_static_metric_time_loop_migrates_only_required_particle_states(self):
        for scheme, checked in itertools.product(("GR_direct", "esirkepov"), (False, True)):
            with self.subTest(scheme=scheme, checked=checked):
                static_parameters, dynamic_parameters = kernel_parameters(
                    guard_cells=3,
                    Nx=8,
                    Ny=1,
                    Nz=1,
                    x_wind=8.0,
                    y_wind=1.0,
                    z_wind=1.0,
                    dt=0.2,
                    tile_shape=(4, 1, 1),
                    solver="static_metric",
                    current_deposition=scheme,
                    particle_pusher="hybrid_boris_geodesic",
                )
                metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
                x = jnp.zeros((2, 1, 1, 1, 2, 3))
                u = jnp.zeros_like(x)
                active = jnp.zeros((2, 1, 1, 1, 2), dtype=bool)
                x = x.at[0, 0, 0, 0, 0].set(jnp.asarray((-0.02, 0.0, 0.0)))
                u = u.at[0, 0, 0, 0, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
                active = active.at[0, 0, 0, 0, 0].set(True)
                particles = shard_tiled_particles(
                    TiledParticles(x=x, u=u, active=active),
                    static_parameters,
                )
                species = SpeciesConfig(
                    charge=jnp.asarray([1.0]),
                    mass=jnp.asarray([1.0]),
                    weight=jnp.asarray([1.0]),
                    update_x=jnp.asarray([[True, True, True]]),
                )
                D = empty_tiled_vector(static_parameters, dynamic_parameters)
                B = empty_tiled_vector(static_parameters, dynamic_parameters)
                J = empty_tiled_vector(static_parameters, dynamic_parameters)
                rho = jnp.zeros_like(J[0])
                phi = jnp.zeros_like(J[0])
                fields = densitize_fields((D, B, J, rho, phi, (D, B), metric, (D, B), jnp.asarray(False)))

                module = importlib.import_module('PyPIC3D.solvers.GR_yee.time_loop')
                with patch.object(module, 'refresh_tiled_particle_tiles',
                                  wraps=module.refresh_tiled_particle_tiles) as refresh:
                    if checked:
                        errors, (particles, fields) = jax.jit(
                            lambda p, f: time_loop_static_metric(
                                p, species, f, static_parameters, dynamic_parameters, return_errors=True
                            )
                        )(particles, fields)
                        errors.throw()
                    else:
                        particles, fields = jax.jit(lambda p, f: time_loop_static_metric(
                            p, species, f, static_parameters, dynamic_parameters
                        ))(particles, fields)

                    self.assertEqual(refresh.call_count, 2 if scheme == 'GR_direct' else 1)

                self.assertEqual(int(jnp.sum(particles.active[0, 0, 0])), 0)
                self.assertEqual(int(jnp.sum(particles.active[1, 0, 0])), 1)
                self.assertGreater(particles.x[1, 0, 0, 0, 0, 0], 0.0)
                self.assertTrue(jnp.any(jnp.abs(fields[2][0][1, 0, 0]) > 0.0))
                self.assertFalse(bool(fields[-1]))


    def test_static_metric_time_loop_reports_particle_refresh_overflow(self):
        for scheme, midpoint_only in itertools.product(
                ("GR_direct", "esirkepov"), (False, True)):
            with self.subTest(scheme=scheme, midpoint_only=midpoint_only):
                static_parameters, dynamic_parameters = kernel_parameters(
                    guard_cells=3,
                    Nx=8,
                    Ny=1,
                    Nz=1,
                    x_wind=8.0,
                    y_wind=1.0,
                    z_wind=1.0,
                    dt=0.2,
                    tile_shape=(4, 1, 1),
                    solver="static_metric",
                    current_deposition=scheme,
                    particle_pusher="hybrid_boris_geodesic",
                )
                metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
                x = jnp.zeros((2, 1, 1, 1, 1, 3))
                u = jnp.zeros_like(x)
                active = jnp.ones((2, 1, 1, 1, 1), dtype=bool)
                x = x.at[0, 0, 0, 0, 0].set(jnp.asarray((-0.02, 0.0, 0.0)))
                x = x.at[1, 0, 0, 0, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
                u = u.at[0, 0, 0, 0, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
                if midpoint_only:
                    # Both midpoints occupy the upper tile, but the endpoints
                    # swap tiles and fit. Only direct deposition needs those
                    # overflowing midpoint slots.
                    x = x.at[1, 0, 0, 0, 0, 0].set(.1)
                    u = u.at[1, 0, 0, 0, 0, 0].set(-1.)
                particles = shard_tiled_particles(
                    TiledParticles(x=x, u=u, active=active),
                    static_parameters,
                )
                species = SpeciesConfig(
                    charge=jnp.asarray([0.0]),
                    mass=jnp.asarray([1.0]),
                    weight=jnp.asarray([1.0]),
                    update_x=jnp.asarray([[True, True, True]]),
                )
                D = empty_tiled_vector(static_parameters, dynamic_parameters)
                B = empty_tiled_vector(static_parameters, dynamic_parameters)
                J = empty_tiled_vector(static_parameters, dynamic_parameters)
                rho = jnp.zeros_like(J[0])
                phi = jnp.zeros_like(J[0])
                fields = densitize_fields((D, B, J, rho, phi, (D, B), metric, (D, B), jnp.asarray(False)))

                particles, fields = jax.jit(lambda p, f: time_loop_static_metric(
                    p, species, f, static_parameters, dynamic_parameters
                ))(particles, fields)

                self.assertEqual(bool(fields[-1]), not midpoint_only or scheme == 'GR_direct')
                self.assertEqual(int(jnp.sum(particles.active)), 2 if midpoint_only else 1)


if __name__ == "__main__":
    unittest.main()
