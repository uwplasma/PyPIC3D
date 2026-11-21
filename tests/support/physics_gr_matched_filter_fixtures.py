from tests.support.gr_matched_filter_runtime_fixtures import (
    Esirkepov_current,
    build_yee_metric,
    compute_rho,
    divergence,
    electromagnetic_constraints,
    empty_tiled_scalar,
    empty_tiled_vector,
    filter_gr_grid,
    initialize_flat_cartesian_metric,
    interpolate_metric,
    jax,
    jnp,
    parameters,
    particle_metric,
    particles_at,
    species,
    time_loop_static_metric,
    unittest,
    varying_metric,
)


class TestGRMatchedFilterFixtures:
    def assertClose(self, actual, expected, atol=2.e-12, rtol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=atol, rtol=rtol)),
                        f"max difference = {float(jnp.max(jnp.abs(actual - expected)))}")
    def continuity_case(self, shape_factor, choice):
        static, dynamic = parameters(choice, shape_factor=shape_factor)
        old = particles_at([7.95, 0., 0.])
        new = old._replace(x=old.x + jnp.array([.12, .03, -.02]))
        template = empty_tiled_vector(static, dynamic)
        raw = Esirkepov_current(old, new, species(), template, static, dynamic)
        current = tuple(filter_gr_grid(c, static, dynamic, tiled=True) for c in raw)
        rho0 = compute_rho(old, species(), template[0], static, dynamic)
        rho1 = compute_rho(new, species(), template[0], static, dynamic)
        active = (Ellipsis,) + (slice(3, -3),) * 3
        residual = ((rho1 - rho0) / dynamic.dt + divergence(current, dynamic))[active]
        scale = jnp.max(jnp.abs(divergence(current, dynamic)[active]))
        self.assertGreater(float(scale), 0.)
        self.assertLess(float(jnp.max(jnp.abs(residual)) / scale), 2.e-12)
        self.assertClose(jnp.sum(rho0[active]) * dynamic.dx * dynamic.dy * dynamic.dz, .7)
