from tests.support.physics_gr_matched_filter_fixtures import (
    TestGRMatchedFilterFixtures,
    build_yee_metric,
    electromagnetic_constraints,
    empty_tiled_scalar,
    empty_tiled_vector,
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


class TestGRMatchedFilter(TestGRMatchedFilterFixtures, unittest.TestCase):


    def test_filtered_metric_and_hermite_derivatives_are_consistent(self):
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice)
            raw = build_yee_metric(dynamic, varying_metric)
            metric = particle_metric(raw, static, dynamic)
            self.assertClose(jnp.einsum("...ij,...jk->...ik", metric.gamma, metric.gamma_inv), jnp.eye(3))
            self.assertClose(metric.sqrt_gamma ** 2, jnp.linalg.det(metric.gamma))
            self.assertGreater(float(jnp.min(jnp.linalg.eigvalsh(metric.gamma))), 0.)
            self.assertGreater(float(jnp.max(jnp.abs(metric.lapse - raw.center.lapse))), 1.e-4)

            local = jax.tree.map(lambda a: a[0, 0, 0], metric)
            grid = tuple(a[0, 0, 0] for a in dynamic.grids.tiled_center_grid)
            position = jnp.array([2.37, 0., 0.])
            def sample(x, derivatives=False):
                return interpolate_metric(local, x, grid, "numerical", (True, False, False),
                                          (3, 3, 3), derivatives=derivatives)
            value = sample(position, True)
            h = 1.e-5
            plus, minus = sample(position + jnp.array([h, 0., 0.])), sample(position - jnp.array([h, 0., 0.]))
            self.assertClose(value.grad_lapse[0], (plus.lapse - minus.lapse) / (2 * h), atol=2.e-10)
            self.assertClose(value.grad_shift[:, 0], (plus.shift - minus.shift) / (2 * h), atol=2.e-10)
            self.assertClose(value.grad_gamma_inv[0], (plus.gamma_inv - minus.gamma_inv) / (2 * h), atol=2.e-10)
            self.assertClose(value.grad_lapse[1:], jnp.zeros(2))
            self.assertClose(value.grad_gamma_inv[1:], jnp.zeros((2, 3, 3)))


    def test_cic_tsc_filtered_continuity_at_periodic_seam(self):
        for shape in (1, 2):
            for choice in ("bilinear", "digital"):
                with self.subTest(shape=shape, filter=choice):
                    self.continuity_case(shape, choice)

    def test_native_step_preserves_filtered_gauss_residual(self):
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice, shape_factor=2)
            metric = initialize_flat_cartesian_metric(static, dynamic)
            old = particles_at([7.999, 0., 0.])
            old = old._replace(u=jnp.broadcast_to(jnp.array([.2, .03, -.01]), old.u.shape))
            zero = empty_tiled_vector(static, dynamic)
            rho = empty_tiled_scalar(static, dynamic)
            fields = (zero, zero, zero, rho, rho, (zero, zero), metric, (zero, zero), False)
            # The initial nonzero residual is intentional: exact continuity
            # preserves it even when this particle crosses the periodic seam.
            before, _ = electromagnetic_constraints(old, species(), fields, static, dynamic)
            new, after = jax.jit(lambda p, f: time_loop_static_metric(p, species(), f, static, dynamic))(old, fields)
            final, magnetic = electromagnetic_constraints(new, species(), after, static, dynamic)
            active = (Ellipsis,) + (slice(3, -3),) * 3
            self.assertClose(final[active], before[active])
            self.assertClose(magnetic[active], jnp.zeros_like(magnetic[active]))
            self.assertFalse(bool(after[-1]))
            self.assertEqual(int(jnp.sum(old.active)), int(jnp.sum(new.active)))


if __name__ == "__main__":
    unittest.main()
