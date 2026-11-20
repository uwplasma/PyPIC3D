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


class TestGRMatchedFilter(unittest.TestCase):
    def assertClose(self, actual, expected, atol=2.e-12, rtol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=atol, rtol=rtol)),
                        f"max difference = {float(jnp.max(jnp.abs(actual - expected)))}")

    def test_fourier_response_and_constant_mode(self):
        for shape in ((16, 1, 1), (16, 8, 1), (16, 8, 4)):
            indices = jnp.meshgrid(*(jnp.arange(n) for n in shape), indexing="ij")
            wave = tuple(2 * jnp.pi / n if n > 1 else 0. for n in shape)
            mode = jnp.cos(sum(k * x for k, x in zip(wave, indices)))
            for choice in ("bilinear", "digital"):
                with self.subTest(shape=shape, filter=choice):
                    static, dynamic = parameters(choice, Nx=shape[0], Ny=shape[1], Nz=shape[2])
                    response = (jnp.prod(jnp.cos(jnp.array(wave) / 2) ** 2)
                                if choice == "bilinear" else
                                dynamic.alpha + (1 - dynamic.alpha) * sum(jnp.cos(k) for k in wave) / 3)
                    self.assertClose(filter_gr_grid(mode, static, dynamic), response * mode)
                    self.assertClose(filter_gr_grid(jnp.ones(shape), static, dynamic), jnp.ones(shape))
        static, dynamic = parameters()
        checkerboard = (-1.) ** jnp.arange(8)[:, None, None]
        self.assertClose(filter_gr_grid(checkerboard, static, dynamic), jnp.zeros_like(checkerboard))

    def test_scalar_vector_tensor_sum_and_adjoint(self):
        keys = jax.random.split(jax.random.PRNGKey(73), 6)
        for leading, key_a, key_b in zip(((), (3,), (3, 3)), keys[:3], keys[3:]):
            a = jax.random.normal(key_a, leading + (8, 4, 1), dtype=jnp.float64)
            b = jax.random.normal(key_b, a.shape, dtype=jnp.float64)
            for choice in ("bilinear", "digital"):
                static, dynamic = parameters(choice, Ny=4)
                fa = filter_gr_grid(a, static, dynamic)
                fb = filter_gr_grid(b, static, dynamic)
                self.assertClose(jnp.sum(fa, axis=(-3, -2, -1)), jnp.sum(a, axis=(-3, -2, -1)))
                self.assertClose(jnp.sum(a * fb), jnp.sum(fa * b))

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
