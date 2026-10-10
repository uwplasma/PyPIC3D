"""Single-device numerical tests."""

from tests.support.particle_metric_fixtures_shared import (
    CheckedSamplingFixtures,
    GR_direct_deposition,
    HermiteFixtures,
    ParticleMetricConsumersFixtures,
    ParticleMetricFixtures,
    SpeciesConfig,
    TiledParticles,
    build_static_parameters,
    checkify,
    consumer_runtime,
    densitize_vector,
    hybrid_boris_geodesic_push,
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
    interpolate_hermite,
    interpolate_metric,
    jax,
    jnp,
    kernel_parameters,
    make_runtime,
    manufactured,
    np,
    partial,
    particle_metric_valid,
    positive_definite_3x3,
    sample_metric,
    sampled_rotation_checks,
    seed_leapfrog_velocity,
    static_parameters_for_output,
    unittest,
)


class TestHermite(HermiteFixtures, unittest.TestCase):
    def test_tensor_quadratic_reproduction(self):
        grid = tuple(jnp.arange(-3, 14, dtype=float) * h for h in (0.1, 0.13, 0.17))

        def f(p):
            x, y, z = jnp.moveaxis(p, -1, 0)
            return (
                1
                + 0.3 * x
                - 0.2 * y
                + 0.1 * z
                + x * x
                + 2 * y * y
                + 0.4 * z * z
                + 0.7 * x * y * z
                + 0.8 * x * x * y * y
            )

        values = f(jnp.stack(jnp.meshgrid(*grid, indexing="ij"), axis=-1))
        q = jnp.array([[0.247, 0.381, 0.418], [0.41, 0.53, 0.67]])
        get = lambda q: interpolate_hermite(values, q, grid, (True, True, True), (3, 3, 3))
        np.testing.assert_allclose(get(q), f(q), atol=3e-14, rtol=3e-14)
        for k in range(3):
            direction = jnp.broadcast_to(jnp.eye(3)[k], q.shape)
            np.testing.assert_allclose(
                jax.jvp(get, (q,), (direction,))[1],
                jax.jvp(f, (q,), (direction,))[1],
                atol=1e-12,
            )


    def test_nodal_values_and_shared_slopes(self):
        x = jnp.arange(-3, 13, dtype=float) * 0.125
        grid = (x, jnp.arange(7, dtype=float), jnp.arange(7, dtype=float))
        f = jnp.sin(x) + 0.3 * jnp.cos(3 * x)
        data = jnp.broadcast_to(f[:, None, None], (len(x), 7, 7))
        get = lambda t: interpolate_hermite(
            data, jnp.array([t, 3.0, 3.0]), grid, (True, False, False), (3, 3, 3)
        )
        derivative = jax.grad(get)
        for i in range(2, len(x) - 3):
            self.assertAlmostEqual(float(get(x[i])), float(f[i]), places=13)
            expected = (f[i + 1] - f[i - 1]) / (2 * 0.125)
            self.assertLess(abs(float(derivative(x[i]) - expected)), 1e-12)
            self.assertLess(
                abs(float(derivative(x[i] - 1e-10) - derivative(x[i] + 1e-10))), 1e-8
            )


    def test_point_outside_stencil_is_nan(self):
        grid, m = manufactured()
        q = jnp.array([[-1.0, 0.3, 3.0]])
        a = interpolate_metric(m, q, grid, "numerical", (True, True, False), (3, 3, 3))
        self.assertTrue(np.isnan(np.asarray(a.gamma)).all())



class TestParticleMetric(ParticleMetricFixtures, unittest.TestCase):
    def test_shared_reconstruction_for_every_supported_chart(self):
        providers = (
            ("flat_cartesian", initialize_flat_cartesian_metric),
            ("flat_cylindrical", initialize_flat_cylindrical_metric),
            ("flat_spherical", initialize_flat_spherical_metric),
            ("kerr_schild_cartesian", partial(initialize_kerr_schild_cartesian_metric, spin=.4)),
            ("kerr_schild_spherical", partial(initialize_kerr_schild_spherical_metric, spin=.4)),
        )
        for name, initialize in providers:
            with self.subTest(metric=name):
                s, d = kernel_parameters(
                    Nx=8, Ny=8, Nz=1, x_min=2., y_min=.6, z_min=0.,
                    x_wind=1., y_wind=.8, z_wind=1., solver="static_metric",
                    particle_pusher="hybrid_boris_geodesic", metric=name, metric_spin=.4,
                )
                m = initialize(s, d)
                tile = jax.tree.map(lambda a: a[0, 0, 0], m.center)
                grid = tuple(a[0, 0, 0] for a in d.grids.tiled_center_grid)
                q = jnp.array([[2.31, .83, .2], [2.67, 1.17, .8]])
                get = lambda metric, derivatives=True: interpolate_metric(
                    metric, q, grid, name, (True, True, False), (3, 3, 3),
                    derivatives=derivatives)
                original = get(tile)
                poisoned = tile._replace(gamma_inv=jnp.full_like(tile.gamma_inv, jnp.nan),
                                         sqrt_gamma=jnp.full_like(tile.sqrt_gamma, jnp.nan))
                for a, b in zip(original, get(poisoned)):
                    np.testing.assert_array_equal(a, b)
                values = get(tile, False)
                for a, b in zip(original[:5], values[:5]):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(values[5:], (None, None, None))
                np.testing.assert_allclose(original.gamma @ original.gamma_inv,
                                           jnp.broadcast_to(jnp.eye(3), original.gamma.shape), atol=1e-12)
                np.testing.assert_allclose(original.sqrt_gamma**2, jnp.linalg.det(original.gamma), rtol=1e-12)
                np.testing.assert_array_equal(original.grad_lapse[..., 2], 0.)
                np.testing.assert_array_equal(original.grad_shift[..., 2], 0.)
                np.testing.assert_array_equal(original.grad_gamma_inv[..., 2, :, :], 0.)
                # A chart name must never override supplied grid primitives.
                changed = get(tile._replace(lapse=tile.lapse*.7, shift=tile.shift+.1,
                                            gamma=tile.gamma*2.))
                np.testing.assert_allclose(changed.lapse, original.lapse*.7, atol=1e-12)
                np.testing.assert_allclose(changed.shift, original.shift+.1, atol=1e-12)
                np.testing.assert_allclose(changed.gamma_inv, original.gamma_inv/2., atol=1e-12)


    def test_inverse_and_derivatives(self):
        grid, m = manufactured()
        q = jnp.array([[0.247, 0.381, 3.0], [0.415, 0.274, 3.0]])
        evaluate = lambda q: interpolate_metric(m, q, grid, "numerical", (True, True, False), (3, 3, 3))
        a = evaluate(q)
        dh = a.grad_gamma_inv
        np.testing.assert_allclose(
            a.gamma @ a.gamma_inv, jnp.broadcast_to(jnp.eye(3), a.gamma.shape), atol=1e-12
        )
        np.testing.assert_allclose(a.sqrt_gamma**2, jnp.linalg.det(a.gamma), rtol=1e-12)
        np.testing.assert_allclose(
            jnp.linalg.det(a.gamma) * jnp.linalg.det(a.gamma_inv), 1.0, atol=1e-12
        )
        for k in range(3):
            tangent = jnp.broadcast_to(jnp.eye(3)[k], q.shape)
            actual = jax.jvp(lambda x: evaluate(x).gamma_inv, (q,), (tangent,))[1]
            np.testing.assert_allclose(dh[:, k], actual, atol=1e-12, rtol=1e-12)
            eps = 1e-5
            plus = evaluate(q + eps * tangent)
            minus = evaluate(q - eps * tangent)
            np.testing.assert_allclose(
                dh[:, k], (plus.gamma_inv - minus.gamma_inv) / (2 * eps), atol=1e-10, rtol=1e-8
            )
            np.testing.assert_allclose(
                a.grad_lapse[:, k], (plus.lapse - minus.lapse) / (2 * eps), atol=1e-10
            )
            np.testing.assert_allclose(
                a.grad_shift[:, :, k], (plus.shift - minus.shift) / (2 * eps), atol=1e-10
            )
        np.testing.assert_array_equal(dh[:, 2], 0.0)
        self.assertTrue(np.asarray(particle_metric_valid(a, q, "numerical")).all())


    def test_adjacent_radial_tiles_agree(self):
        g1, m1 = manufactured()
        g2, m2 = manufactured(0.5)
        q = jnp.array([[0.499, 0.381, 3.0], [0.501, 0.381, 3.0]])
        a = interpolate_metric(m1, q, g1, "numerical", (True, True, False), (3, 3, 3))
        b = interpolate_metric(m2, q, g2, "numerical", (True, True, False), (3, 3, 3))
        for v, w in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(v, w, rtol=1e-12, atol=1e-12)


    def test_unresolved_axes_use_fixed_nodes_and_zero_derivatives(self):
        grid, metric = manufactured()
        q = jnp.array([[.247, .381, 3.], [.415, .274, 3.]])
        sample = lambda p: interpolate_metric(
            metric, p, grid, "numerical", (True, False, False), (3, 3, 3))
        a = sample(q)
        b = sample(q.at[:, 1:].set(100.))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a.grad_lapse[..., 1:], 0.)
        np.testing.assert_array_equal(a.grad_shift[..., 1:], 0.)
        np.testing.assert_array_equal(a.grad_gamma_inv[..., 1:, :, :], 0.)


    def test_stored_inverse_and_determinant_do_not_enter_particle_geometry(self):
        s, d, m, D, B = make_runtime("spherical", 16, 32)
        q = jnp.array(
            [
                [2.37, .4 + 0.05 * d.dy, 0.1],
                [2.37, .4 + 0.5 * d.dy, 0.1],
                [2.37, jnp.pi - .4 - 0.05 * d.dy, 0.1],
                [2.37, jnp.pi - .4 - 0.5 * d.dy, 0.1],
            ]
        )
        a = sample_metric(q, m, s, d)
        np.testing.assert_allclose(a.gamma_inv[:, 0, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(a.grad_gamma_inv[:, :, 0, 0], 0.0, atol=1e-12)
        # Only lapse, shift and gamma are interpolated; the grid inverse and
        # determinant never enter the particle metric, even when poisoned.
        poisoned = m._replace(
            center=m.center._replace(
                gamma_inv=jnp.full_like(m.center.gamma_inv, jnp.nan),
                sqrt_gamma=jnp.full_like(m.center.sqrt_gamma, jnp.nan),
            )
        )
        for v, w in zip(jax.tree.leaves(a), jax.tree.leaves(sample_metric(q, poisoned, s, d))):
            np.testing.assert_allclose(v, w, atol=1e-12)
        changed = m._replace(
            center=m.center._replace(
                gamma=2 * m.center.gamma,
                lapse=0.7 * m.center.lapse,
                shift=m.center.shift + 0.1,
            )
        )
        b = sample_metric(q, changed, s, d)
        np.testing.assert_allclose(b.gamma_inv, a.gamma_inv / 2, atol=1e-12)
        np.testing.assert_allclose(b.lapse, 0.7 * a.lapse, atol=1e-12)
        np.testing.assert_allclose(b.shift, a.shift + 0.1, atol=1e-12)
        axes = q.at[:, 1].set(jnp.array([0.0, jnp.pi, 0.0, jnp.pi]))
        self.assertFalse(
            np.asarray(particle_metric_valid(sample_metric(axes, m, s, d), axes, s.metric)).any()
        )


    def test_sampled_rotation_invariants(self):
        for row in sampled_rotation_checks(32):
            for key in ("inverse_defect", "norm_error", "parallel_error", "reversal_error"):
                self.assertLess(row[key], 1e-12, (key, row))


    def test_signed_orientation_and_invalid_metrics(self):
        grid, m = manufactured()
        for chart, q in [
            ("flat_cylindrical", jnp.array([[-0.15, 0.25, 3.0]])),
            ("flat_spherical", jnp.array([[0.25, -0.15, 3.0]])),
        ]:
            a = interpolate_metric(m, q, grid, chart, (True, True, False), (3, 3, 3))
            self.assertLess(float(a.sqrt_gamma[0]), 0.0)
            np.testing.assert_allclose(a.sqrt_gamma**2, jnp.linalg.det(a.gamma), rtol=1e-12)
            self.assertTrue(bool(particle_metric_valid(a, q, chart)[0]))
        q = jnp.array([[0.25, 0.35, 3.0]])
        for tensor in (jnp.diag(jnp.array([-1.0, 1.0, 1.0])), jnp.zeros((3, 3))):
            bad = m._replace(gamma=jnp.broadcast_to(tensor, m.gamma.shape))
            a = interpolate_metric(bad, q, grid, "numerical", (True, True, False), (3, 3, 3))
            self.assertFalse(bool(particle_metric_valid(a, q, "numerical")[0]))


    def test_positive_definite_matches_eigenvalue_signs(self):
        rng = np.random.default_rng(519)
        q, _ = np.linalg.qr(rng.normal(size=(2000, 3, 3)))
        values = np.exp(rng.uniform(-4, 4, size=(2000, 3)))
        values[500:1000, 0] *= -1
        values[1000:1500, :2] *= -1
        values[1500:] *= -1
        tensors = np.einsum("nij,nj,nkj->nik", q, values, q)
        expected = np.linalg.eigvalsh(tensors).min(-1) > 0
        np.testing.assert_array_equal(np.asarray(jax.jit(positive_definite_3x3)(jnp.asarray(tensors))), expected)
        edge = jnp.array([np.diag([1.0, 1.0, 0.0]), np.diag([1.0, 1.0, 1e-20]),
                          np.diag([1.0, 1.0, -1e-20]), np.diag([1.0, np.nan, 1.0])])
        np.testing.assert_array_equal(positive_definite_3x3(edge), [False, True, False, False])
        # a singular non-diagonal tensor is rejected without a floor or repair
        self.assertFalse(bool(positive_definite_3x3(jnp.ones((3, 3)))))



class TestParticleMetricConsumers(ParticleMetricConsumersFixtures, unittest.TestCase):
    def test_pusher_and_direct_deposition_use_supplied_primitives(self):
        first_order_push = None
        for order in (1, 2):
            with self.subTest(shape_factor=order):
                s, d, m, D, B, p, species = consumer_runtime(order)
                # The push consumes native densities of the constant physical fields.
                D, B = densitize_vector(D, m.D), densitize_vector(B, m.B)
                # Inactive coordinates lie outside the supplied metric stencil.
                # Keep them finite: deposition's particle-shape weights use x too.
                p = p._replace(
                    x=jnp.concatenate((p.x, jnp.zeros_like(p.x[..., :1, :])), axis=-2),
                    u=jnp.concatenate((p.u, jnp.full_like(p.u[..., :1, :], jnp.nan)), axis=-2),
                    active=jnp.concatenate((p.active, jnp.zeros_like(p.active[..., :1])), axis=-1),
                )
                push = jax.jit(
                    lambda metric: hybrid_boris_geodesic_push(p, species, D, B, metric, s, d))
                deposit = jax.jit(
                    lambda metric: GR_direct_deposition(p, species, tuple(jnp.zeros_like(a) for a in D), metric, s, d))
                poisoned = m._replace(center=m.center._replace(
                    gamma_inv=jnp.full_like(m.center.gamma_inv, jnp.nan),
                    sqrt_gamma=jnp.full_like(m.center.sqrt_gamma, jnp.nan)))
                changed = m._replace(center=m.center._replace(
                    lapse=m.center.lapse*.7, shift=m.center.shift+.1, gamma=m.center.gamma*2.))
                for name, run in (("push", push), ("deposit", deposit)):
                    original = run(m)
                    same = run(poisoned)
                    self.assert_same_tree(same, original)
                    different = run(changed)
                    if name == "deposit":
                        self.assertTrue(all(bool(jnp.isfinite(a).all()) for a in original))
                    else:
                        for state in original:
                            self.assertTrue(bool(jnp.isfinite(state.x[..., :2, :]).all()))
                            self.assertTrue(bool(jnp.isfinite(state.u[..., :2, :]).all()))
                        # Constant EM fields remove shape-dependent gather error,
                        # so changing particle shape must not change the GR push.
                        if order == 1:
                            first_order_push = original
                        else:
                            self.assert_same_tree(original, first_order_push)
                    difference = max(float(jnp.max(jnp.abs(jnp.nan_to_num(a)-jnp.nan_to_num(b))))
                                     for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(different))
                                     if a.dtype != jnp.bool_)
                    self.assertGreater(difference, 1e-6)



class TestCheckedSampling(CheckedSamplingFixtures, unittest.TestCase):
    def test_checked_sampler_reports_stencil_and_tensor_errors(self):
        grid, m = manufactured()
        get = jax.jit(checkify.checkify(
            lambda q, metric: interpolate_metric(metric, q, grid, "numerical", (True, True, False), (3, 3, 3))
        ))
        q = jnp.array([[0.23, 0.35, 3.0]])
        errors, _ = get(q, m)
        errors.throw()
        for pos, metric in [
            (q.at[0, 0].set(-5.0), m),
            (q, m._replace(gamma=m.gamma.at[..., 0, 0].set(-1.0))),
            (q, m._replace(lapse=-m.lapse)),
        ]:
            errors, _ = get(pos, metric)
            with self.assertRaisesRegex(Exception, "particle metric.*tile=.*flattened species/slot"):
                errors.throw()


    def test_checked_pusher_and_seed_preserve_inactive_slots(self):
        s, d, m, D, B = make_runtime("spherical", 16, 32)
        D, B = densitize_vector(D, m.D), densitize_vector(B, m.B)
        x = jnp.array([[[[[[2.0, 0.7, 0.0], [0.0, 0.0, 0.0]]]]]])
        p = TiledParticles(x, jnp.zeros_like(x), jnp.array([[[[[True, False]]]]]))
        sp = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1), jnp.ones((1, 3), bool))
        seed = jax.jit(checkify.checkify(lambda p: seed_leapfrog_velocity(p, sp, D, B, s, d, m)))
        errors, seeded = seed(p)
        errors.throw()
        push = jax.jit(checkify.checkify(lambda p: hybrid_boris_geodesic_push(p, sp, D, B, m, s, d)))
        errors, (new, mid) = push(seeded)
        errors.throw()
        self.assertTrue(np.isfinite(np.asarray(new.x)).all())
        self.assertTrue(np.isfinite(np.asarray(new.u)).all())
        np.testing.assert_array_equal(new.x[..., 1, :], p.x[..., 1, :])
        np.testing.assert_array_equal(mid.x[..., 1, :], p.x[..., 1, :])
        # Invalid active metric samples must fail the functionalized pusher too.
        errors, _ = push(p._replace(x=p.x.at[..., 0, 1].set(0.0)))
        with self.assertRaisesRegex(Exception, "invalid particle sample"):
            errors.throw()
        far = d._replace(dt=jnp.asarray(100.0))
        errors, _ = checkify.checkify(
            lambda p: hybrid_boris_geodesic_push(p, sp, D, B, m, s, far)
        )(p._replace(u=p.u.at[..., 0, 0].set(10.0)))
        self.assertIsNotNone(errors.get())


    def test_guard_cell_default_minimum_and_metadata(self):
        s, _ = kernel_parameters(solver="static_metric", particle_pusher="hybrid_boris_geodesic")
        config = s._asdict()
        config.pop("guard_cells")
        new = build_static_parameters(config)
        self.assertEqual(new.guard_cells, 3)
        self.assertEqual(
            static_parameters_for_output(new)["particle_metric_reconstruction"],
            "cardinal_cubic_hermite_consistent_v1",
        )
        config["guard_cells"] = 2
        with self.assertRaisesRegex(ValueError, "guard_cells >= 3"):
            build_static_parameters(config)
        config.update(solver="electrodynamic_yee", particle_pusher="boris")
        config.pop("guard_cells")
        self.assertEqual(build_static_parameters(config).guard_cells, 2)



if __name__ == "__main__":
    unittest.main()
