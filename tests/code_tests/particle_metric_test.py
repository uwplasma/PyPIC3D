"""One-tile API contracts."""

from tests.support.particle_metric_fixtures import (
    GR_direct_deposition,
    SpeciesConfig,
    TiledParticles,
    build_static_parameters,
    consumer_runtime,
    densitize_vector,
    hybrid_boris_geodesic_push,
    interpolate_metric,
    jax,
    jnp,
    kernel_parameters,
    make_runtime,
    manufactured,
    np,
    sample_metric,
    seed_leapfrog_velocity,
    static_parameters_for_output,
    unittest,
    assert_same_tree,
)


class TestHermite(unittest.TestCase):
    def test_point_outside_stencil_is_nan(self):
        grid, m = manufactured()
        q = jnp.array([[-1.0, 0.3, 3.0]])
        a = interpolate_metric(m, q, grid, "numerical", (True, True, False), (3, 3, 3))
        self.assertTrue(np.isnan(np.asarray(a.gamma)).all())


class TestParticleMetric(unittest.TestCase):
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


class TestParticleMetricConsumers(unittest.TestCase):
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
                    assert_same_tree(self, same, original)
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
                            assert_same_tree(self, original, first_order_push)
                    difference = max(float(jnp.max(jnp.abs(jnp.nan_to_num(a)-jnp.nan_to_num(b))))
                                     for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(different))
                                     if a.dtype != jnp.bool_)
                    self.assertGreater(difference, 1e-6)


class TestSampling(unittest.TestCase):
    def test_pusher_and_seed_preserve_inactive_slots(self):
        s, d, m, D, B = make_runtime("spherical", 16, 32)
        D, B = densitize_vector(D, m.D), densitize_vector(B, m.B)
        x = jnp.array([[[[[[2.0, 0.7, 0.0], [0.0, 0.0, 0.0]]]]]])
        p = TiledParticles(x, jnp.zeros_like(x), jnp.array([[[[[True, False]]]]]))
        sp = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1), jnp.ones((1, 3), bool))
        seeded = jax.jit(lambda p: seed_leapfrog_velocity(p, sp, D, B, s, d, m))(p)
        push = jax.jit(lambda p: hybrid_boris_geodesic_push(p, sp, D, B, m, s, d))
        new, mid = push(seeded)
        self.assertTrue(np.isfinite(np.asarray(new.x)).all())
        self.assertTrue(np.isfinite(np.asarray(new.u)).all())
        np.testing.assert_array_equal(new.x[..., 1, :], p.x[..., 1, :])
        np.testing.assert_array_equal(mid.x[..., 1, :], p.x[..., 1, :])
        # An active particle on the polar axis has no valid metric sample, so
        # the pusher propagates NaN rather than repairing it.
        invalid, _ = push(p._replace(x=p.x.at[..., 0, 1].set(0.0)))
        self.assertFalse(np.isfinite(np.asarray(invalid.u)[..., 0, :]).all())

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
