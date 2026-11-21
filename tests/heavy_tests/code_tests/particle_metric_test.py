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


if __name__ == "__main__":
    unittest.main()
