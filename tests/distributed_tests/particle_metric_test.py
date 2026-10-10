"""Focused cross-device regression tests."""

from tests.support.particle_metric_fixtures_shared import (
    ParticleMetricConsumersFixtures,
    checkify,
    consumer_runtime,
    coordinate_velocity,
    densitize_vector,
    hybrid_boris_geodesic_push,
    interpolate_metric,
    jax,
    jnp,
    np,
    unittest,
)


class TestParticleMetricConsumers(ParticleMetricConsumersFixtures, unittest.TestCase):
    def test_midpoint_sampling_agrees_across_tile_seam(self):
        results = []
        for tile_shape in ((8, 8, 1), (4, 8, 1)):
            s, d, m, D, B, p, species = consumer_runtime(tile_shape=tile_shape)
            D, B = densitize_vector(D, m.D), densitize_vector(B, m.B)
            err, (new, mid) = jax.jit(checkify.checkify(
                lambda p: hybrid_boris_geodesic_push(p, species, D, B, m, s, d)))(p)
            err.throw()
            self.assertGreater(float(mid.x.reshape(-1, 3)[0, 0]), 1.5)
            for tx in range(p.x.shape[0]):
                grid = tuple(a[tx, 0, 0] for a in d.grids.tiled_center_grid)
                tile = jax.tree.map(lambda a: a[tx, 0, 0], m.center)
                get = lambda q: interpolate_metric(tile, q, grid, s.metric,
                    (True, True, False), (3, 3, 3), derivatives=False)
                x, u = p.x[tx], new.u[tx]
                expected_mid = x + .5*d.dt*coordinate_velocity(u, get(x))
                expected_new = x + d.dt*coordinate_velocity(u, get(expected_mid))
                np.testing.assert_allclose(mid.x[tx], expected_mid, atol=1e-14)
                np.testing.assert_allclose(new.x[tx], expected_new, atol=1e-14)
                stale = x + d.dt*coordinate_velocity(u, get(x))
                self.assertGreater(float(jnp.max(jnp.abs(expected_new-stale))), 1e-9)
            results.append(tuple(a.reshape(-1, 3) for a in (new.x, new.u, mid.x)))
        self.assert_same_tree(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
