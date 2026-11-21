"""Focused cross-device regression tests."""

from tests.support.particle_metric_fixtures import (
    manufactured,
    np,
)
from tests.support.particle_metric_fixtures import (
    consumer_runtime,
    coordinate_velocity,
    densitize_vector,
    hybrid_boris_geodesic_push,
    interpolate_metric,
    jax,
    jnp,
    unittest,
)


class TestParticleMetric(unittest.TestCase):
    def test_adjacent_radial_tiles_agree(self):
        g1, m1 = manufactured()
        g2, m2 = manufactured(0.5)
        q = jnp.array([[0.499, 0.381, 3.0], [0.501, 0.381, 3.0]])
        a = interpolate_metric(m1, q, g1, "numerical", (True, True, False), (3, 3, 3))
        b = interpolate_metric(m2, q, g2, "numerical", (True, True, False), (3, 3, 3))
        for v, w in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(v, w, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
