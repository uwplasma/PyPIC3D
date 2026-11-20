from tests.support.dynamical_relativity_runtime_fixtures import (
    coupled_state,
    jax,
    jnp,
    rephase_pic_state,
    unittest,
)


class DynamicalRelativityTests(unittest.TestCase):
    def test_identical_metric_preserves_every_history_sample(self):
        static, dynamic, particles, species, fields = coupled_state(particles=True)
        p, f = jax.jit(lambda p, f: rephase_pic_state(
            p, species, f, f[6], static, dynamic,
        ))(particles, fields)
        for actual, expected in zip(jax.tree_util.tree_leaves((p, f)),
                                    jax.tree_util.tree_leaves((particles, fields))):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))


if __name__ == "__main__":
    unittest.main()
