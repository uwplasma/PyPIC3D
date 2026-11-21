from tests.support.pair_injection_runtime_fixtures import (
    insert_pairs,
    jax,
    jnp,
    patch,
    run_PyPIC3D,
    setup,
    source,
    time_loop_static_metric,
    unittest,
)


class TestPairInjectionFixtures:
    def assertTreeClose(self, a, b):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            self.assertTrue(bool(jnp.allclose(x, y, atol=2e-13, rtol=2e-13, equal_nan=True)))
