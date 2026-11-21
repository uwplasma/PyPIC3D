from tests.support.physics_gr_matched_filter_fixtures import (
    TestGRMatchedFilterFixtures,
    filter_gr_grid,
    jax,
    jnp,
    parameters,
    unittest,
)


class TestGRMatchedFilter(TestGRMatchedFilterFixtures, unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
