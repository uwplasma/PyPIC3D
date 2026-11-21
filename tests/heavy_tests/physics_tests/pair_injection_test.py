from tests.support.pair_injection_runtime_fixtures import (
    birth_covariant_momentum,
    compute_rho,
    insert_pairs,
    jax,
    jnp,
    seed_leapfrog_velocity,
    setup,
    unittest,
)


class TestPairInjection(unittest.TestCase):

    def test_newborn_only_backward_seed_in_electric_field(self):
        s, d, p, species, f, x, u = setup()
        D = tuple(jnp.ones_like(a)*v for a, v in zip(f[0], (1., .2, 0.)))
        p = p._replace(active=p.active.at[..., 1, 0].set(True), u=p.u+.3)
        result, counts = insert_pairs(p, species, x, u, jnp.ones((1, 1, 1)),
                                      D, f[1], f[6], s, d, (0, 2))
        newborn = result.active & ~p.active
        physical = result._replace(u=jnp.where(newborn[..., None], u[..., :1, :][..., None, :, :], result.u),
                                   active=newborn)
        seeded = seed_leapfrog_velocity(physical, species, D, f[1], s, d, f[6])
        self.assertTrue(bool(jnp.allclose(result.u[newborn], seeded.u[newborn], atol=1e-13, rtol=1e-7, equal_nan=True)))
        self.assertTrue(bool(jnp.array_equal(result.u[~newborn], p.u[~newborn])))
        self.assertGreater(float(jnp.max(jnp.abs(result.u[..., 0, 0, :]-result.u[..., 2, 0, :]))), 0)


if __name__ == "__main__":
    unittest.main()
