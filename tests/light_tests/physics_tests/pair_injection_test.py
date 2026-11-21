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
    def test_pair_injection_preserves_charge_density(self):
        s, d, p, species, f, x, u = setup()
        active = p.active.at[..., 0, :4].set(True).at[..., 2, :1].set(True)
        active = active.at[..., 1, :].set(True)
        p = p._replace(active=active, u=jnp.ones_like(p.u)*.03)
        result, counts = jax.jit(lambda pts: insert_pairs(
            pts, species, x, u, jnp.full((1, 1, 1), 5), f[0], f[1], f[6], s, d,
            species_indices=(0, 2)))(p)
        self.assertTrue(bool(jnp.all(counts[1] == 2)))
        before = compute_rho(p, species, f[3], s, d)
        after = compute_rho(result, species, f[3], s, d)
        self.assertTrue(bool(jnp.allclose(after, before, atol=1e-13, rtol=1e-7, equal_nan=True)))


    def test_covariant_transform_preserves_metric_norm(self):
        s, d, p, species, f, x, u = setup()
        gamma = jnp.array([[2., .3, .1], [.3, 3., .2], [.1, .2, 1.]])
        metric = jax.tree.map(lambda a: a[0, 0, 0], f[6].center)
        metric = metric._replace(gamma=jnp.broadcast_to(gamma, metric.gamma.shape))
        grid = tuple(a[0, 0, 0] for a in d.grids.tiled_center_grid)
        active = jnp.array([True, True, False])
        positions = x[0, 0, 0].at[-1].set(jnp.nan)
        momenta = u[0, 0, 0].at[-1].set(jnp.nan)
        cov = jax.jit(lambda q, u: birth_covariant_momentum(
            u, q, active, metric, grid, s, (True, True, True)))(positions, momenta)
        expected = jnp.einsum('ij,nj->ni', jnp.linalg.cholesky(gamma), momenta[:2])
        self.assertTrue(bool(jnp.allclose(cov[:2], expected, atol=1e-14, rtol=1e-7, equal_nan=True)))
        norm = jnp.einsum('ni,ij,nj->n', cov[:2], jnp.linalg.inv(gamma), cov[:2])
        self.assertTrue(bool(jnp.allclose(norm, jnp.sum(momenta[:2]**2, axis=-1), atol=1e-14, rtol=1e-7, equal_nan=True)))
        self.assertTrue(bool(jnp.all(cov[-1] == 0.)))


if __name__ == "__main__":
    unittest.main()
