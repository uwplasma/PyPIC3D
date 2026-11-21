from tests.support.code_pair_injection_fixtures import (
    TestPairInjectionFixtures,
    insert_pairs,
    jax,
    jnp,
    setup,
    unittest,
)


class TestPairInjection(TestPairInjectionFixtures, unittest.TestCase):

    def test_complete_pairs_reuse_slots_and_preserve_other_particles(self):
        s, d, p, species, f, x, u = setup()
        active = p.active.at[..., 0, :4].set(True).at[..., 2, :1].set(True)
        active = active.at[..., 1, :].set(True)
        p = p._replace(active=active, u=jnp.ones_like(p.u)*.03)
        result, counts = jax.jit(lambda pts: insert_pairs(
            pts, species, x, u, jnp.full((1, 1, 1), 5), f[0], f[1], f[6], s, d,
            species_indices=(0, 2)))(p)
        self.assertTrue(bool(jnp.all(counts[1] == 2)))
        self.assertTrue(bool(jnp.all(counts[0] - counts[1] == 3)))
        self.assertTrue(bool(jnp.array_equal(result.x[p.active], p.x[p.active])))
        self.assertTrue(bool(jnp.array_equal(result.u[p.active], p.u[p.active])))
        self.assertFalse(bool(jnp.any((result.active & ~p.active)[..., 1, :])))
        self.assertTrue(bool(jnp.array_equal(result.x[..., 0, 4:6, :], result.x[..., 2, 1:3, :])))

    def test_candidate_capacity_and_incompatible_species_reject_pairs(self):
        s, d, p, species, f, x, u = setup()
        inject = jax.jit(lambda sp: insert_pairs(
            p, sp, x, u, jnp.full((1, 1, 1), 5), f[0], f[1], f[6], s, d,
            species_indices=(0, 2)))
        result, counts = inject(species)
        self.assertTrue(bool(jnp.all(counts[1] == 3)))
        self.assertTrue(bool(jnp.all(counts[0] - counts[1] == 2)))
        for bad in (species._replace(charge=jnp.ones(3)),
                    species._replace(mass=species.mass.at[2].set(2.)),
                    species._replace(weight=species.weight.at[2].set(2.))):
            result, counts = inject(bad)
            self.assertTreeClose(result, p)
            self.assertTrue(bool(jnp.all(counts[0] - counts[1] == 5)))


    def test_empty_candidates_and_full_storage_preserve_particles(self):
        s, d, p, species, f, x, u = setup()
        for full in (False, True):
            with self.subTest(full_storage=full):
                pts = p._replace(active=jnp.full_like(p.active, full), u=p.u+.3)
                positions, momenta = (x, u) if full else (x[..., :0, :], u[..., :0, :])
                result, counts = jax.jit(lambda pts: insert_pairs(
                    pts, species, positions, momenta, jnp.ones((1, 1, 1), jnp.int32),
                    f[0], f[1], f[6], s, d, (0, 2)))(pts)
                self.assertTreeClose(result, pts)
                self.assertTrue(bool(jnp.all(counts[0] == 1)))
                self.assertTrue(bool(jnp.all(counts[1] == 0)))

    def test_candidates_exceed_storage_keep_priority_order(self):
        s, d, p, species, f, x, u = setup(slots=2)
        x = x.at[..., 1, 0].add(.05).at[..., 2, 0].add(.1)
        result, counts = jax.jit(lambda pts: insert_pairs(
            pts, species, x, u, jnp.full((1, 1, 1), 5, jnp.int32),
            f[0], f[1], f[6], s, d, (0, 2)))(p)
        self.assertTrue(bool(jnp.all(counts[0] == 5)))
        self.assertTrue(bool(jnp.all(counts[1] == 2)))
        for species_index in (0, 2):
            self.assertTrue(bool(jnp.array_equal(result.x[..., species_index, :, :], x[..., :2, :])))


if __name__ == "__main__":
    unittest.main()
