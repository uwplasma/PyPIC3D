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


class TestPairInjection(unittest.TestCase):
    def assertTreeClose(self, a, b):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            self.assertTrue(bool(jnp.allclose(x, y, atol=2e-13, rtol=2e-13, equal_nan=True)))

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

    def test_zero_capacity_reports_rejection(self):
        s, d, p, species, f, x, u = setup(slots=0)
        result, counts = insert_pairs(p, species, x, u, jnp.ones((1, 1, 1)),
                                      f[0], f[1], f[6], s, d, (0, 2))
        self.assertTreeClose(result, p)
        self.assertTrue(bool(jnp.all(counts[0] - counts[1] == 1)))

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

    def test_jit_scan_matches_explicit_inject_then_evolve(self):
        s, d, p, species, f, x, u = setup()
        inject = source(x, u)
        key = jax.random.PRNGKey(17)
        def step(carry, index):
            pts, fields, key = carry
            pts, fields, key, counts = time_loop_static_metric(
                pts, species, fields, s, d, pair_injector=inject,
                injection_key=key, step=index, injection_interval_steps=2)
            return (pts, fields, key), counts[1]
        actual, counts = jax.jit(lambda carry: jax.lax.scan(step, carry, jnp.arange(4)))((p, f, key))
        self.assertTrue(bool(jnp.array_equal(counts.reshape(-1), jnp.array([1, 0, 1, 0]))))
        evolve = jax.jit(lambda pts, fields: time_loop_static_metric(pts, species, fields, s, d))
        for index in range(4):
            if index % 2 == 0:
                p, key, _ = inject(p, species, f, s, d, key, index//2)
            p, f = evolve(p, f)
        self.assertTreeClose(actual, (p, f, key))

    def test_skipped_step_unchanged_key_and_original_output_contract(self):
        s, d, p, species, f, x, u = setup()
        inject = source(x, u)
        key = jax.random.PRNGKey(19)
        reference = time_loop_static_metric(p, species, f, s, d, return_diagnostics=True)
        result = jax.jit(lambda key: time_loop_static_metric(
            p, species, f, s, d, return_diagnostics=True, pair_injector=inject,
            injection_key=key, step=1, injection_interval_steps=2))(key)
        self.assertEqual(len(reference), 3)
        self.assertEqual(len(result), 5)
        self.assertTreeClose(reference, result[:3])
        self.assertTrue(bool(jnp.array_equal(result[3], key)))
        self.assertEqual(len(result[4]), 2)
        for count in result[4]:
            self.assertEqual(count.shape, p.active.shape[:3])
            self.assertEqual(count.dtype, jnp.int32)
            self.assertTrue(bool(jnp.all(count == 0)))

    def test_nonstatic_solver_rejected(self):
        s, d, p, species, f, x, u = setup()
        s = s._replace(solver='electrodynamic_yee')
        with self.assertRaisesRegex(ValueError, 'static_metric'):
            insert_pairs(p, species, x, u, jnp.ones((1, 1, 1)), f[0], f[1], f[6], s, d)
        initial = (time_loop_static_metric, p, f, s, d,
                   {'particle_species_names': None}, {}, species)
        with patch('PyPIC3D.__main__.initialize_simulation', return_value=initial):
            with self.assertRaisesRegex(ValueError, 'static_metric'):
                run_PyPIC3D({}, pair_injector=source(x, u), injection_key=jax.random.PRNGKey(0))

    def test_main_runner_passes_key_and_schedule(self):
        s, d, p, species, f, x, u = setup()
        s = s._replace(Nt=4)
        plotting = dict(particle_species_names=None, plot_openpmd_fields=False,
                        plot_openpmd_particles=False, plotting_interval=1)
        initial = (time_loop_static_metric, p, f, s, d, plotting, {}, species)
        with patch('PyPIC3D.__main__.initialize_simulation', return_value=initial):
            result = run_PyPIC3D({}, pair_injector=source(x, u),
                                injection_key=jax.random.PRNGKey(0), injection_interval_steps=2)
        self.assertEqual(int(result[4].active.sum()), 4)

    def test_main_runner_rejects_uninserted_pairs(self):
        s, d, p, species, f, x, u = setup(slots=0)
        s = s._replace(Nt=1)
        plotting = dict(particle_species_names=None, plot_openpmd_fields=False,
                        plot_openpmd_particles=False, plotting_interval=1)
        initial = (time_loop_static_metric, p, f, s, d, plotting, {}, species)
        with patch('PyPIC3D.__main__.initialize_simulation', return_value=initial):
            with self.assertRaisesRegex(RuntimeError, 'requested=.*inserted=.*rejected='):
                run_PyPIC3D({}, pair_injector=source(x, u),
                            injection_key=jax.random.PRNGKey(0))


if __name__ == "__main__":
    unittest.main()
