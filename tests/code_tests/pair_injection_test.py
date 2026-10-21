"""Pair birth, capacity, and compiled scheduling on the production GR path."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from PyPIC3D.relativity.pair_injection import insert_pairs, birth_covariant_momentum
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters

jax.config.update('jax_enable_x64', True)


def setup(tile_shape=(4, 4, 4), slots=6, cells=(4, 4, 4)):
    static, dynamic = kernel_parameters(
        Nx=cells[0], Ny=cells[1], Nz=cells[2], tile_shape=tile_shape, dt=.01,
        solver='static_metric', current_deposition='esirkepov',
        particle_pusher='hybrid_boris_geodesic')
    metric = initialize_flat_cartesian_metric(static, dynamic)
    zero = empty_tiled_vector(static, dynamic)
    fields = (zero, zero, zero, zero[0], zero[0], (zero, zero), metric,
              (zero, zero), jnp.asarray(False))
    # Each candidate lies strictly inside its own tile, including y and z tiles.
    center = jnp.stack([axis[..., static.guard_cells] + (axis[..., 1]-axis[..., 0])*.4
                        for axis in dynamic.grids.tiled_center_grid], axis=-1)
    shape = center.shape[:3] + (3, slots)
    x = jnp.broadcast_to(center[..., None, None, :], shape+(3,))
    particles = TiledParticles(x, jnp.zeros_like(x), jnp.zeros(shape, bool))
    species = SpeciesConfig(jnp.array([-1., 2., 1.]), jnp.ones(3),
                            jnp.ones(3), jnp.ones((3, 3), bool))
    positions = jnp.broadcast_to(center[..., None, :], center.shape[:3]+(3, 3))
    momenta = jnp.broadcast_to(jnp.array([.1, .2, -.1]), positions.shape)
    return static, dynamic, particles, species, fields, positions, momenta


def source(positions, momenta):
    def inject(particles, species, fields, static, dynamic, key, event):
        next_key, sample_key = jax.random.split(key)
        sample_key = jax.random.fold_in(sample_key, event)
        # A random momentum makes event/key progression observable.
        u = momenta * jax.random.uniform(sample_key, (), minval=.5, maxval=1.)
        requested = jnp.ones(particles.active.shape[:3], jnp.int32)
        result, counts = insert_pairs(
            particles, species, positions, u, requested, fields[0], fields[1],
            fields[6], static, dynamic, species_indices=(0, 2))
        return result, next_key, counts
    return inject


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
        before = compute_rho(p, species, f[3], s, d)
        after = compute_rho(result, species, f[3], s, d)
        self.assertTrue(bool(jnp.allclose(after, before, atol=1e-13, rtol=1e-7, equal_nan=True)))

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


if __name__ == '__main__':
    unittest.main()
