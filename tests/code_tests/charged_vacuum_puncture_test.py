"""Gaussian RN initialization, isolated source filtering and exact restarts."""

import tempfile
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
from PyPIC3D.relativity.energy_momentum_tensor import compute_electromagnetic_energy_momentum, compute_energy_momentum_tensor, electromagnetic_moments
from PyPIC3D.relativity.filtering import binomial_source_filter
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity
from demos.dynamical_relativity.charged_vacuum_puncture.initial_data import build_initial_data, native_coordinates
from demos.dynamical_relativity.charged_vacuum_puncture.run import save_checkpoint, load_checkpoint
from demos.dynamical_relativity.charged_vacuum_puncture.diagnostics import valid_state

jax.config.update("jax_enable_x64", True)


class ChargedPunctureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static, cls.dynamic, cls.particles, cls.species, cls.fields = build_initial_data(12, half_width=4., cfl=.05)

    def assertClose(self, a, b, atol=2.e-12):
        self.assertTrue(bool(jnp.allclose(a, b, rtol=2.e-12, atol=atol)))

    def test_initial_geometry_and_schwarzschild_limit(self):
        axis = self.dynamic.grids.center[0][1:-1]
        x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
        r = jnp.sqrt(x*x+y*y+z*z)
        q = self.dynamic.bssn_parameters.state
        self.assertClose(q.conformal_factor, 1/((1+1/(2*r))**2-.5**2/(4*r*r)))
        self.assertClose(q.lapse, q.conformal_factor)
        self.assertClose(q.trace_K, 0.)
        self.assertClose(q.traceless_K, 0.)
        self.assertClose(q.shift, 0.)
        _, d, p, _, f = build_initial_data(12, charge=0., half_width=4., cfl=.05)
        self.assertClose(d.bssn_parameters.state.conformal_factor, (1+1/(2*r))**-2)
        for value in f[0]+f[1]:
            self.assertClose(value, 0.)
        self.assertEqual(p.active.size, 0)

    def test_native_density_and_single_densitization(self):
        for i, location in enumerate(D_FIELD_LOCATIONS):
            xyz = native_coordinates(self.dynamic, location)
            r2 = sum(q*q for q in xyz)
            self.assertGreater(float(jnp.min(r2)), 0.)
            self.assertClose(self.fields[0][i], .5*xyz[i]/r2**1.5)
        # Check normalization independently of the initializer and filter.
        q = self.dynamic.bssn_parameters.state
        W = q.conformal_factor
        D = jnp.broadcast_to(jnp.array([.3, -.2, .1])[:, None, None, None], (3,)+W.shape)
        dd = jnp.einsum("i...,j...->ij...", D, D)
        zero = jnp.zeros_like(dd)
        rho, momentum, _ = compute_electromagnetic_energy_momentum((dd, zero, zero), q)
        self.assertClose(rho, W**4*(.3**2+.2**2+.1**2)/(8*jnp.pi))
        self.assertClose(momentum, 0.)

    def test_isolated_filter_stencil_edges_and_sum(self):
        q = jnp.zeros((2, 7, 3, 1)).at[:, 0, :, :].set(1.)
        actual = binomial_source_filter(q, (3, 3, 3))
        expected = jnp.zeros_like(q).at[:, 0, :, :].set(.75).at[:, 1, :, :].set(.25)
        self.assertClose(actual, expected)
        self.assertClose(jnp.sum(actual), jnp.sum(q))
        self.assertClose(binomial_source_filter(jnp.ones_like(q), (3, 3, 3)), 1.)
        # Three-dimensional reference, applied directly one axis at a time.
        q = jax.random.normal(jax.random.PRNGKey(7), (2, 7, 5, 3))
        reference = q
        for axis in (-3, -2, -1):
            n = reference.shape[axis]
            lo = jnp.take(reference, jnp.maximum(jnp.arange(n)-1, 0), axis=axis)
            hi = jnp.take(reference, jnp.minimum(jnp.arange(n)+1, n-1), axis=axis)
            reference = (lo + 2*reference + hi)/4
        self.assertClose(binomial_source_filter(q, (3, 3, 3)), reference)

    def test_stage_weighting_and_checkerboard_energy(self):
        s, d, p, species, f = self.static, self.dynamic, self.particles, self.species, self.fields
        moments = electromagnetic_moments(f[0], f[7][2], s)
        for scale in (1., 1.2):
            state = d.bssn_parameters.state
            state = state._replace(conformal_factor=state.conformal_factor*scale)
            raw = compute_energy_momentum_tensor(p, species, moments, state, s, d)
            actual = compute_energy_momentum_tensor(p, species, moments, state, s, d, source_filter="binomial")
            volume = state.conformal_factor**-3
            for a, b in zip(actual, raw):
                self.assertClose(a, binomial_source_filter(volume*b, s.boundary_conditions)/volume)
        state = d.bssn_parameters.state._replace(conformal_factor=jnp.ones((12,)*3))
        alternating = (-1.)**jnp.arange(f[0][0].shape[3])[None, None, None, :, None, None]
        D = (jnp.broadcast_to(alternating, f[0][0].shape), jnp.zeros_like(f[0][0]), jnp.zeros_like(f[0][0]))
        moments = electromagnetic_moments(D, tuple(jnp.zeros_like(q) for q in D), s)
        result = compute_energy_momentum_tensor(p, species, moments, state, s, d, source_filter="binomial")
        self.assertClose(result[0], 1/(8*jnp.pi))

    def test_filter_rejects_particles_and_double_filtering(self):
        s, d, p, species, f = self.static, self.dynamic, self.particles, self.species, self.fields
        moments = electromagnetic_moments(f[0], f[7][2], s)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            compute_energy_momentum_tensor(p, species, moments, d.bssn_parameters.state,
                                           s._replace(current_filter="bilinear"), d, source_filter="binomial")
        occupied = p._replace(x=jnp.zeros((1,1,1,1,1,3)), u=jnp.zeros((1,1,1,1,1,3)), active=jnp.ones((1,1,1,1,1), bool))
        with self.assertRaisesRegex(ValueError, "empty particle"):
            compute_energy_momentum_tensor(occupied, species, moments, d.bssn_parameters.state, s, d, source_filter="binomial")

    def test_empty_step_and_restart_preserve_all_native_histories(self):
        s, d, p, species, f = self.static, self.dynamic, self.particles, self.species, self.fields
        advance = jax.jit(lambda p, f, d: time_loop_dynamical_relativity(p, species, f, s, d, source_filter="binomial"))
        p1, f1, d1 = advance(p, f, d)
        self.assertTrue(bool(valid_state(f1, d1.bssn_parameters)))
        self.assertEqual(p1.active.size, 0)
        self.assertClose(d1.bssn_parameters.time, d.dt)
        expected = advance(p1, f1, d1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"restart.npz"
            save_checkpoint(path, f1, d1, {"test": True}, 1)
            restored, restored_d, step = load_checkpoint(path, f, s, d, {"test": True})
            self.assertEqual(step, 1)
            for a, b in zip(jax.tree.leaves((f1, d1)), jax.tree.leaves((restored, restored_d))):
                self.assertTrue(bool(jnp.array_equal(a, b)))
            actual = advance(p1, restored, restored_d)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            self.assertTrue(bool(jnp.array_equal(a, b)))


if __name__ == "__main__":
    unittest.main()
