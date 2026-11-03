"""Joint spatial/time refinement of a smooth, evolving BSSN--Maxwell budget.

The periodic wave uses the solver's documented exactly flat initial geometry.
With nonzero EM stress this initial state violates the Einstein constraints;
this test checks matter/EM tracker on the evolved numerical geometry, not
Einstein-constraint convergence or an exact spacetime solution.
"""

import unittest

import jax
import jax.numpy as jnp

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed")
    raise

from PyPIC3D.diagnostics.gr_work_energy_theorem import tracker_values
from tests.physics_tests.gr_work_energy_theorem_test import evolve_scan, maxwell_state

jax.config.update("jax_enable_x64", True)


class GREnergyCoupledRefinementTests(unittest.TestCase):
    def test_periodic_wave_budget_refines_with_evolving_geometry(self):
        errors = []
        geometric_work = []
        for n in (16, 32, 64):
            # All runs cover the same 4 x 1 x 1 coordinate volume. The two
            # collapsed axes retain unit widths while dx and dt both halve.
            static, dynamic, particles, species, fields = maxwell_state(n=n)
            bssn = dynamic.bssn_parameters
            dynamic = dynamic._replace(
                bssn_parameters=bssn._replace(gravity_G=jnp.asarray(1.)))
            particles, fields, final = evolve_scan(
                static, species, (particles, fields, dynamic), n//2)

            values = tracker_values(final.work_energy_tracker)
            state = final.bssn_parameters.state
            gamma = state.conformal_metric / state.conformal_factor**2
            initial_gamma = bssn.state.conformal_metric / bssn.state.conformal_factor**2
            self.assertGreater(float(jnp.max(jnp.abs(gamma-initial_gamma))), 1.e-4)
            self.assertGreater(float(jnp.max(jnp.abs(state.trace_K))), 1.e-3)
            self.assertAlmostEqual(float(final.bssn_parameters.time), .2, places=12)
            self.assertEqual(float(values["boundary_work"]), 0.)
            self.assertTrue(bool(jnp.all(jnp.isfinite(gamma))))

            errors.append(values["relative_residual"])
            geometric_work.append(jnp.abs(values["geometry_work"] / values["reference"]))

        errors = jnp.asarray(errors)
        geometric_work = jnp.asarray(geometric_work)
        orders = jnp.log2(errors[:-1] / errors[1:])
        message = f"residuals={errors}; orders={orders}; geometric_work/E0={geometric_work}"
        self.assertTrue(bool(jnp.all(geometric_work > 1.e-5)), message)
        self.assertTrue(bool(jnp.all(errors[1:] < errors[:-1])), message)
        self.assertTrue(bool(jnp.all(orders > 1.7)), message)
        self.assertLess(float(errors[-1]), 1.e-6, message)


if __name__ == "__main__":
    unittest.main()
