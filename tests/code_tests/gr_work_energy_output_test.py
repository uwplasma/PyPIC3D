"""GR energy output cadence through the real coupled driver and timestep."""

from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed")
    raise

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.diagnostics.gr_work_energy_theorem import (
    baseline_tracker,
    tracker_values,
    initialize_tracker,
)
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity
from tests.code_tests.dynamical_relativity_test import coupled_state


jax.config.update("jax_enable_x64", True)


class GREnergyOutputTests(unittest.TestCase):
    def run_driver(self, directory, *, steps, cadence, gravity_boundary=None):
        # Uniform vacuum fields leave a cheap but genuine Maxwell/BSSN step.
        # Only initialization and unrelated output are replaced; the driver,
        # JIT, energy accumulator, and scalar text writer execute normally.
        static, dynamic, particles, species, fields = coupled_state()
        static = static._replace(gr_work_energy_theorem=True, Nt=steps, output_dir=directory)
        dynamic = dynamic._replace(work_energy_tracker=initialize_tracker(
            particles, species, fields, static, dynamic))
        initial = dynamic.work_energy_tracker
        plotting = dict(plotting_interval=cadence, plot_openpmd_fields=False,
                        plot_openpmd_particles=False)
        loop = partial(time_loop_dynamical_relativity, gravity_boundary=gravity_boundary)
        runtime = (loop, particles, fields, static, dynamic, plotting, {}, species)
        data_dir = Path(directory) / "data"
        data_dir.mkdir()

        with patch("PyPIC3D.__main__.initialize_simulation", return_value=runtime), \
             patch("PyPIC3D.__main__.tqdm", side_effect=lambda values: values), \
             patch("PyPIC3D.diagnostics.static_metric.bssn_constraint_norms", return_value={}):
            result = run_PyPIC3D({})

        final = result[1].work_energy_tracker
        rows = {
            path.stem.removeprefix("work_energy_"): jnp.asarray([
                [float(value) for value in line.split(",")]
                for line in path.read_text().splitlines()
            ])
            for path in data_dir.glob("work_energy_*.txt")
        }
        self.assertEqual(set(rows), set(tracker_values(final)))
        return initial, final, rows

    def test_write_cadence_preserves_budget_and_includes_unique_final_endpoint(self):
        results = []
        for cadence in (1, 3):
            with tempfile.TemporaryDirectory() as directory:
                _, final, rows = self.run_driver(directory, steps=4, cadence=cadence)
            expected_times = .02 * jnp.asarray([0, 1, 2, 3, 4] if cadence == 1 else [0, 3, 4])
            for name, values in rows.items():
                self.assertTrue(bool(jnp.allclose(values[:, 0], expected_times, rtol=0., atol=1.e-14)), name)
                self.assertEqual(int(jnp.unique(values[:, 0]).size), values.shape[0], name)
                self.assertTrue(bool(jnp.all(jnp.isfinite(values))), name)
                self.assertAlmostEqual(float(values[-1, 1]),
                                       float(tracker_values(final)[name]), places=14)
            self.assertEqual(int(final.steps), 4)
            self.assertEqual(float(rows["residual"][0, 1]), 0.)
            self.assertEqual(float(rows["step_residual"][0, 1]), 0.)
            results.append(final)

        for frequent, sparse in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
            self.assertTrue(bool(jnp.allclose(frequent, sparse, rtol=0., atol=1.e-14)))

    def test_zero_steps_write_one_initial_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            initial, final, rows = self.run_driver(directory, steps=0, cadence=3)
        self.assertEqual(int(final.steps), 0)
        for name, values in rows.items():
            self.assertEqual(values.shape, (1, 2), name)
            self.assertEqual(float(values[0, 0]), float(initial.baseline.time), name)
            self.assertAlmostEqual(float(values[0, 1]),
                                   float(tracker_values(baseline_tracker(initial))[name]), places=14)

    def test_initial_row_uses_the_baseline_after_geometry_preparation(self):
        def prescribed_geometry(state, time):
            return state._replace(conformal_factor=jnp.full_like(state.conformal_factor, 1.2))

        with tempfile.TemporaryDirectory() as directory:
            initial, final, rows = self.run_driver(
                directory, steps=1, cadence=3, gravity_boundary=prescribed_geometry)
        self.assertGreater(abs(float(final.baseline.electromagnetic - initial.baseline.electromagnetic)), 1.e-5)
        self.assertAlmostEqual(float(rows["total"][0, 1]),
                               float(final.baseline.electromagnetic), places=14)
        self.assertEqual(float(rows["residual"][0, 1]), 0.)
        self.assertEqual(float(rows["reference"][0, 1]), float(final.reference))
        self.assertEqual(rows["total"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
