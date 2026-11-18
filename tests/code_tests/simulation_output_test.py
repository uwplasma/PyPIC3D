"""Output cleanup and endpoint reports through the simulation driver."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import jax.numpy as jnp

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.diagnostics import simulation_output
from PyPIC3D.diagnostics.static_metric import bssn_constraint_norms, print_bssn_constraints
from PyPIC3D.initialization import initialize_simulation


class SimulationOutputTests(unittest.TestCase):
    def test_writers_close_after_step_or_writer_failure(self):
        for failure in ("step", "close", "setup"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                static = SimpleNamespace(output_dir=directory)
                plotting = dict(plot_openpmd_fields=True, plot_openpmd_particles=True)
                field_writer, particle_writer = Mock(), Mock()
                if failure in ("step", "close"):
                    particle_writer.close.side_effect = OSError("writer failed")
                expected_error = RuntimeError if failure == "step" else OSError
                with patch.object(simulation_output, "setup_pmd_files"), \
                     patch.object(simulation_output, "create_async_tiled_openpmd_field_writer",
                                  return_value=field_writer), \
                     patch.object(simulation_output, "create_async_tiled_openpmd_particle_writer",
                                  side_effect=OSError("setup failed") if failure == "setup" else None,
                                  return_value=particle_writer):
                    with self.assertRaises(expected_error):
                        with simulation_output.open_simulation_output(static, None, plotting):
                            if failure == "step":
                                raise RuntimeError("step failed")
                field_writer.close.assert_called_once_with(raise_errors=failure == "close")
                if failure != "setup":
                    particle_writer.close.assert_called_once_with(raise_errors=failure == "close")

    def test_constraint_reduction_runs_on_nonprinting_ranks(self):
        with patch("PyPIC3D.diagnostics.static_metric.bssn_constraint_norms",
                   return_value={"hamiltonian_l2": 2., "momentum_l2": 3.}) as compute, \
             patch("jax.process_index", return_value=1), redirect_stdout(StringIO()) as output:
            print_bssn_constraints("Final", None, None, None, None, None)
        compute.assert_called_once()
        self.assertEqual(output.getvalue(), "")

    def test_initial_and_final_constraints_match_common_time_states(self):
        try:
            import JAX_BSSN
        except ModuleNotFoundError:
            self.skipTest("Optional JAX_BSSN dependency is not installed")

        for steps in (0, 1):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as directory:
                config = {
                    "simulation_parameters": dict(
                        solver="dynamic_metric", output_dir=directory, Nt=steps, dt=.01,
                        Nx=4, Ny=4, Nz=4, x_wind=4., y_wind=4., z_wind=4.),
                    "plotting": dict(plotting_interval=3),
                }
                initial_states = []

                def initialize(*args, **kwargs):
                    initial = initialize_simulation(*args, **kwargs)
                    initial_states.append(initial)
                    return initial

                with patch("PyPIC3D.__main__.initialize_simulation", side_effect=initialize), \
                     patch("PyPIC3D.diagnostics.static_metric.print_bssn_constraints",
                           wraps=print_bssn_constraints) as report, \
                     redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()):
                    result = run_PyPIC3D(config, field_initializers={
                        "Ex": lambda x, y, z: .02 * jnp.sin(jnp.pi * x / 2)})

                _, particles, fields, static, dynamic, _, _, species = initial_states[0]
                initial_norms = bssn_constraint_norms(particles, species, fields, static, dynamic)
                static, dynamic, _, _, particles, fields, species = result
                final_norms = bssn_constraint_norms(particles, species, fields, static, dynamic)
                self.assertEqual(float(dynamic.bssn_parameters.time), steps * .01)
                self.assertEqual([call.args[0] for call in report.call_args_list], ["Initial", "Final"])
                self.assertEqual(float(report.call_args.args[-1].bssn_parameters.time), steps * .01)
                for label, norms in (("Initial", initial_norms), ("Final", final_norms)):
                    expected = (f"{label} Einstein constraints: Hamiltonian RMS={float(norms['hamiltonian_l2']):.6e}, "
                                f"momentum RMS={float(norms['momentum_l2']):.6e}")
                    self.assertIn(expected, output.getvalue())
                self.assertNotIn("Field Energy", output.getvalue())
                self.assertNotIn("Step compilation", output.getvalue())
                self.assertNotIn("slowest process", output.getvalue())
                # The final console report must not add a new scalar-file endpoint.
                path = Path(directory) / "data" / "hamiltonian_l2.txt"
                self.assertEqual(len(path.read_text().splitlines()) if path.exists() else 0,
                                 int(steps > 0))


if __name__ == "__main__":
    unittest.main()
