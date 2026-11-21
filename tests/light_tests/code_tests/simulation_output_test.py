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


if __name__ == "__main__":
    unittest.main()
