"""Every process participates in collective diagnostics, including quiet ranks."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from PyPIC3D.diagnostics.static_metric import print_bssn_constraints


class SimulationOutputTests(unittest.TestCase):
    def test_constraint_reduction_runs_on_nonprinting_ranks(self):
        with patch("PyPIC3D.diagnostics.static_metric.bssn_constraint_norms",
                   return_value={"hamiltonian_l2": 2., "momentum_l2": 3.}) as compute, \
             patch("jax.process_index", return_value=1), redirect_stdout(StringIO()) as output:
            print_bssn_constraints("Final", None, None, None, None, None)
        compute.assert_called_once()
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
