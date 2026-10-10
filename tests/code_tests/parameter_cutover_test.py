import inspect
import unittest

from PyPIC3D.solvers.electrostatic.time_loop import time_loop_electrostatic
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic


class TestStaticDynamicParameterCutover(unittest.TestCase):
    def test_evolve_methods_take_only_split_parameter_contract(self):
        electrodynamic_signature = inspect.signature(time_loop_electrodynamic)
        electrostatic_signature = inspect.signature(time_loop_electrostatic)

        self.assertEqual(
            list(electrodynamic_signature.parameters),
            ["particles", "species_config", "fields", "static_parameters", "dynamic_parameters"],
        )
        self.assertEqual(
            list(electrostatic_signature.parameters),
            ["particles", "species_config", "fields", "static_parameters", "dynamic_parameters"],
        )

if __name__ == "__main__":
    unittest.main()
