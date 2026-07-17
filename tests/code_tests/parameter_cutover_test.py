import inspect
from pathlib import Path
import unittest

from PyPIC3D import parameters
from PyPIC3D.evolve import time_loop_electrodynamic, time_loop_electrostatic


REPO_ROOT = Path(__file__).resolve().parents[2]


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

    def test_parameter_module_does_not_preserve_world_constants_compatibility(self):
        self.assertFalse(hasattr(parameters, "world_from_parameters"))
        self.assertFalse(hasattr(parameters, "constants_from_parameters"))
        self.assertFalse(hasattr(parameters, "kernel_parameters_from_inputs"))
        self.assertFalse(hasattr(parameters, "static_parameters_from_world"))
        self.assertFalse(hasattr(parameters, "dynamic_parameters_from_world"))

    def test_jit_kernel_path_has_no_world_constants_contract(self):
        kernel_paths = [
            "PyPIC3D/evolve.py",
            "PyPIC3D/parameters.py",
            "PyPIC3D/deposition/J_from_rhov.py",
            "PyPIC3D/deposition/Esirkepov.py",
            "PyPIC3D/deposition/rho.py",
            "PyPIC3D/pusher/particle_push.py",
            "PyPIC3D/pusher/boris.py",
            "PyPIC3D/pusher/higuera_cary.py",
            "PyPIC3D/particles/particle_tile_communication.py",
            "PyPIC3D/solvers/first_order_yee.py",
            "PyPIC3D/solvers/electrostatic_yee.py",
        ]
        forbidden = (
            "world",
            "constants",
            "world_from_parameters",
            "constants_from_parameters",
            "kernel_parameters_from_inputs",
            "static_parameters_from_world",
            "dynamic_parameters_from_world",
        )

        for relative_path in kernel_paths:
            source = (REPO_ROOT / relative_path).read_text()
            for word in forbidden:
                with self.subTest(path=relative_path, word=word):
                    self.assertNotIn(word, source)


if __name__ == "__main__":
    unittest.main()
