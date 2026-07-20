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

    def test_parameter_module_does_not_preserve_legacy_parameter_compatibility(self):
        self.assertFalse(hasattr(parameters, "world_from_parameters"))
        self.assertFalse(hasattr(parameters, "constants_from_parameters"))
        self.assertFalse(hasattr(parameters, "kernel_parameters_from_inputs"))
        self.assertFalse(hasattr(parameters, "static_parameters_from_legacy_mapping"))
        self.assertFalse(hasattr(parameters, "dynamic_parameters_from_legacy_mapping"))

    def test_production_path_has_no_legacy_parameter_contract(self):
        production_paths = sorted((REPO_ROOT / "PyPIC3D").rglob("*.py"))
        forbidden = (
            "parameter_set",
            "dynamic_values",
            "world_from_parameters",
            "constants_from_parameters",
            "kernel_parameters_from_inputs",
            "static_parameters_from_legacy_mapping",
            "dynamic_parameters_from_legacy_mapping",
        )

        for path in production_paths:
            source = path.read_text()
            for word in forbidden:
                with self.subTest(path=str(path.relative_to(REPO_ROOT)), word=word):
                    self.assertNotIn(word, source)

    def test_production_path_does_not_use_parameter_value_fallback(self):
        production_paths = sorted((REPO_ROOT / "PyPIC3D").rglob("*.py"))

        for path in production_paths:
            source = path.read_text()
            with self.subTest(path=str(path.relative_to(REPO_ROOT))):
                self.assertNotIn("_parameter_value", source)


if __name__ == "__main__":
    unittest.main()
