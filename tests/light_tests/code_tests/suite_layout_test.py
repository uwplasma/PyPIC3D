"""Keep the six unittest suites independent and shared fixtures undiscoverable."""

import ast
from pathlib import Path
import unittest


class TestSuiteLayout(unittest.TestCase):
    def test_test_modules_are_not_imported_as_fixtures(self):
        root = Path(__file__).resolve().parents[2]
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                    if node.module == "tests":
                        modules.extend("tests." + alias.name for alias in node.names)
                    self.assertEqual(node.level, 0, str(path.relative_to(root)))
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    parts = module.split(".")
                    if len(parts) < 2 or parts[0] != "tests":
                        continue
                    with self.subTest(path=path.relative_to(root), module=module):
                        self.assertIn(parts[1], ("support", "kernel_fixtures"))

    def test_shared_fixtures_do_not_contain_discoverable_tests(self):
        root = Path(__file__).resolve().parents[2]
        for path in (root / "support").rglob("*.py"):
            methods = [node.name for node in ast.walk(ast.parse(path.read_text()))
                       if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
            self.assertEqual(methods, [], str(path.relative_to(root)))
        for tier in ("light_tests", "heavy_tests"):
            self.assertTrue((root / tier / "__init__.py").is_file())
            for group in ("code_tests", "physics_tests", "distributed_tests"):
                folder = root / tier / group
                self.assertTrue((folder / "__init__.py").is_file())
                self.assertTrue(list(folder.glob("*test*.py")), str(folder))


if __name__ == "__main__":
    unittest.main()
