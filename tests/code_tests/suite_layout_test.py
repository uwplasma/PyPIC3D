"""Keep unittest discovery and shared fixtures independent of the other suites."""

import ast
from pathlib import Path
import unittest


class TestSuiteLayout(unittest.TestCase):
    def test_test_modules_are_not_imported_as_fixtures(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    parts = module.split(".")
                    if len(parts) < 3 or parts[0] != "tests" or not parts[1].endswith("_tests"):
                        continue
                    with self.subTest(path=path.relative_to(root), module=module):
                        self.assertEqual(parts[2], "fixtures")
                        self.assertEqual(parts[1], "distributed_tests")
                        self.assertTrue(path.is_relative_to(root / "distributed_tests"))

    def test_shared_fixtures_do_not_contain_discoverable_tests(self):
        root = Path(__file__).resolve().parents[1]
        for folder in (root / "support", root / "distributed_tests" / "fixtures"):
            for path in folder.glob("*.py"):
                methods = [node.name for node in ast.walk(ast.parse(path.read_text()))
                           if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
                self.assertEqual(methods, [], str(path.relative_to(root)))


if __name__ == "__main__":
    unittest.main()
