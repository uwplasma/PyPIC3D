import unittest
import types
from PyPIC3D.experiment import PyPIC3DExperiment, ParameterScan

class TestExperimentFunctions(unittest.TestCase):
    def test_PyPIC3DExperiment_init(self):
        config = {'dummy': 'value'}
        exp = PyPIC3DExperiment(config)
        self.assertEqual(exp.config, config)

    def test_ParameterScan_parameters_and_build(self):
        class DummyExp:
            def __init__(self, params):
                self.params = params
        base_config = {'section': {'param': 0}, 'simulation_parameters': {'output_dir': 'dummy'}}
        scan = ParameterScan('test', 'run_dir', base_config, 'section', 'param', [1,2])
        params = list(scan.parameters())
        self.assertTrue(len(params) > 0)
        exp = scan.build(params[0][0])
        self.assertTrue(hasattr(exp, 'run'))

if __name__ == '__main__':
    unittest.main()
