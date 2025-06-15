import unittest
import jax
import jax.numpy as jnp
from PyPIC3D.initialization import setup_write_dir, default_parameters

jax.config.update("jax_enable_x64", True)

class TestInitializationFunctions(unittest.TestCase):
    def setUp(self):
        self.plotting_parameters, self.simulation_parameters, self.constants = default_parameters()
        self.simulation_parameters['output_dir'] = 'test_output'
        self.plotting_parameters['plotfields'] = False
        # check the  default parameters are set correctly

    def test_setup_write_dir(self):
        # Should not raise
        setup_write_dir(self.simulation_parameters, self.plotting_parameters)
        # check that the output directory is created

    def test_default_parameters(self):
        plotting, sim, const = default_parameters()
        self.assertIn('Nx', sim)
        self.assertIn('eps', const)
        self.assertIn('plotfields', plotting)
        # check that the default parameters contain expected keys

if __name__ == '__main__':
    unittest.main()
