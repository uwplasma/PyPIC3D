import unittest
import tempfile
import jax
import jax.numpy as jnp
from PyPIC3D.initialization import setup_write_dir, default_parameters, initialize_simulation

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
        self.assertIn('particle_pusher', sim)
        self.assertEqual(sim['particle_pusher'], 'boris')
        self.assertIn('eps', const)
        self.assertIn('plotfields', plotting)
        # check that the default parameters contain expected keys

    def test_initialize_simulation_rejects_unknown_solver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "simulation_parameters": {
                    "name": "unknown solver test",
                    "output_dir": tmpdir,
                    "solver": "old_solver",
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": 1.0e-10,
                    "fast_backend": "default",
                },
                "plotting": {"plotting": False},
            }

            with self.assertRaisesRegex(ValueError, "Unsupported solver"):
                initialize_simulation(config)

if __name__ == '__main__':
    unittest.main()
