import unittest
from unittest import mock
from utils import fix_boundary_condition, grab_particle_keys, load_particles_from_toml, debugprint

class TestUtils(unittest.TestCase):

    def test_fix_boundary_condition(self):
        def sample_func(a, bc):
            return a + bc

        bc_value = 5
        fixed_func = fix_boundary_condition(sample_func, bc_value)
        self.assertEqual(fixed_func(10), 15)

    def test_grab_particle_keys(self):
        config = {
            'particle1': {},
            'particle2': {},
            'other_key': {}
        }
        expected_keys = ['particle1', 'particle2']
        self.assertEqual(grab_particle_keys(config), expected_keys)

    @mock.patch('utils.toml.load')
    @mock.patch('utils.initial_particles')
    @mock.patch('utils.particle_species')
    def test_load_particles_from_toml(self, mock_particle_species, mock_initial_particles, mock_toml_load):
        mock_toml_load.return_value = {
            'particle1': {
                'name': 'electron',
                'N_particles': 100,
                'charge': -1,
                'mass': 9.11e-31,
                'temperature': 300,
                'update_pos': True,
                'update_v': True
            }
        }
        mock_initial_particles.return_value = ([], [], [], [], [], [])
        simulation_parameters = {
            'x_wind': 0,
            'y_wind': 0,
            'z_wind': 0,
            'kb': 1.38e-23
        }
        dx, dy, dz = 1.0, 1.0, 1.0

        particles = load_particles_from_toml('dummy.toml', simulation_parameters, dx, dy, dz)
        self.assertEqual(len(particles), 1)
        mock_particle_species.assert_called_once()

    @mock.patch('utils.jax.debug.print')
    def test_debugprint(self, mock_debug_print):
        value = 'test'
        debugprint(value)
        mock_debug_print.assert_called_once_with('{x}', x=value)

if __name__ == '__main__':
    unittest.main()