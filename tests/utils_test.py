import unittest
import jax
import jax.numpy as jnp
from PyPIC3D.utils import print_stats, build_yee_grid, build_collocated_grid, check_stability, particle_sanity_check

jax.config.update("jax_enable_x64", True)

class TestUtilsFunctions(unittest.TestCase):
    def setUp(self):
        self.world = {
            'Nx': 4,
            'Ny': 4,
            'Nz': 4,
            'dx': 0.1,
            'dy': 0.1,
            'dz': 0.1,
            'dt': 0.01,
            'x_wind': 1.0,
            'y_wind': 1.0,
            'z_wind': 1.0
        }
        self.plasma_parameters = {
            "Theoretical Plasma Frequency": 1.0,
            "Debye Length": 0.01,
            "Thermal Velocity": 1.0,
            "Number of Electrons": 10,
            "dx per debye length": 2.0
        }

    def test_build_yee_grid(self):
        grid, staggered = build_yee_grid(self.world)
        self.assertEqual(len(grid), 3)
        self.assertEqual(len(staggered), 3)
        #  Check that the grid and staggered arrays have the expected lengths

    def test_check_stability(self):
        # Should not raise
        check_stability(self.plasma_parameters, 0.01)
        # Check that the stability check does not raise an error

    def test_particle_sanity_check(self):
        class DummyParticles:
            def __iter__(self):
                return iter([self])
            def get_position(self):
                N = 5
                return (jnp.zeros(N), jnp.zeros(N), jnp.zeros(N))
            def get_velocity(self):
                N = 5
                return (jnp.zeros(N), jnp.zeros(N), jnp.zeros(N))
            def get_number_of_particles(self):
                return 5
        particles = DummyParticles()
        # Should not raise
        particle_sanity_check(particles)
        # Check that the particle sanity check does not raise an error

if __name__ == '__main__':
    unittest.main()
