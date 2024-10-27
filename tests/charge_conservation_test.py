import unittest
import numpy as np
import jax.numpy as jnp
from charge_conservation import current_correction

class MockParticleSpecies:
    def __init__(self, charge, subcell_position, resolution, index):
        self.charge = charge
        self.subcell_position = subcell_position
        self.resolution = resolution
        self.index = index

    def get_charge(self):
        return self.charge

    def get_subcell_position(self):
        return self.subcell_position

    def get_resolution(self):
        return self.resolution

    def get_index(self):
        return self.index

class TestChargeConservation(unittest.TestCase):

    def test_current_correction(self):
        particles = [
            MockParticleSpecies(
                charge=1.0,
                subcell_position=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                resolution=(1.0, 1.0, 1.0),
                index=(1, 1, 1)
            )
        ]
        Nx, Ny, Nz = 3, 3, 3
        Jx, Jy, Jz = current_correction(particles, Nx, Ny, Nz)

        # Check if the current arrays have the expected shape
        self.assertEqual(Jx.shape, (Nx, Ny, Nz))
        self.assertEqual(Jy.shape, (Nx, Ny, Nz))
        self.assertEqual(Jz.shape, (Nx, Ny, Nz))

        # Check if the current arrays are not all zeros
        self.assertFalse(np.all(Jx == 0))
        self.assertFalse(np.all(Jy == 0))
        self.assertFalse(np.all(Jz == 0))

if __name__ == '__main__':
    unittest.main()