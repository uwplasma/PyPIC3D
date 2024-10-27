import unittest
import numpy as np
from boundaryconditions import apply_supergaussian_boundary_condition, apply_zero_boundary_condition

class TestBoundaryConditions(unittest.TestCase):

    def test_apply_supergaussian_boundary_condition(self):
        field = np.ones((10, 10, 10))
        boundary_thickness = 2
        order = 2
        strength = 1.0

        result = apply_supergaussian_boundary_condition(field, boundary_thickness, order, strength)

        # Check if the boundary values are modified
        for i in range(boundary_thickness):
            factor = np.exp(-strength * (i / boundary_thickness)**order)
            self.assertTrue(np.allclose(result[i, :, :], factor))
            self.assertTrue(np.allclose(result[-1 - i, :, :], factor))
            self.assertTrue(np.allclose(result[:, i, :], factor))
            self.assertTrue(np.allclose(result[:, -1 - i, :], factor))
            self.assertTrue(np.allclose(result[:, :, i], factor))
            self.assertTrue(np.allclose(result[:, :, -1 - i], factor))

    def test_apply_zero_boundary_condition(self):
        field = np.random.rand(10, 10, 10)

        result = apply_zero_boundary_condition(field)

        # Check if the boundary values are set to zero
        self.assertTrue(np.all(result[0, :, :] == 0))
        self.assertTrue(np.all(result[-1, :, :] == 0))
        self.assertTrue(np.all(result[:, 0, :] == 0))
        self.assertTrue(np.all(result[:, -1, :] == 0))
        self.assertTrue(np.all(result[:, :, 0] == 0))
        self.assertTrue(np.all(result[:, :, -1] == 0))

if __name__ == '__main__':
    unittest.main()