import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.deposition.Esirkepov import (
    eliminate_esirkepov_ghost_cells,
    enforce_bc_along_axis,
)

jax.config.update("jax_enable_x64", True)


class TestExtendedEsirkepovGhostCells(unittest.TestCase):

    def test_two_periodic_ghost_layers_fold_to_opposite_interior(self):
        field = jnp.zeros((8, 3, 3))
        field = field.at[0, 1, 1].set(1.0)
        field = field.at[1, 1, 1].set(2.0)
        field = field.at[-2, 1, 1].set(3.0)
        field = field.at[-1, 1, 1].set(4.0)
        field = field.at[2, 1, 1].set(10.0)
        field = field.at[3, 1, 1].set(20.0)
        field = field.at[-4, 1, 1].set(30.0)
        field = field.at[-3, 1, 1].set(40.0)

        folded = enforce_bc_along_axis(field, axis=0, bc="periodic", component_axis=0)

        self.assertAlmostEqual(float(folded[-4, 1, 1]), 31.0)
        self.assertAlmostEqual(float(folded[-3, 1, 1]), 42.0)
        self.assertAlmostEqual(float(folded[2, 1, 1]), 13.0)
        self.assertAlmostEqual(float(folded[3, 1, 1]), 24.0)
        self.assertEqual(float(folded[0, 1, 1]), 0.0)
        self.assertEqual(float(folded[1, 1, 1]), 0.0)
        self.assertEqual(float(folded[-2, 1, 1]), 0.0)
        self.assertEqual(float(folded[-1, 1, 1]), 0.0)

    def test_two_reflecting_ghost_layers_fold_to_same_side_with_normal_sign(self):
        field = jnp.zeros((8, 3, 3))
        field = field.at[0, 1, 1].set(1.0)
        field = field.at[1, 1, 1].set(2.0)
        field = field.at[-2, 1, 1].set(3.0)
        field = field.at[-1, 1, 1].set(4.0)
        field = field.at[2, 1, 1].set(10.0)
        field = field.at[3, 1, 1].set(20.0)
        field = field.at[-4, 1, 1].set(30.0)
        field = field.at[-3, 1, 1].set(40.0)

        folded = enforce_bc_along_axis(field, axis=0, bc="reflecting", component_axis=0)

        self.assertAlmostEqual(float(folded[2, 1, 1]), 9.0)
        self.assertAlmostEqual(float(folded[3, 1, 1]), 18.0)
        self.assertAlmostEqual(float(folded[-4, 1, 1]), 27.0)
        self.assertAlmostEqual(float(folded[-3, 1, 1]), 36.0)

    def test_two_absorbing_ghost_layers_are_discarded(self):
        field = jnp.zeros((8, 3, 3))
        field = field.at[0, 1, 1].set(1.0)
        field = field.at[1, 1, 1].set(2.0)
        field = field.at[-2, 1, 1].set(3.0)
        field = field.at[-1, 1, 1].set(4.0)
        field = field.at[2, 1, 1].set(10.0)
        field = field.at[-3, 1, 1].set(40.0)

        folded = enforce_bc_along_axis(field, axis=0, bc="absorbing", component_axis=0)

        self.assertAlmostEqual(float(folded[2, 1, 1]), 10.0)
        self.assertAlmostEqual(float(folded[-3, 1, 1]), 40.0)
        self.assertEqual(float(folded[0, 1, 1]), 0.0)
        self.assertEqual(float(folded[1, 1, 1]), 0.0)
        self.assertEqual(float(folded[-2, 1, 1]), 0.0)
        self.assertEqual(float(folded[-1, 1, 1]), 0.0)

    def test_eliminate_esirkepov_ghost_cells_returns_one_ghost_layer(self):
        field = jnp.arange(8 * 3 * 3, dtype=float).reshape((8, 3, 3))

        restricted = eliminate_esirkepov_ghost_cells(field)

        self.assertEqual(restricted.shape, (6, 1, 1))
        self.assertTrue(jnp.array_equal(restricted[:, 0, 0], field[1:-1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
