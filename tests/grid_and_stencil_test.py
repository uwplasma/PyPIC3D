import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_CONDUCTING,
    BC_PERIODIC,
    axis_has_active_cells,
    build_axis_stencil_points,
    build_collocated_axis,
    build_staggered_axis,
    collapse_axis_stencil,
    compute_particle_anchor,
    inactive_axis_index,
    particle_axis_offset,
    prepare_particle_axis_stencil,
    uniform_axis_spacing,
    wrap_periodic_position,
)

jax.config.update("jax_enable_x64", True)


class TestGhostCellHelpers(unittest.TestCase):

    def test_wrap_periodic_position(self):
        wind = 4.0
        x = jnp.array([-2.5, -2.0, 1.75, 2.0, 2.25, 6.5])
        wrapped = wrap_periodic_position(x, wind)
        expected = jnp.array([1.5, -2.0, 1.75, 2.0, -1.75, -1.5])
        self.assertTrue(jnp.allclose(wrapped, expected))

    def test_axis_activity_and_inactive_index(self):
        self.assertTrue(axis_has_active_cells(6, ghost_cells=True))
        self.assertFalse(axis_has_active_cells(3, ghost_cells=True))
        self.assertEqual(inactive_axis_index(3, ghost_cells=True), 1)
        self.assertEqual(inactive_axis_index(1, ghost_cells=False), 0)

    def test_axis_spacing_and_anchor_helpers(self):
        axis = build_collocated_axis(-1.0, 0.5, 4)
        position = jnp.array([0.1, 0.6])
        anchor = compute_particle_anchor(position, axis, shape_factor=1)
        offset = particle_axis_offset(position, anchor, axis)

        self.assertAlmostEqual(float(uniform_axis_spacing(axis)), 0.5)
        self.assertTrue(jnp.array_equal(anchor, jnp.array([3, 4])))
        self.assertTrue(jnp.allclose(offset, jnp.array([0.1, 0.1])))

    def test_build_axis_stencil_points_periodic_and_conducting(self):
        anchor = jnp.array([0, 5])
        offsets = jnp.array([-1, 0, 1])

        periodic = build_axis_stencil_points(anchor, 6, BC_PERIODIC, offsets)
        conducting = build_axis_stencil_points(anchor, 6, BC_CONDUCTING, offsets)

        self.assertTrue(jnp.array_equal(periodic[:, 0], jnp.array([5, 0, 1])))
        self.assertTrue(jnp.array_equal(periodic[:, 1], jnp.array([4, 5, 0])))
        self.assertTrue(jnp.array_equal(conducting[:, 0], jnp.array([-1, 0, 1])))
        self.assertTrue(jnp.array_equal(conducting[:, 1], jnp.array([4, 5, 6])))

    def test_prepare_particle_axis_stencil_wraps_periodic_indices(self):
        axis = build_collocated_axis(-1.0, 0.5, 4)
        position = jnp.array([1.1])
        _, anchor, offset, points = prepare_particle_axis_stencil(
            position,
            axis,
            axis_size=6,
            shape_factor=1,
            bc=BC_PERIODIC,
            ghost_cells=True,
        )

        self.assertTrue(jnp.array_equal(anchor, jnp.array([5])))
        self.assertTrue(jnp.allclose(offset, jnp.array([0.1])))
        self.assertTrue(jnp.array_equal(points[:, 0], jnp.array([4, 5, 0])))

    def test_collapse_axis_stencil_for_inactive_axis(self):
        points = jnp.array([[0, 1], [1, 1], [2, 1]])
        weights = jnp.array([[0.2, 0.1], [0.5, 0.6], [0.3, 0.3]])

        collapsed_points, collapsed_weights = collapse_axis_stencil(
            points,
            weights,
            axis_size=3,
            ghost_cells=True,
        )

        self.assertTrue(jnp.array_equal(collapsed_points, jnp.array([[1, 1]])))
        self.assertTrue(jnp.allclose(collapsed_weights, jnp.array([[1.0, 1.0]])))

    def test_build_axis_helpers_include_ghost_cells(self):
        collocated = build_collocated_axis(-1.0, 0.5, 4)
        staggered = build_staggered_axis(-1.0, 0.5, 4)

        self.assertTrue(jnp.allclose(collocated, jnp.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])))
        self.assertTrue(jnp.allclose(staggered, jnp.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25])))


if __name__ == "__main__":
    unittest.main()
