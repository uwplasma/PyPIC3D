from tests.support.direct_deposition_fixtures import (
    BC_CONSTANT,
    get_first_order_weights,
    prepare_particle_axis_stencil,
)
from tests.support.distributed.direct_deposition import DirectDepositionFixtures
"""Focused cross-device regression tests."""

from tests.support.direct_deposition_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    J_from_rhov,
    assemble_tiled_vector_field,
    jnp,
    kernel_parameters_from_values,
    unittest,
)


class TestDirectDeposition(DirectDepositionFixtures, unittest.TestCase):


    def test_face_stencil_at_grid_node_is_independent_of_tile_origin(self):
        dx = 0.5
        position = jnp.asarray([-1.1102230246251565e-16])
        grid_axes = (
            jnp.asarray((-1.0, -0.5, 0.0, 0.5)),
            jnp.asarray((-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)),
        )
        face_stencils = []

        for grid_axis in grid_axes:
            face_grid_axis = grid_axis + 0.5 * dx
            _, _, delta_face, points_face = prepare_particle_axis_stencil(
                position,
                face_grid_axis,
                len(face_grid_axis),
                shape_factor=1,
                bc=BC_CONSTANT,
                ghost_cells=True,
            )
            weights_face, _, _ = get_first_order_weights(
                delta_face,
                delta_face,
                delta_face,
                dx,
                dx,
                dx,
            )
            weights_face = jnp.stack(weights_face)[:, 0]
            coordinates_face = face_grid_axis[0] + points_face[:, 0] * dx
            face_stencils.append((coordinates_face, weights_face))

            self.assertTrue(jnp.all(weights_face >= 0.0))
            self.assertAlmostEqual(float(jnp.sum(weights_face)), 1.0)

        for tiled_values, one_tile_values in zip(face_stencils[0], face_stencils[1]):
            self.assertTrue(jnp.allclose(tiled_values, one_tile_values))
        self.assertTrue(jnp.allclose(face_stencils[0][1], jnp.asarray((0.0, 0.5, 0.5))))


if __name__ == "__main__":
    unittest.main()
