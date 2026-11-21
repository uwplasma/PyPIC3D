"""One-tile API contracts."""

from tests.support.ghost_cells_fixtures import (
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    GhostCellsFixtures,
    SimpleNamespace,
    _assert_allclose,
    _particle_wall_parameters,
    ghost_cells,
    jax,
    jnp,
    unittest,
)


class TestGhostCells(GhostCellsFixtures, unittest.TestCase):


    def test_apply_tiled_zero_boundary_zeros_global_tangential_faces(self):
        # this tests the application of axis-wise conducting boundary conditions to a tiled electric field

        parameter_set = SimpleNamespace(
            solver="electrodynamic_yee",
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
        )
        # create a parameter_set with conducting boundary conditions in all directions

        E = tuple(jnp.ones((1, 1, 1, 4, 4, 4)) for _ in range(3))
        # create a tuple of three electric field components (Ex, Ey, Ez) with shape (1, 1, 1, 4, 4, 4) and all values set to 1.0

        Ex, Ey, Ez = E
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=0, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=0, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=1, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=1, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=2, num_guard_cells=self.g)
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=2, num_guard_cells=self.g)
        # call the shared scalar zero boundary method for each tangential electric component

        self.assertTrue(jnp.all(Ey[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ey[0, 0, 0, :, :, 1] == 0.0))


if __name__ == "__main__":
    unittest.main()
