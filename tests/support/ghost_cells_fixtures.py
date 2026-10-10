import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC
from PyPIC3D.boundary_conditions import ghost_cells
from tests.kernel_fixtures import kernel_parameters


def _assert_allclose(test_case, actual, expected, **kwargs):
    test_case.assertTrue(
        bool(jnp.allclose(jnp.asarray(actual), jnp.asarray(expected), **kwargs))
    )


def _particle_wall_parameters(tile_shape, g, particle_boundaries, solver="electrodynamic_yee"):
    """One-tile parameters whose particle walls are the only non-periodic boundaries."""
    return kernel_parameters(
        Nx=tile_shape[0], Ny=tile_shape[1], Nz=tile_shape[2], tile_shape=tile_shape, guard_cells=g,
        particle_boundary_conditions=particle_boundaries, solver=solver,
    )[0]


class GhostCellsFixtures:
    """Tests for the tiled ghost-cell boundary condition approach."""

    def setUp(self):
        self.tile_shape = (2, 2, 2)
        self.g = 1
        self.parameter_set = {
            "tile_shape": self.tile_shape,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }

    def _parameters_with_field_mesh(self, tile_grid_shape):
        parameter_set = dict(self.parameter_set)
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh(tile_grid_shape)
        return SimpleNamespace(
            solver="electrodynamic_yee",
            tile_shape=tuple(int(width) for width in parameter_set["tile_shape"]),
            guard_cells=self.g,
            boundary_conditions=(
                int(parameter_set["boundary_conditions"]["x"]),
                int(parameter_set["boundary_conditions"]["y"]),
                int(parameter_set["boundary_conditions"]["z"]),
            ),
            field_mesh=parameter_set["field_mesh"],
        )
