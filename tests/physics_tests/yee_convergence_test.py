import math
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.solvers.first_order_yee import update_B
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid


jax.config.update("jax_enable_x64", True)


class TestTiledYeeConvergence(unittest.TestCase):
    def _world(self, Nx, tile_shape):
        x_wind = 2.0 * math.pi
        world = {
            "Nx": Nx,
            "Ny": 1,
            "Nz": 1,
            "x_wind": x_wind,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "dx": x_wind / Nx,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 1.0e-3,
            "tile_shape": tile_shape,
            "guard_cells": 2,
            "boundary_conditions": {
                "x": BC_PERIODIC,
                "y": BC_PERIODIC,
                "z": BC_PERIODIC,
            },
            "field_boundary_conditions": (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            "grids": {},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"]["vertex"] = vertex_grid
        world["grids"]["center"] = center_grid
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world)
        world["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
        world["grids"]["tiled_center_grid"] = tiled_center_grid
        return world

    def _one_step_by_error(self, Nx, tile_shape):
        world = self._world(Nx, tile_shape)
        E, B, _J, _phi, _rho = initialize_fields(world)
        Ex, Ey, Ez = E
        g = int(world["guard_cells"])
        active = slice(g, -g)

        x_center = world["grids"]["tiled_center_grid"][0][:, :, :, active]
        Ez_values = jnp.sin(x_center)[:, :, :, :, jnp.newaxis, jnp.newaxis]
        Ez = Ez.at[:, :, :, active, active, active].set(Ez_values)

        _Bx, By, _Bz = update_B((Ex, Ey, Ez), B, world, {"alpha": 1.0})[0]

        x_vertex = world["grids"]["tiled_vertex_grid"][0][:, :, :, active]
        exact_By = world["dt"] * jnp.cos(x_vertex)
        diff = By[:, :, :, active, active, active] - exact_By[:, :, :, :, jnp.newaxis, jnp.newaxis]

        return float(jnp.sqrt(jnp.mean(diff**2)))

    def test_public_tiled_update_B_converges_for_one_tile_and_multi_tile_layouts(self):
        # A no-particle periodic E_z = sin(x) mode isolates the Yee curl
        # dE_z/dx used in the B_y update.  Errors are evaluated on tile-local
        # B_y interiors before any global field assembly.
        layouts = (
            lambda Nx: (Nx, 1, 1),
            lambda _Nx: (8, 1, 1),
            lambda _Nx: (16, 1, 1),
        )

        for layout in layouts:
            errors = []
            for Nx in (32, 64, 128):
                errors.append(self._one_step_by_error(Nx, layout(Nx)))

            first_order = math.log(errors[0] / errors[1], 2.0)
            second_order = math.log(errors[1] / errors[2], 2.0)

            self.assertGreater(first_order, 1.8)
            self.assertGreater(second_order, 1.8)


if __name__ == "__main__":
    unittest.main()
