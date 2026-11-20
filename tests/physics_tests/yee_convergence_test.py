import math
import unittest
from types import SimpleNamespace

import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.initialization import initialize_fields
from tests.support.compiled_yee import update_B
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.kernel_fixtures import kernel_parameters_from_values


class TestYeeConvergence(unittest.TestCase):
    def _parameter_values(self, Nx):
        tile_shape = (Nx, 1, 1)
        x_wind = 2.0 * math.pi
        parameter_set = {
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
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"]["center"] = center_grid
        parameter_set["grids"]["vertex"] = vertex_grid
        parameter_set["field_mesh"] = make_field_mesh((
            int(parameter_set["Nx"]) // int(tile_shape[0]),
            int(parameter_set["Ny"]) // int(tile_shape[1]),
            int(parameter_set["Nz"]) // int(tile_shape[2]),
        ))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, {"alpha": 1.0})
        tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
        parameter_set["grids"]["tiled_center_grid"] = tiled_center_grid
        parameter_set["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
        return parameter_set

    def call_updateB(self, Nx):
        parameter_set = self._parameter_values(Nx)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, {"alpha": 1.0})
        E, B, _J, _phi, _rho = initialize_fields(static_parameters, dynamic_parameters)
        Ex, Ey, Ez = E
        g = int(parameter_set["guard_cells"])
        active = slice(g, -g)

        x_center = parameter_set["grids"]["tiled_center_grid"][0][:, :, :, active]
        Ez_values = jnp.sin(x_center)[:, :, :, :, jnp.newaxis, jnp.newaxis]
        Ez = Ez.at[:, :, :, active, active, active].set(Ez_values)

        _Bx, By, _Bz = update_B((Ex, Ey, Ez), B, static_parameters, dynamic_parameters)[0]
        _Bx, By, _Bz = update_B((Ex, Ey, Ez), (_Bx, By, _Bz), static_parameters, dynamic_parameters)[0]
        # perform two updates to ensure that the B field is updated with the correct Ez values

        x_vertex = parameter_set["grids"]["tiled_vertex_grid"][0][:, :, :, active]
        exact_By = parameter_set["dt"] * jnp.cos(x_vertex)
        diff = By[:, :, :, active, active, active] - exact_By[:, :, :, :, jnp.newaxis, jnp.newaxis]

        return float(jnp.sqrt(jnp.mean(diff**2)))

    def test_public_update_B_converges_on_one_tile(self):
        # E_z = sin(x) isolates the Yee derivative at the native B_y locations.
        errors = [self.call_updateB(Nx) for Nx in (32, 64, 128)]
        first_order = math.log(errors[0] / errors[1], 2.0)
        second_order = math.log(errors[1] / errors[2], 2.0)
        self.assertGreater(first_order, 1.8)
        self.assertGreater(second_order, 1.8)


if __name__ == "__main__":
    unittest.main()
