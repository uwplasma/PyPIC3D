import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from tests.support.compiled_yee import (
    update_B,
    update_E,
    yee_curl_b_to_e,
    yee_curl_e_to_b,
    yee_derivatives_b_to_e,
    yee_derivatives_e_to_b,
)
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC
from tests.kernel_fixtures import kernel_parameters_from_values, tile_vector_field


def _update_ghost_cells(field, bc_x, bc_y, bc_z):
    field = jax.lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[0, :, :].set(f[-2, :, :]).at[-1, :, :].set(f[1, :, :]),
        lambda f: f.at[0, :, :].set(0.0).at[-1, :, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 0, :].set(f[:, -2, :]).at[:, -1, :].set(f[:, 1, :]),
        lambda f: f.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 0].set(f[:, :, -2]).at[:, :, -1].set(f[:, :, 1]),
        lambda f: f.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0),
        operand=field,
    )
    return field


def _field_dot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(a, b))


class YeeTiledFixtures:
    def _build_parameter_values(self):
        parameter_set = {
            "Nx": 8,
            "Ny": 6,
            "Nz": 4,
            "dx": 0.5,
            "dy": 0.5,
            "dz": 0.5,
            "dt": 0.05,
            "x_wind": 4.0,
            "y_wind": 3.0,
            "z_wind": 2.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        return parameter_set

    def _conducting_parameters(self):
        parameter_set = self._build_parameter_values()
        parameter_set["boundary_conditions"] = {"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING}
        return parameter_set

    def _constant_x_parameters(self):
        parameter_set = self._build_parameter_values()
        parameter_set["boundary_conditions"] = {"x": BC_CONSTANT, "y": BC_PERIODIC, "z": BC_PERIODIC}
        return parameter_set

    def _assert_x_ghosts_are_constant(self, vector_field, g):
        for component in vector_field:
            self.assertTrue(jnp.allclose(component[:, :, :, :g, :, :], component[:, :, :, g:g + 1, :, :]))
            self.assertTrue(jnp.allclose(component[:, :, :, -g:, :, :], component[:, :, :, -g - 1:-g, :, :]))

    def _with_tile_metadata(self, parameter_set, tile_shape, g=2):
        parameter_set["tile_shape"] = tuple(int(width) for width in tile_shape)
        parameter_set["guard_cells"] = int(g)
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((
            int(parameter_set["Nx"]) // int(tile_shape[0]),
            int(parameter_set["Ny"]) // int(tile_shape[1]),
            int(parameter_set["Nz"]) // int(tile_shape[2]),
        ))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
        parameter_set["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
        parameter_set["grids"]["tiled_center_grid"] = tiled_center_grid
        return parameter_set

    def _fill_ghosts(self, field, parameter_set):
        bc_x = parameter_set["boundary_conditions"]["x"]
        bc_y = parameter_set["boundary_conditions"]["y"]
        bc_z = parameter_set["boundary_conditions"]["z"]
        return _update_ghost_cells(field, bc_x, bc_y, bc_z)

    def _deterministic_vector_field(self, parameter_set, scale):
        Nx, Ny, Nz = parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"]
        ii, jj, kk = jnp.meshgrid(
            jnp.arange(Nx, dtype=float),
            jnp.arange(Ny, dtype=float),
            jnp.arange(Nz, dtype=float),
            indexing="ij",
        )

        shape = (Nx + 2, Ny + 2, Nz + 2)
        Fx = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (0.2 + 0.03 * ii - 0.02 * jj + 0.04 * kk))
        Fy = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (-0.1 + 0.05 * ii + 0.01 * jj - 0.03 * kk))
        Fz = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (0.3 - 0.04 * ii + 0.02 * jj + 0.01 * kk))

        return tuple(self._fill_ghosts(component, parameter_set) for component in (Fx, Fy, Fz))

    def _random_tiled_vector_field(self, parameter_set, tile_shape, seed):
        template = tile_vector_field(
            self._deterministic_vector_field(parameter_set, scale=1.0),
            parameter_set,
            tile_shape,
        )
        keys = jax.random.split(jax.random.key(seed), 3)
        return tuple(
            jax.random.normal(key, component.shape, dtype=jnp.float64)
            for key, component in zip(keys, template)
        )

    def _copy_parameters_for_tile_shape(self, parameter_set, tile_shape, g=2):
        reference_parameters = dict(parameter_set)
        reference_parameters["boundary_conditions"] = dict(parameter_set["boundary_conditions"])
        reference_parameters["grids"] = dict(parameter_set["grids"])
        reference_parameters["tile_shape"] = tuple(int(width) for width in tile_shape)
        reference_parameters["guard_cells"] = int(g)
        reference_parameters["field_mesh"] = ghost_cells.make_field_mesh((
            int(reference_parameters["Nx"]) // int(tile_shape[0]),
            int(reference_parameters["Ny"]) // int(tile_shape[1]),
            int(reference_parameters["Nz"]) // int(tile_shape[2]),
        ))
        return reference_parameters

    def _split_parameters(self, parameter_set, dynamic_values):
        return kernel_parameters_from_values(parameter_set, dynamic_values)

    def _reference_update_E(self, E, B, J, parameter_set, dynamic_values):
        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, tile_shape, int(parameter_set["guard_cells"]))
        static_parameters, dynamic_parameters = self._split_parameters(reference_parameters, dynamic_values)
        E_reference, pml_state = update_E(
            tile_vector_field(E, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            tile_vector_field(B, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            tile_vector_field(J, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            static_parameters,
            dynamic_parameters,
        )
        return assemble_tiled_vector_field(
            E_reference,
            reference_parameters,
            tile_shape,
            num_guard_cells=int(reference_parameters["guard_cells"]),
        ), pml_state

    def _reference_update_B(self, E, B, parameter_set, dynamic_values):
        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, tile_shape, int(parameter_set["guard_cells"]))
        static_parameters, dynamic_parameters = self._split_parameters(reference_parameters, dynamic_values)
        B_reference, pml_state = update_B(
            tile_vector_field(E, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            tile_vector_field(B, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            static_parameters,
            dynamic_parameters,
        )
        return assemble_tiled_vector_field(
            B_reference,
            reference_parameters,
            tile_shape,
            num_guard_cells=int(reference_parameters["guard_cells"]),
        ), pml_state

    def _reference_yee_step(self, E, B, J, parameter_set, dynamic_values):
        E_reference, pml_state = self._reference_update_E(E, B, J, parameter_set, dynamic_values)
        B_reference, pml_state = self._reference_update_B(E_reference, B, parameter_set, dynamic_values)
        return E_reference, B_reference, pml_state
