import inspect
import importlib
import importlib.util
import unittest

import jax.numpy as jnp

from PyPIC3D import evolve
from PyPIC3D.deposition import Esirkepov
from PyPIC3D.deposition import J_from_rhov
from PyPIC3D.deposition import rho
from PyPIC3D.boundary_conditions import boundaryconditions
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.diagnostics import output_adapters
from PyPIC3D.initialization import build_tiled_array, initialize_fields
from PyPIC3D.solvers import electrostatic_yee
from PyPIC3D.solvers import first_order_yee
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid


def tile_scalar_field(field, world, tile_shape, num_guard_cells=2):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    ntx = int(world["Nx"]) // tile_nx
    nty = int(world["Ny"]) // tile_ny
    ntz = int(world["Nz"]) // tile_nz

    field_tiles = jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=field.dtype,
    )
    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                ix = 1 + tx * tile_nx
                iy = 1 + ty * tile_ny
                iz = 1 + tz * tile_nz
                interior = field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                field_tiles = field_tiles.at[tx, ty, tz, g:-g, g:-g, g:-g].set(interior)

    return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g, tile_shape)


class TestTiledRefactorContracts(unittest.TestCase):
    def _world(self):
        world = {
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
            "tile_shape": (2, 3, 2),
            "guard_cells": 2,
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world)
        world["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
        world["grids"]["tiled_center_grid"] = tiled_center_grid
        return world

    def _field(self, world, scale):
        Nx, Ny, Nz = world["Nx"], world["Ny"], world["Nz"]
        ii, jj, kk = jnp.meshgrid(
            jnp.arange(Nx, dtype=float),
            jnp.arange(Ny, dtype=float),
            jnp.arange(Nz, dtype=float),
            indexing="ij",
        )
        shape = (Nx + 2, Ny + 2, Nz + 2)
        return jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (ii + 2.0 * jj - kk))

    def test_tiled_vector_ghost_refresh_accepts_tuple_field_state(self):
        world = self._world()
        tile_shape = (2, 3, 2)
        E = tuple(self._field(world, scale) for scale in (1.0, -0.5, 0.25))
        E_tiles = tuple(tile_scalar_field(component, world, tile_shape) for component in E)

        refreshed = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, world, num_guard_cells=2, tile_shape=tile_shape)
        self.assertEqual(len(refreshed), 3)
        for component in refreshed:
            self.assertEqual(component.shape, (4, 2, 2, 6, 7, 6))

    def test_evolve_loops_read_runtime_controls_from_world(self):
        electrodynamic_signature = inspect.signature(evolve.time_loop_electrodynamic)
        electrostatic_signature = inspect.signature(evolve.time_loop_electrostatic)
        for name in ("curl_func", "J_func", "tile_shape", "g", "current_deposition"):
            self.assertNotIn(name, electrodynamic_signature.parameters)
        for name in ("curl_func", "J_func", "tile_shape", "g"):
            self.assertNotIn(name, electrostatic_signature.parameters)

    def test_tiled_yee_updates_read_tile_metadata_from_world(self):
        update_E_signature = inspect.signature(first_order_yee.update_E)
        update_B_signature = inspect.signature(first_order_yee.update_B)
        for name in ("curl_func", "tile_shape", "g"):
            self.assertNotIn(name, update_E_signature.parameters)
            self.assertNotIn(name, update_B_signature.parameters)
        self.assertFalse(hasattr(first_order_yee, "update_tiled_E"))
        self.assertFalse(hasattr(first_order_yee, "update_tiled_B"))
        self.assertFalse(hasattr(first_order_yee, "tiled_grid_axes_from_world"))
        self.assertFalse(hasattr(first_order_yee, "tile_grid_axes"))
        self.assertFalse(hasattr(first_order_yee, "tile_vector_field"))
        self.assertFalse(hasattr(first_order_yee, "stack_tiled_vector_field"))
        self.assertFalse(hasattr(first_order_yee, "unstack_tiled_vector_field"))
        self.assertFalse(hasattr(first_order_yee, "update_tiled_ghost_cells"))
        self.assertFalse(hasattr(first_order_yee, "update_tiled_vector_ghost_cells"))
        self.assertFalse(hasattr(first_order_yee, "update_tiled_ghost_cells_for_pml"))
        self.assertFalse(hasattr(first_order_yee, "update_tiled_vector_ghost_cells_for_pml"))
        self.assertFalse(hasattr(first_order_yee, "fold_tiled_ghost_cells"))
        self.assertFalse(hasattr(first_order_yee, "fold_tiled_vector_ghost_cells"))
        self.assertFalse(hasattr(first_order_yee, "apply_tiled_conducting_bc"))
        self.assertFalse(hasattr(first_order_yee, "update_tiled_ghost_cells_periodic"))
        self.assertFalse(hasattr(first_order_yee, "update_tiled_vector_ghost_cells_periodic"))
        self.assertFalse(hasattr(first_order_yee, "fold_tiled_ghost_cells_periodic"))
        self.assertFalse(hasattr(first_order_yee, "fold_tiled_vector_ghost_cells_periodic"))
        self.assertFalse(hasattr(first_order_yee, "empty_tiled_scalar_field"))
        self.assertFalse(hasattr(first_order_yee, "empty_tiled_vector_field"))
        self.assertFalse(hasattr(first_order_yee, "tile_scalar_field"))
        self.assertFalse(hasattr(first_order_yee, "assemble_tiled_scalar_field"))
        self.assertFalse(hasattr(first_order_yee, "assemble_tiled_vector_field"))

    def test_global_assembly_lives_at_output_boundary(self):
        self.assertTrue(hasattr(output_adapters, "assemble_tiled_scalar_field"))
        self.assertTrue(hasattr(output_adapters, "assemble_tiled_vector_field"))

        source = inspect.getsource(output_adapters)
        self.assertNotIn("from PyPIC3D.solvers.yee_tiled import", source)

    def test_ghost_cell_logic_lives_in_boundary_conditions_ghost_cells(self):
        for name in (
            "update_tiled_ghost_cells",
            "update_tiled_vector_ghost_cells",
            "update_tiled_ghost_cells_for_pml",
            "update_tiled_vector_ghost_cells_for_pml",
            "fold_tiled_ghost_cells",
            "fold_tiled_vector_ghost_cells",
            "apply_tiled_conducting_bc",
        ):
            self.assertTrue(hasattr(ghost_cells, name))

        for name in (
            "update_ghost_cells",
            "fold_ghost_cells",
            "apply_conducting_bc",
            "apply_scalar_conducting_bc",
        ):
            self.assertFalse(hasattr(boundaryconditions, name))

    def test_tiled_array_initialization_lives_in_initialization(self):
        world = self._world()
        world["tile_shape"] = (2, 3, 2)
        world["guard_cells"] = 2

        array = build_tiled_array(world, dtype=jnp.float32)
        self.assertEqual(array.shape, (4, 2, 2, 6, 7, 6))
        self.assertEqual(array.dtype, jnp.float32)
        self.assertTrue(jnp.all(array == 0.0))

    def test_initialize_fields_builds_tiled_field_state_from_world(self):
        world = self._world()
        world["tile_shape"] = (2, 3, 2)
        world["guard_cells"] = 2

        E, B, J, phi, rho_tiles = initialize_fields(world)
        for vector_field in (E, B, J):
            self.assertEqual(len(vector_field), 3)
            for component in vector_field:
                self.assertEqual(component.shape, (4, 2, 2, 6, 7, 6))
                self.assertEqual(component.dtype, jnp.float64)
                self.assertTrue(jnp.all(component == 0.0))
        for scalar_field in (phi, rho_tiles):
            self.assertEqual(scalar_field.shape, (4, 2, 2, 6, 7, 6))
            self.assertEqual(scalar_field.dtype, jnp.float64)
            self.assertTrue(jnp.all(scalar_field == 0.0))

    def test_grid_builders_live_in_utilities(self):
        from PyPIC3D import utils
        from PyPIC3D.utilities import grids

        self.assertFalse(hasattr(utils, "build_yee_grid"))
        self.assertFalse(hasattr(utils, "build_collocated_grid"))
        self.assertTrue(hasattr(grids, "build_yee_grid"))
        self.assertTrue(hasattr(grids, "build_collocated_grid"))
        self.assertFalse(hasattr(grids, "tile_grid_axes"))
        self.assertTrue(hasattr(grids, "build_tiled_yee_grids"))

    def test_tiled_benchmarking_module_is_removed(self):
        self.assertIsNone(importlib.util.find_spec("PyPIC3D.diagnostics.tiled_benchmarking"))

    def test_particle_push_owns_tiled_pusher_contract(self):
        particle_push = importlib.import_module("PyPIC3D.pusher.particle_push")

        self.assertIsNone(importlib.util.find_spec("PyPIC3D.pusher.tiled_pusher"))
        signature = inspect.signature(particle_push.particle_push)
        for name in ("grid", "staggered_grid", "tile_shape", "g"):
            self.assertNotIn(name, signature.parameters)

        source = inspect.getsource(particle_push.particle_push)
        self.assertIn('world["grids"]["tiled_center_grid"]', source)
        self.assertIn('world["grids"]["tiled_vertex_grid"]', source)
        self.assertNotIn("tiled_grid_axes_from_world", source)
        self.assertNotIn("tile_grid_axes", source)
        self.assertNotIn("get_forward_position", source)
        self.assertNotIn("set_velocity", source)

    def test_electrodynamic_hot_step_does_not_assemble_global_fields(self):
        source = inspect.getsource(evolve.time_loop_electrodynamic)
        self.assertNotIn("assemble_tiled_scalar_field", source)
        self.assertNotIn("assemble_tiled_vector_field", source)
        self.assertNotIn("fields_for_output", source)

    def test_electrostatic_solver_uses_single_tile_arrays_directly(self):
        source = inspect.getsource(electrostatic_yee.calculate_tiled_electrostatic_fields)
        self.assertNotIn("assemble_tiled_scalar_field", source)
        self.assertNotIn("tile_scalar_field", source)
        self.assertIn("[0, 0, 0]", source)

    def test_flat_current_and_electrodynamic_references_are_removed(self):
        self.assertFalse(hasattr(J_from_rhov, "_J_from_rhov_flat"))
        self.assertFalse(hasattr(J_from_rhov, "_J_from_rhov_tiled"))
        self.assertFalse(hasattr(evolve, "_time_loop_electrodynamic_global_reference"))
        self.assertFalse(hasattr(evolve, "_time_loop_electrostatic_global_reference"))

    def test_direct_current_api_reads_tile_metadata_from_world(self):
        signature = inspect.signature(J_from_rhov.J_from_rhov)
        self.assertNotIn("tile_shape", signature.parameters)
        self.assertNotIn("g", signature.parameters)

    def test_esirkepov_api_reads_tile_metadata_from_world(self):
        signature = inspect.signature(Esirkepov.Esirkepov_current)
        self.assertNotIn("tile_shape", signature.parameters)
        self.assertNotIn("g", signature.parameters)

    def test_tiled_esirkepov_module_is_removed(self):
        self.assertIsNone(importlib.util.find_spec("PyPIC3D.deposition.esirkepov_tiled"))

    def test_rho_api_reads_tile_metadata_from_world(self):
        signature = inspect.signature(rho.compute_rho)
        self.assertNotIn("tile_shape", signature.parameters)
        self.assertNotIn("g", signature.parameters)

    def test_flat_rho_and_tiled_rho_module_are_removed(self):
        self.assertFalse(hasattr(rho, "_compute_rho_flat"))
        self.assertIsNone(importlib.util.find_spec("PyPIC3D.deposition.rho_tiled"))

    def test_current_methods_module_is_removed(self):
        self.assertIsNone(importlib.util.find_spec("PyPIC3D.deposition.current_methods"))


if __name__ == "__main__":
    unittest.main()
