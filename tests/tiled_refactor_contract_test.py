import inspect
import importlib.util
import unittest

import jax.numpy as jnp

from PyPIC3D import evolve
from PyPIC3D.deposition import Esirkepov
from PyPIC3D.deposition import J_from_rhov
from PyPIC3D.deposition import rho
from PyPIC3D.solvers import electrostatic_yee
from PyPIC3D.solvers import yee_tiled
from PyPIC3D.utils import build_yee_grid


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
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
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

    def test_tiled_vector_field_stack_roundtrips_and_refreshes_halos(self):
        world = self._world()
        tile_shape = (2, 3, 2)
        E = tuple(self._field(world, scale) for scale in (1.0, -0.5, 0.25))
        E_tiles = yee_tiled.tile_vector_field(E, world, tile_shape)

        stacked = yee_tiled.stack_tiled_vector_field(E_tiles)
        self.assertEqual(stacked.shape[0], 3)

        unstacked = yee_tiled.unstack_tiled_vector_field(stacked)
        for original, recovered in zip(E_tiles, unstacked):
            self.assertTrue(jnp.allclose(original, recovered, rtol=1.0e-12, atol=1.0e-12))

        refreshed = yee_tiled.update_tiled_vector_ghost_cells(E_tiles, world, num_guard_cells=1, tile_shape=tile_shape)
        refreshed_from_stack = yee_tiled.unstack_tiled_vector_field(
            yee_tiled.update_tiled_vector_ghost_cells(stacked, world, num_guard_cells=1, tile_shape=tile_shape)
        )
        for reference, candidate in zip(refreshed, refreshed_from_stack):
            self.assertTrue(jnp.allclose(reference, candidate, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_benchmark_stage_specs_expose_tiled_hot_paths(self):
        from PyPIC3D.diagnostics import tiled_benchmarking

        case = tiled_benchmarking.build_synthetic_tiled_yee_state(
            nx=4,
            ny=4,
            nz=4,
            particles_per_species=8,
            species_count=1,
            tile_shape=(2, 2, 2),
            slots_per_tile=8,
        )
        stage_specs = tiled_benchmarking.build_tiled_stage_specs(case)

        expected = {
            "tiled_pic_step",
            "tiled_particle_push",
            "tiled_particle_retile",
            "tiled_current_deposition",
            "tiled_field_update",
            "tiled_diagnostics",
            "tiled_output_bridge",
        }
        self.assertTrue(expected.issubset(stage_specs))
        self.assertEqual(stage_specs["tiled_pic_step"].static_argnames, ("relativistic", "particle_pusher"))
        self.assertEqual(stage_specs["tiled_current_deposition"].static_argnames, ())

    def test_evolve_loops_read_runtime_controls_from_world(self):
        electrodynamic_signature = inspect.signature(evolve.time_loop_electrodynamic)
        electrostatic_signature = inspect.signature(evolve.time_loop_electrostatic)
        for name in ("curl_func", "J_func", "tile_shape", "g", "current_deposition"):
            self.assertNotIn(name, electrodynamic_signature.parameters)
        for name in ("curl_func", "J_func", "tile_shape", "g"):
            self.assertNotIn(name, electrostatic_signature.parameters)

    def test_tiled_yee_updates_read_tile_metadata_from_world(self):
        update_E_signature = inspect.signature(yee_tiled.update_E)
        update_B_signature = inspect.signature(yee_tiled.update_B)
        for name in ("curl_func", "tile_shape", "g"):
            self.assertNotIn(name, update_E_signature.parameters)
            self.assertNotIn(name, update_B_signature.parameters)
        self.assertFalse(hasattr(yee_tiled, "update_tiled_E"))
        self.assertFalse(hasattr(yee_tiled, "update_tiled_B"))

    def test_electrodynamic_hot_step_does_not_assemble_global_fields(self):
        source = inspect.getsource(evolve.time_loop_electrodynamic)
        self.assertNotIn("assemble_tiled_scalar_field", source)
        self.assertNotIn("assemble_tiled_vector_field", source)
        self.assertNotIn("fields_for_output", source)

    def test_electrostatic_global_poisson_bridge_stays_explicit(self):
        source = inspect.getsource(electrostatic_yee.calculate_tiled_electrostatic_fields)
        self.assertIn("assemble_tiled_scalar_field", source)
        self.assertIn("tile_scalar_field", source)

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
