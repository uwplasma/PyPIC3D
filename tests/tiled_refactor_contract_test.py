import inspect
import unittest

import jax.numpy as jnp

from PyPIC3D import evolve
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
        self.assertEqual(stage_specs["tiled_pic_step"].static_argnames, ("J_func", "relativistic", "particle_pusher"))
        self.assertEqual(stage_specs["tiled_current_deposition"].static_argnames, ("J_func",))

    def test_electrodynamic_hot_step_does_not_assemble_global_fields(self):
        source = inspect.getsource(evolve.time_loop_electrodynamic)
        self.assertNotIn("assemble_tiled_scalar_field", source)
        self.assertNotIn("assemble_tiled_vector_field", source)
        self.assertNotIn("fields_for_output", source)

    def test_electrostatic_global_poisson_bridge_stays_explicit(self):
        source = inspect.getsource(electrostatic_yee.calculate_tiled_electrostatic_fields)
        self.assertIn("assemble_tiled_scalar_field", source)
        self.assertIn("tile_scalar_field", source)


if __name__ == "__main__":
    unittest.main()
