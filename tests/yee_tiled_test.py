import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.initialization import initialize_fields
from PyPIC3D.electrodynamic_tiled import time_loop_electrodynamic_tiled
from PyPIC3D.solvers.first_order_yee import update_B, update_E
from PyPIC3D.solvers.yee_tiled import (
    assemble_tiled_vector_field,
    tile_vector_field,
    update_tiled_ghost_cells_periodic,
    update_tiled_vector_ghost_cells_periodic,
    update_tiled_B,
    update_tiled_E,
)
from PyPIC3D.utils import build_yee_grid
from PyPIC3D.boundary_conditions.boundaryconditions import update_ghost_cells


jax.config.update("jax_enable_x64", True)


def unused_curl(Ex, Ey, Ez):
    return None


class TestYeeTiled(unittest.TestCase):
    def _build_world(self):
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

    def _fill_ghosts(self, field, world):
        bc_x = world["boundary_conditions"]["x"]
        bc_y = world["boundary_conditions"]["y"]
        bc_z = world["boundary_conditions"]["z"]
        return update_ghost_cells(field, bc_x, bc_y, bc_z)

    def _deterministic_vector_field(self, world, scale):
        Nx, Ny, Nz = world["Nx"], world["Ny"], world["Nz"]
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

        return tuple(self._fill_ghosts(component, world) for component in (Fx, Fy, Fz))

    def test_tile_vector_field_assembles_to_original_ghost_celled_field(self):
        world = self._build_world()
        tile_shape = (2, 3, 2)
        E = self._deterministic_vector_field(world, scale=1.0)

        E_tiles = tile_vector_field(E, world, tile_shape)
        E_assembled = assemble_tiled_vector_field(E_tiles, world, tile_shape)

        for original, assembled in zip(E, E_assembled):
            self.assertTrue(jnp.allclose(assembled, original, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_ghost_cells_periodic_refreshes_neighbor_halos(self):
        world = self._build_world()
        tile_shape = (2, 3, 2)
        field = self._deterministic_vector_field(world, scale=1.0)[0]
        tiles = tile_vector_field((field,), world, tile_shape)[0]

        stale_tiles = tiles.at[:, :, :, 0, :, :].set(-100.0)
        stale_tiles = stale_tiles.at[:, :, :, -1, :, :].set(-200.0)
        stale_tiles = stale_tiles.at[:, :, :, :, 0, :].set(-300.0)
        stale_tiles = stale_tiles.at[:, :, :, :, -1, :].set(-400.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, 0].set(-500.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, -1].set(-600.0)

        refreshed = update_tiled_ghost_cells_periodic(stale_tiles)

        self.assertTrue(jnp.allclose(refreshed, tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_vector_ghost_cells_periodic_refreshes_each_component(self):
        world = self._build_world()
        tile_shape = (2, 3, 2)
        E = self._deterministic_vector_field(world, scale=1.0)
        E_tiles = tile_vector_field(E, world, tile_shape)

        stale_tiles = tuple(component.at[:, :, :, 0, :, :].set(-10.0 * (i + 1)) for i, component in enumerate(E_tiles))
        refreshed = update_tiled_vector_ghost_cells_periodic(stale_tiles)

        for original_tiles, refreshed_component in zip(E_tiles, refreshed):
            self.assertTrue(jnp.allclose(refreshed_component, original_tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_E_matches_standard_yee_update(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)

        E_reference, _ = update_E(E, B, J, world, constants, unused_curl)
        E_tiled = update_tiled_E(
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            world,
            constants,
            unused_curl,
            tile_shape,
        )
        E_from_tiles = assemble_tiled_vector_field(E_tiled, world, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_B_matches_standard_yee_update(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)

        B_reference, _ = update_B(E, B, world, constants, unused_curl)
        B_tiled = update_tiled_B(
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            world,
            constants,
            unused_curl,
            tile_shape,
        )
        B_from_tiles = assemble_tiled_vector_field(B_tiled, world, tile_shape)

        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_electrodynamic_step_matches_standard_yee_sequence(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)

        E_reference, _ = update_E(E, B, J, world, constants, unused_curl)
        B_reference, _ = update_B(E_reference, B, world, constants, unused_curl)

        E_tiles = tile_vector_field(E, world, tile_shape)
        B_tiles = tile_vector_field(B, world, tile_shape)
        J_tiles = tile_vector_field(J, world, tile_shape)
        E_tiles = update_tiled_E(E_tiles, B_tiles, J_tiles, world, constants, unused_curl, tile_shape)
        B_tiles = update_tiled_B(E_tiles, B_tiles, world, constants, unused_curl, tile_shape)

        E_from_tiles = assemble_tiled_vector_field(E_tiles, world, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, world, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_evolve_updates_fields_without_pushing_particles(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)

        E_reference, _ = update_E(E, B, J, world, constants, unused_curl)
        B_reference, _ = update_B(E_reference, B, world, constants, unused_curl)
        particles = object()
        external_fields = (
            tuple(jnp.zeros_like(component) for component in E),
            tuple(jnp.zeros_like(component) for component in B),
        )
        fields = (
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            jnp.zeros_like(E[0]),
            jnp.zeros_like(E[0]),
            external_fields,
            None,
        )

        particles_after, fields_after = time_loop_electrodynamic_tiled(
            particles,
            fields,
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="fdtd",
            tile_shape=tile_shape,
            relativistic=False,
            particle_pusher="boris",
        )

        E_tiles, B_tiles, J_tiles, rho, phi, external_after, pml_state = fields_after
        E_from_tiles = assemble_tiled_vector_field(E_tiles, world, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, world, tile_shape)

        self.assertIs(particles_after, particles)
        self.assertIsNone(pml_state)
        self.assertIs(external_after, external_fields)
        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for original, after in zip(fields[2], J_tiles):
            self.assertTrue(jnp.allclose(after, original, rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
