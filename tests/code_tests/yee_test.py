import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.solvers.first_order_yee import (
    update_B,
    update_E,
)
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from tests.parameter_helpers import split_test_parameters


jax.config.update("jax_enable_x64", True)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, world, tile_shape, num_guard_cells=2):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

    if g != 1:
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
        world = dict(world)
        world["tile_shape"] = tuple(int(width) for width in tile_shape)
        world["field_mesh"] = ghost_cells.make_field_mesh((ntx, nty, ntz))
        return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g)

    def tile_at(tx, ty, tz):
        start = (tx * tile_nx, ty * tile_ny, tz * tile_nz)
        size = (tile_nx + 2, tile_ny + 2, tile_nz + 2)
        return jax.lax.dynamic_slice(field, start, size)

    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack([tile_at(tx, ty, tz) for tz in range(ntz)], axis=0)
                    for ty in range(nty)
                ],
                axis=0,
            )
            for tx in range(ntx)
        ],
        axis=0,
    )


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


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

    def _conducting_world(self):
        world = self._build_world()
        world["boundary_conditions"] = {"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING}
        return world

    def _with_tile_metadata(self, world, tile_shape, g=2):
        world["tile_shape"] = tuple(int(width) for width in tile_shape)
        world["guard_cells"] = int(g)
        world["field_mesh"] = ghost_cells.make_field_mesh((
            int(world["Nx"]) // int(tile_shape[0]),
            int(world["Ny"]) // int(tile_shape[1]),
            int(world["Nz"]) // int(tile_shape[2]),
        ))
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world, tile_shape, g)
        world["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
        world["grids"]["tiled_center_grid"] = tiled_center_grid
        return world

    def _mixed_bc_world(self):
        world = self._build_world()
        world["boundary_conditions"] = {"x": BC_PERIODIC, "y": BC_CONDUCTING, "z": BC_CONDUCTING}
        return world

    def _fill_ghosts(self, field, world):
        bc_x = world["boundary_conditions"]["x"]
        bc_y = world["boundary_conditions"]["y"]
        bc_z = world["boundary_conditions"]["z"]
        return _update_ghost_cells(field, bc_x, bc_y, bc_z)

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

    def _copy_world_for_tile_shape(self, world, tile_shape, g=2):
        reference_world = dict(world)
        reference_world["boundary_conditions"] = dict(world["boundary_conditions"])
        reference_world["grids"] = dict(world["grids"])
        reference_world["tile_shape"] = tuple(int(width) for width in tile_shape)
        reference_world["guard_cells"] = int(g)
        reference_world["field_mesh"] = ghost_cells.make_field_mesh((
            int(reference_world["Nx"]) // int(tile_shape[0]),
            int(reference_world["Ny"]) // int(tile_shape[1]),
            int(reference_world["Nz"]) // int(tile_shape[2]),
        ))
        return reference_world

    def _split_parameters(self, world, constants):
        return split_test_parameters(world, constants)

    def _reference_update_E(self, E, B, J, world, constants):
        tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._copy_world_for_tile_shape(world, tile_shape, int(world["guard_cells"]))
        static_parameters, dynamic_parameters = self._split_parameters(reference_world, constants)
        E_reference, pml_state = update_E(
            tile_vector_field(E, reference_world, tile_shape, num_guard_cells=int(reference_world["guard_cells"])),
            tile_vector_field(B, reference_world, tile_shape, num_guard_cells=int(reference_world["guard_cells"])),
            tile_vector_field(J, reference_world, tile_shape, num_guard_cells=int(reference_world["guard_cells"])),
            static_parameters,
            dynamic_parameters,
        )
        return assemble_tiled_vector_field(
            E_reference,
            reference_world,
            tile_shape,
            num_guard_cells=int(reference_world["guard_cells"]),
        ), pml_state

    def _reference_update_B(self, E, B, world, constants):
        tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._copy_world_for_tile_shape(world, tile_shape, int(world["guard_cells"]))
        static_parameters, dynamic_parameters = self._split_parameters(reference_world, constants)
        B_reference, pml_state = update_B(
            tile_vector_field(E, reference_world, tile_shape, num_guard_cells=int(reference_world["guard_cells"])),
            tile_vector_field(B, reference_world, tile_shape, num_guard_cells=int(reference_world["guard_cells"])),
            static_parameters,
            dynamic_parameters,
        )
        return assemble_tiled_vector_field(
            B_reference,
            reference_world,
            tile_shape,
            num_guard_cells=int(reference_world["guard_cells"]),
        ), pml_state

    def _reference_yee_step(self, E, B, J, world, constants):
        E_reference, pml_state = self._reference_update_E(E, B, J, world, constants)
        B_reference, pml_state = self._reference_update_B(E_reference, B, world, constants)
        return E_reference, B_reference, pml_state

    def _fill_guard_cells(self, field, world, num_guard_cells):
        g = num_guard_cells
        bc_x = world["boundary_conditions"]["x"]
        bc_y = world["boundary_conditions"]["y"]
        bc_z = world["boundary_conditions"]["z"]

        if bc_x == BC_PERIODIC:
            field = field.at[:g, :, :].set(field[-2 * g:-g, :, :])
            field = field.at[-g:, :, :].set(field[g:2 * g, :, :])
        else:
            field = field.at[:g, :, :].set(0.0)
            field = field.at[-g:, :, :].set(0.0)

        if bc_y == BC_PERIODIC:
            field = field.at[:, :g, :].set(field[:, -2 * g:-g, :])
            field = field.at[:, -g:, :].set(field[:, g:2 * g, :])
        else:
            field = field.at[:, :g, :].set(0.0)
            field = field.at[:, -g:, :].set(0.0)

        if bc_z == BC_PERIODIC:
            field = field.at[:, :, :g].set(field[:, :, -2 * g:-g])
            field = field.at[:, :, -g:].set(field[:, :, g:2 * g])
        else:
            field = field.at[:, :, :g].set(0.0)
            field = field.at[:, :, -g:].set(0.0)

        return field

    def _tile_scalar_field_with_guard(self, field, tile_shape, num_guard_cells):
        g = num_guard_cells
        tile_nx, tile_ny, tile_nz = tile_shape
        Nx = int(field.shape[0]) - 2 * g
        Ny = int(field.shape[1]) - 2 * g
        Nz = int(field.shape[2]) - 2 * g
        ntx = Nx // tile_nx
        nty = Ny // tile_ny
        ntz = Nz // tile_nz

        tiles = []
        for tx in range(ntx):
            y_tiles = []
            for ty in range(nty):
                z_tiles = []
                for tz in range(ntz):
                    ix = tx * tile_nx
                    iy = ty * tile_ny
                    iz = tz * tile_nz
                    z_tiles.append(
                        field[
                            ix:ix + tile_nx + 2 * g,
                            iy:iy + tile_ny + 2 * g,
                            iz:iz + tile_nz + 2 * g,
                        ]
                    )
                y_tiles.append(jnp.stack(z_tiles, axis=0))
            tiles.append(jnp.stack(y_tiles, axis=0))

        return jnp.stack(tiles, axis=0)

    def test_tile_vector_field_assembles_to_original_ghost_celled_field(self):
        world = self._build_world()
        tile_shape = (2, 3, 2)
        E = self._deterministic_vector_field(world, scale=1.0)

        E_tiles = tile_vector_field(E, world, tile_shape)
        E_assembled = assemble_tiled_vector_field(E_tiles, world, tile_shape)

        for original, assembled in zip(E, E_assembled):
            self.assertTrue(jnp.allclose(assembled, original, rtol=1.0e-12, atol=1.0e-12))

    def test_tile_grid_axes_include_configured_guard_cells(self):
        world = self._build_world()
        tile_shape = (2, 3, 2)
        g = 2
        world = self._with_tile_metadata(world, tile_shape, g)
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world)

        self.assertEqual(tiled_center_grid[0].shape, (4, 2, 2, tile_shape[0] + 2 * g))
        self.assertEqual(tiled_center_grid[1].shape, (4, 2, 2, tile_shape[1] + 2 * g))
        self.assertEqual(tiled_center_grid[2].shape, (4, 2, 2, tile_shape[2] + 2 * g))

        for tx in range(4):
            for ty in range(2):
                for tz in range(2):
                    center_x = world["grids"]["center"][0][0] + (
                        jnp.arange(tile_shape[0] + 2 * g) + tx * tile_shape[0] - (g - 1)
                    ) * world["dx"]
                    vertex_y = world["grids"]["vertex"][1][0] + (
                        jnp.arange(tile_shape[1] + 2 * g) + ty * tile_shape[1] - (g - 1)
                    ) * world["dy"]

                    self.assertTrue(jnp.allclose(tiled_center_grid[0][tx, ty, tz], center_x))
                    self.assertTrue(jnp.allclose(tiled_vertex_grid[1][tx, ty, tz], vertex_y))

    def test_update_tiled_ghost_cells_periodic_refreshes_neighbor_halos(self):
        world = self._build_world()
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape, g=2)
        field = self._deterministic_vector_field(world, scale=1.0)[0]
        tiles = tile_vector_field((field,), world, tile_shape)[0]

        stale_tiles = tiles.at[:, :, :, 0, :, :].set(-100.0)
        stale_tiles = stale_tiles.at[:, :, :, -1, :, :].set(-200.0)
        stale_tiles = stale_tiles.at[:, :, :, :, 0, :].set(-300.0)
        stale_tiles = stale_tiles.at[:, :, :, :, -1, :].set(-400.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, 0].set(-500.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, -1].set(-600.0)

        refreshed = ghost_cells.update_tiled_ghost_cells(stale_tiles, world, num_guard_cells=2)

        self.assertTrue(jnp.allclose(refreshed, tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_vector_ghost_cells_periodic_refreshes_each_component(self):
        world = self._build_world()
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape, g=2)
        E = self._deterministic_vector_field(world, scale=1.0)
        E_tiles = tile_vector_field(E, world, tile_shape)

        stale_tiles = tuple(component.at[:, :, :, 0, :, :].set(-10.0 * (i + 1)) for i, component in enumerate(E_tiles))
        refreshed = ghost_cells.update_tiled_vector_ghost_cells(stale_tiles, world, num_guard_cells=2)

        for original_tiles, refreshed_component in zip(E_tiles, refreshed):
            self.assertTrue(jnp.allclose(refreshed_component, original_tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_ghost_cells_conducting_matches_global_ghost_cells(self):
        world = self._conducting_world()
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape, g=2)
        field = self._deterministic_vector_field(world, scale=1.0)[0]
        tiles = tile_vector_field((field,), world, tile_shape)[0]

        stale_tiles = tiles.at[:, :, :, 0, :, :].set(-100.0)
        stale_tiles = stale_tiles.at[:, :, :, -1, :, :].set(-200.0)
        stale_tiles = stale_tiles.at[:, :, :, :, 0, :].set(-300.0)
        stale_tiles = stale_tiles.at[:, :, :, :, -1, :].set(-400.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, 0].set(-500.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, -1].set(-600.0)

        refreshed = ghost_cells.update_tiled_ghost_cells(stale_tiles, world)
        reference = _update_ghost_cells(
            field,
            world["boundary_conditions"]["x"],
            world["boundary_conditions"]["y"],
            world["boundary_conditions"]["z"],
        )
        reference_tiles = tile_vector_field((reference,), world, tile_shape)[0]

        self.assertTrue(jnp.allclose(refreshed, reference_tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_ghost_cells_mixed_matches_global_ghost_cells(self):
        world = self._mixed_bc_world()
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape, g=2)
        field = self._deterministic_vector_field(world, scale=1.0)[0]
        tiles = tile_vector_field((field,), world, tile_shape)[0]

        stale_tiles = tiles.at[:, :, :, 0, :, :].set(-100.0)
        stale_tiles = stale_tiles.at[:, :, :, -1, :, :].set(-200.0)
        stale_tiles = stale_tiles.at[:, :, :, :, 0, :].set(-300.0)
        stale_tiles = stale_tiles.at[:, :, :, :, -1, :].set(-400.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, 0].set(-500.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, -1].set(-600.0)

        refreshed = ghost_cells.update_tiled_ghost_cells(stale_tiles, world)
        reference = _update_ghost_cells(
            field,
            world["boundary_conditions"]["x"],
            world["boundary_conditions"]["y"],
            world["boundary_conditions"]["z"],
        )
        reference_tiles = tile_vector_field((reference,), world, tile_shape)[0]

        self.assertTrue(jnp.allclose(refreshed, reference_tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_ghost_cells_two_guard_layers_matches_global_refresh(self):
        world = self._mixed_bc_world()
        tile_shape = (2, 3, 2)
        num_guard_cells = 2
        world = self._with_tile_metadata(world, tile_shape, g=num_guard_cells)
        Nx, Ny, Nz = world["Nx"], world["Ny"], world["Nz"]
        shape = (
            Nx + 2 * num_guard_cells,
            Ny + 2 * num_guard_cells,
            Nz + 2 * num_guard_cells,
        )

        field = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float64).reshape(shape)
        field = self._fill_guard_cells(field, world, num_guard_cells)
        tiles = self._tile_scalar_field_with_guard(field, tile_shape, num_guard_cells)

        stale_tiles = tiles.at[:, :, :, :num_guard_cells, :, :].set(-100.0)
        stale_tiles = stale_tiles.at[:, :, :, -num_guard_cells:, :, :].set(-200.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :num_guard_cells, :].set(-300.0)
        stale_tiles = stale_tiles.at[:, :, :, :, -num_guard_cells:, :].set(-400.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, :num_guard_cells].set(-500.0)
        stale_tiles = stale_tiles.at[:, :, :, :, :, -num_guard_cells:].set(-600.0)

        refreshed = ghost_cells.update_tiled_ghost_cells(stale_tiles, world, num_guard_cells)

        self.assertTrue(jnp.allclose(refreshed, tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_tiled_vector_ghost_cells_conducting_refreshes_each_component(self):
        world = self._conducting_world()
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape, g=2)
        E = self._deterministic_vector_field(world, scale=1.0)
        E_tiles = tile_vector_field(E, world, tile_shape)

        stale_tiles = tuple(component.at[:, :, :, 0, :, :].set(-10.0 * (i + 1)) for i, component in enumerate(E_tiles))
        refreshed = ghost_cells.update_tiled_vector_ghost_cells(stale_tiles, world)

        for original, refreshed_component in zip(E, refreshed):
            reference = _update_ghost_cells(
                original,
                world["boundary_conditions"]["x"],
                world["boundary_conditions"]["y"],
                world["boundary_conditions"]["z"],
            )
            reference_tiles = tile_vector_field((reference,), world, tile_shape)[0]
            self.assertTrue(jnp.allclose(refreshed_component, reference_tiles, rtol=1.0e-12, atol=1.0e-12))

    def test_update_E_matches_single_tile_yee_update(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)

        E_reference, _ = self._reference_update_E(E, B, J, world, constants)
        E_tiled, pml_state = update_E(
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        E_from_tiles = assemble_tiled_vector_field(E_tiled, world, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_update_E_is_the_public_tiled_update(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)
        E_tiles = tile_vector_field(E, world, tile_shape, num_guard_cells=2)
        B_tiles = tile_vector_field(B, world, tile_shape, num_guard_cells=2)
        J_tiles = tile_vector_field(J, world, tile_shape, num_guard_cells=2)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)

        E_public, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)

        self.assertIsNone(pml_state)
        self.assertEqual(E_public[0].ndim, 6)
        self.assertEqual(E_public[0].shape[:3], E_tiles[0].shape[:3])

    def test_update_E_matches_single_tile_yee_update_with_conducting_boundaries(self):
        world = self._conducting_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)

        E_reference, _ = self._reference_update_E(E, B, J, world, constants)
        E_tiled, pml_state = update_E(
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        E_from_tiles = assemble_tiled_vector_field(E_tiled, world, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_update_B_matches_single_tile_yee_update(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)

        B_reference, _ = self._reference_update_B(E, B, world, constants)
        B_tiled, pml_state = update_B(
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        B_from_tiles = assemble_tiled_vector_field(B_tiled, world, tile_shape)

        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_update_B_is_the_public_tiled_update(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        E_tiles = tile_vector_field(E, world, tile_shape, num_guard_cells=2)
        B_tiles = tile_vector_field(B, world, tile_shape, num_guard_cells=2)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)

        B_public, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters)

        self.assertIsNone(pml_state)
        self.assertEqual(B_public[0].ndim, 6)
        self.assertEqual(B_public[0].shape[:3], B_tiles[0].shape[:3])

    def test_update_B_matches_single_tile_yee_update_with_conducting_boundaries(self):
        world = self._conducting_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)

        B_reference, _ = self._reference_update_B(E, B, world, constants)
        B_tiled, pml_state = update_B(
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            static_parameters,
            dynamic_parameters,
        )
        self.assertIsNone(pml_state)
        B_from_tiles = assemble_tiled_vector_field(B_tiled, world, tile_shape)

        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_electrodynamic_step_matches_single_tile_yee_sequence(self):
        world = self._build_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)

        E_reference, B_reference, _ = self._reference_yee_step(E, B, J, world, constants)

        E_tiles = tile_vector_field(E, world, tile_shape)
        B_tiles = tile_vector_field(B, world, tile_shape)
        J_tiles = tile_vector_field(J, world, tile_shape)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)
        E_tiles, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)
        B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state)
        self.assertIsNone(pml_state)

        E_from_tiles = assemble_tiled_vector_field(E_tiles, world, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, world, tile_shape)

        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_electrodynamic_step_matches_single_tile_yee_sequence_with_conducting_boundaries(self):
        world = self._conducting_world()
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 3, 2)
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)

        E_reference, B_reference, _ = self._reference_yee_step(E, B, J, world, constants)

        E_tiles = tile_vector_field(E, world, tile_shape)
        B_tiles = tile_vector_field(B, world, tile_shape)
        J_tiles = tile_vector_field(J, world, tile_shape)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)
        E_tiles, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)
        B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state)
        self.assertIsNone(pml_state)

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
        world = self._with_tile_metadata(world, tile_shape)
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        J = self._deterministic_vector_field(world, scale=0.05)

        E_reference, B_reference, _ = self._reference_yee_step(E, B, J, world, constants)
        E_tiles = tile_vector_field(E, world, tile_shape)
        B_tiles = tile_vector_field(B, world, tile_shape)
        J_tiles = tile_vector_field(J, world, tile_shape)
        static_parameters, dynamic_parameters = self._split_parameters(world, constants)
        E_tiles, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)
        B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state)

        E_from_tiles = assemble_tiled_vector_field(E_tiles, world, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, world, tile_shape)

        self.assertIsNone(pml_state)
        for reference, tiled in zip(E_reference, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for original, after in zip(tile_vector_field(J, world, tile_shape), J_tiles):
            self.assertTrue(jnp.allclose(after, original, rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
