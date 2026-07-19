import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions import ghost_cells


jax.config.update("jax_enable_x64", True)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _mesh(mesh_shape):
    n_devices = int(np.prod(mesh_shape))
    devices = jax.devices()
    if len(devices) < n_devices:
        raise unittest.SkipTest(f"Need {n_devices} JAX devices, got {len(devices)}")
    return Mesh(np.asarray(devices[:n_devices]).reshape(mesh_shape), ghost_cells.MESH_AXES)


def _parameter_values(boundary_conditions, tile_shape):
    return {
        "tile_shape": tuple(int(width) for width in tile_shape),
        "guard_cells": 1,
        "boundary_conditions": {
            "x": boundary_conditions[0],
            "y": boundary_conditions[1],
            "z": boundary_conditions[2],
        },
    }


def _static_parameters(boundary_conditions, tile_shape, mesh_shape, g=1, particle_boundary_conditions=None):
    if particle_boundary_conditions is None:
        particle_boundary_conditions = boundary_conditions
    return SimpleNamespace(
        tile_shape=tuple(int(width) for width in tile_shape),
        guard_cells=int(g),
        boundary_conditions=tuple(int(bc) for bc in boundary_conditions),
        particle_boundary_conditions=tuple(int(bc) for bc in particle_boundary_conditions),
        field_mesh=_mesh(mesh_shape),
    )


def _coordinate_tiles(mesh_shape, tile_shape, g=1):
    ntx, nty, ntz = mesh_shape
    tile_nx, tile_ny, tile_nz = tile_shape
    tiles = jnp.zeros(
        (ntx, nty, ntz, tile_nx + 2 * g, tile_ny + 2 * g, tile_nz + 2 * g),
        dtype=jnp.float64,
    )
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(tile_nx, dtype=jnp.float64),
        jnp.arange(tile_ny, dtype=jnp.float64),
        jnp.arange(tile_nz, dtype=jnp.float64),
        indexing="ij",
    )
    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                value = 100.0 * tx + 10.0 * ty + tz + 0.01 * ii + 0.001 * jj + 0.0001 * kk
                tiles = tiles.at[tx, ty, tz, g:-g, g:-g, g:-g].set(value)
    return tiles


def _reference_update(field_tiles, boundary_conditions, tile_shape, g=1):
    bc_x, bc_y, bc_z = boundary_conditions
    mesh_shape = tuple(int(width) for width in field_tiles.shape[:3])
    reduced_x, reduced_y, reduced_z = (
        int(tile_shape[0]) == 1 and mesh_shape[0] == 1,
        int(tile_shape[1]) == 1 and mesh_shape[1] == 1,
        int(tile_shape[2]) == 1 and mesh_shape[2] == 1,
    )

    if reduced_x:
        lower = jnp.broadcast_to(field_tiles[:, :, :, g:g + 1, :, :], field_tiles[:, :, :, :g, :, :].shape)
        upper = jnp.broadcast_to(field_tiles[:, :, :, g:g + 1, :, :], field_tiles[:, :, :, -g:, :, :].shape)
        field_tiles = field_tiles.at[:, :, :, :g, :, :].set(lower)
        field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(upper)
        if bc_x != BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, :g, :, :].set(0.0)
            field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(0.0)
    else:
        if bc_x == BC_PERIODIC:
            lower = jnp.roll(field_tiles[:, :, :, -2 * g:-g, :, :], shift=1, axis=0)
            upper = jnp.roll(field_tiles[:, :, :, g:2 * g, :, :], shift=-1, axis=0)
        else:
            lower = jnp.zeros_like(field_tiles[:, :, :, :g, :, :])
            upper = jnp.zeros_like(field_tiles[:, :, :, -g:, :, :])
            lower = lower.at[1:, :, :, :, :, :].set(field_tiles[:-1, :, :, -2 * g:-g, :, :])
            upper = upper.at[:-1, :, :, :, :, :].set(field_tiles[1:, :, :, g:2 * g, :, :])
        field_tiles = field_tiles.at[:, :, :, :g, :, :].set(lower)
        field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(upper)

    if reduced_y:
        lower = jnp.broadcast_to(field_tiles[:, :, :, :, g:g + 1, :], field_tiles[:, :, :, :, :g, :].shape)
        upper = jnp.broadcast_to(field_tiles[:, :, :, :, g:g + 1, :], field_tiles[:, :, :, :, -g:, :].shape)
        field_tiles = field_tiles.at[:, :, :, :, :g, :].set(lower)
        field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(upper)
        if bc_y != BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, :, :g, :].set(0.0)
            field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(0.0)
    else:
        if bc_y == BC_PERIODIC:
            lower = jnp.roll(field_tiles[:, :, :, :, -2 * g:-g, :], shift=1, axis=1)
            upper = jnp.roll(field_tiles[:, :, :, :, g:2 * g, :], shift=-1, axis=1)
        else:
            lower = jnp.zeros_like(field_tiles[:, :, :, :, :g, :])
            upper = jnp.zeros_like(field_tiles[:, :, :, :, -g:, :])
            lower = lower.at[:, 1:, :, :, :, :].set(field_tiles[:, :-1, :, :, -2 * g:-g, :])
            upper = upper.at[:, :-1, :, :, :, :].set(field_tiles[:, 1:, :, :, g:2 * g, :])
        field_tiles = field_tiles.at[:, :, :, :, :g, :].set(lower)
        field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(upper)

    if reduced_z:
        lower = jnp.broadcast_to(field_tiles[:, :, :, :, :, g:g + 1], field_tiles[:, :, :, :, :, :g].shape)
        upper = jnp.broadcast_to(field_tiles[:, :, :, :, :, g:g + 1], field_tiles[:, :, :, :, :, -g:].shape)
        field_tiles = field_tiles.at[:, :, :, :, :, :g].set(lower)
        field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(upper)
        if bc_z != BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, :, :, :g].set(0.0)
            field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(0.0)
    else:
        if bc_z == BC_PERIODIC:
            lower = jnp.roll(field_tiles[:, :, :, :, :, -2 * g:-g], shift=1, axis=2)
            upper = jnp.roll(field_tiles[:, :, :, :, :, g:2 * g], shift=-1, axis=2)
        else:
            lower = jnp.zeros_like(field_tiles[:, :, :, :, :, :g])
            upper = jnp.zeros_like(field_tiles[:, :, :, :, :, -g:])
            lower = lower.at[:, :, 1:, :, :, :].set(field_tiles[:, :, :-1, :, :, -2 * g:-g])
            upper = upper.at[:, :, :-1, :, :, :].set(field_tiles[:, :, 1:, :, :, g:2 * g])
        field_tiles = field_tiles.at[:, :, :, :, :, :g].set(lower)
        field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(upper)

    return field_tiles


def _reference_fold(field_tiles, boundary_conditions, tile_shape, g=1):
    bc_x, bc_y, bc_z = boundary_conditions
    mesh_shape = tuple(int(width) for width in field_tiles.shape[:3])
    reduced_x, reduced_y, reduced_z = (
        int(tile_shape[0]) == 1 and mesh_shape[0] == 1,
        int(tile_shape[1]) == 1 and mesh_shape[1] == 1,
        int(tile_shape[2]) == 1 and mesh_shape[2] == 1,
    )

    if reduced_x:
        ghost_sum = jnp.sum(field_tiles[:, :, :, :g, :, :], axis=3, keepdims=True)
        ghost_sum = ghost_sum + jnp.sum(field_tiles[:, :, :, -g:, :, :], axis=3, keepdims=True)
        if bc_x == BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, g:g + 1, :, :].add(ghost_sum)
        elif bc_x == BC_CONDUCTING:
            field_tiles = field_tiles.at[:, :, :, g:g + 1, :, :].add(-ghost_sum)
    else:
        lower_ghost = field_tiles[:, :, :, :g, :, :]
        upper_ghost = field_tiles[:, :, :, -g:, :, :]
        if bc_x == BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, -2 * g:-g, :, :].add(jnp.roll(lower_ghost, shift=-1, axis=0))
            field_tiles = field_tiles.at[:, :, :, g:2 * g, :, :].add(jnp.roll(upper_ghost, shift=1, axis=0))
        else:
            field_tiles = field_tiles.at[:-1, :, :, -2 * g:-g, :, :].add(lower_ghost[1:, :, :, :, :, :])
            field_tiles = field_tiles.at[1:, :, :, g:2 * g, :, :].add(upper_ghost[:-1, :, :, :, :, :])
            if bc_x == BC_CONDUCTING:
                field_tiles = field_tiles.at[0, :, :, g:2 * g, :, :].add(-lower_ghost[0, :, :, :, :, :])
                field_tiles = field_tiles.at[-1, :, :, -2 * g:-g, :, :].add(-upper_ghost[-1, :, :, :, :, :])
    field_tiles = field_tiles.at[:, :, :, :g, :, :].set(0.0)
    field_tiles = field_tiles.at[:, :, :, -g:, :, :].set(0.0)

    if reduced_y:
        ghost_sum = jnp.sum(field_tiles[:, :, :, :, :g, :], axis=4, keepdims=True)
        ghost_sum = ghost_sum + jnp.sum(field_tiles[:, :, :, :, -g:, :], axis=4, keepdims=True)
        if bc_y == BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, :, g:g + 1, :].add(ghost_sum)
        elif bc_y == BC_CONDUCTING:
            field_tiles = field_tiles.at[:, :, :, :, g:g + 1, :].add(-ghost_sum)
    else:
        lower_ghost = field_tiles[:, :, :, :, :g, :]
        upper_ghost = field_tiles[:, :, :, :, -g:, :]
        if bc_y == BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, :, -2 * g:-g, :].add(jnp.roll(lower_ghost, shift=-1, axis=1))
            field_tiles = field_tiles.at[:, :, :, :, g:2 * g, :].add(jnp.roll(upper_ghost, shift=1, axis=1))
        else:
            field_tiles = field_tiles.at[:, :-1, :, :, -2 * g:-g, :].add(lower_ghost[:, 1:, :, :, :, :])
            field_tiles = field_tiles.at[:, 1:, :, :, g:2 * g, :].add(upper_ghost[:, :-1, :, :, :, :])
            if bc_y == BC_CONDUCTING:
                field_tiles = field_tiles.at[:, 0, :, :, g:2 * g, :].add(-lower_ghost[:, 0, :, :, :, :])
                field_tiles = field_tiles.at[:, -1, :, :, -2 * g:-g, :].add(-upper_ghost[:, -1, :, :, :, :])
    field_tiles = field_tiles.at[:, :, :, :, :g, :].set(0.0)
    field_tiles = field_tiles.at[:, :, :, :, -g:, :].set(0.0)

    if reduced_z:
        ghost_sum = jnp.sum(field_tiles[:, :, :, :, :, :g], axis=5, keepdims=True)
        ghost_sum = ghost_sum + jnp.sum(field_tiles[:, :, :, :, :, -g:], axis=5, keepdims=True)
        if bc_z == BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, :, :, g:g + 1].add(ghost_sum)
        elif bc_z == BC_CONDUCTING:
            field_tiles = field_tiles.at[:, :, :, :, :, g:g + 1].add(-ghost_sum)
    else:
        lower_ghost = field_tiles[:, :, :, :, :, :g]
        upper_ghost = field_tiles[:, :, :, :, :, -g:]
        if bc_z == BC_PERIODIC:
            field_tiles = field_tiles.at[:, :, :, :, :, -2 * g:-g].add(jnp.roll(lower_ghost, shift=-1, axis=2))
            field_tiles = field_tiles.at[:, :, :, :, :, g:2 * g].add(jnp.roll(upper_ghost, shift=1, axis=2))
        else:
            field_tiles = field_tiles.at[:, :, :-1, :, :, -2 * g:-g].add(lower_ghost[:, :, 1:, :, :, :])
            field_tiles = field_tiles.at[:, :, 1:, :, :, g:2 * g].add(upper_ghost[:, :, :-1, :, :, :])
            if bc_z == BC_CONDUCTING:
                field_tiles = field_tiles.at[:, :, 0, :, :, g:2 * g].add(-lower_ghost[:, :, 0, :, :, :])
                field_tiles = field_tiles.at[:, :, -1, :, :, -2 * g:-g].add(-upper_ghost[:, :, -1, :, :, :])
    field_tiles = field_tiles.at[:, :, :, :, :, :g].set(0.0)
    field_tiles = field_tiles.at[:, :, :, :, :, -g:].set(0.0)

    return field_tiles


class TestDistributedGhostCells(unittest.TestCase):
    def assert_allclose(self, actual, expected):
        self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12), msg=f"\n{actual}\n!=\n{expected}")

    def test_no_single_device_fast_paths_remain_in_tiled_ghost_module(self):
        source = (REPO_ROOT / "PyPIC3D" / "boundary_conditions" / "ghost_cells.py").read_text()

        self.assertNotIn("_local_refresh_single_tile_axis", source)
        self.assertNotIn("_local_fold_single_tile_axis", source)
        self.assertNotIn("mesh_shape == (1, 1, 1)", source)
        self.assertNotIn("elif mesh_shape[0] == 1", source)
        self.assertNotIn("elif mesh_shape[1] == 1", source)
        self.assertNotIn("elif mesh_shape[2] == 1", source)

    def test_public_scalar_update_on_one_device_uses_mapped_periodic_self_exchange(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)

        sharded_tiles = jax.device_put(tiles, sharding)
        actual = jax.jit(lambda field: ghost_cells.update_tiled_ghost_cells(field, static_parameters, 1))(sharded_tiles)

        self.assertEqual(actual.sharding, sharding)
        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 0, 1:-1, 1:-1], tiles[0, 0, 0, -2, 1:-1, 1:-1])
        self.assert_allclose(actual[0, 0, 0, -1, 1:-1, 1:-1], tiles[0, 0, 0, 1, 1:-1, 1:-1])

    def test_public_vector_update_on_one_device_preserves_stacked_and_tuple_layouts(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        stacked = jnp.stack((tiles, tiles + 1000.0, tiles + 2000.0), axis=0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.VECTOR_TILE_SPEC)

        sharded_stacked = jax.device_put(stacked, sharding)
        stacked_actual = jax.jit(
            lambda field: ghost_cells.update_tiled_vector_ghost_cells(field, static_parameters, 1)
        )(sharded_stacked)
        tuple_actual = ghost_cells.update_tiled_vector_ghost_cells((stacked[0], stacked[1], stacked[2]), static_parameters, 1)

        self.assertEqual(stacked_actual.sharding, sharding)
        self.assertEqual(stacked_actual.shape, stacked.shape)
        self.assertIsInstance(tuple_actual, tuple)
        for i in range(3):
            self.assert_allclose(stacked_actual[i], _reference_update(stacked[i], bcs, tile_shape))
            self.assert_allclose(tuple_actual[i], stacked_actual[i])

    def test_one_device_periodic_fold_uses_self_exchange_and_clears_ghosts(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        tiles = tiles.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)

        sharded_tiles = jax.device_put(tiles, sharding)
        actual = jax.jit(lambda field: ghost_cells.fold_tiled_ghost_cells(field, static_parameters, 1))(sharded_tiles)

        self.assertEqual(actual.sharding, sharding)
        self.assert_allclose(actual, _reference_fold(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, -2, 1:-1, 1:-1], 3.0)
        self.assert_allclose(actual[0, 0, 0, 1, 1:-1, 1:-1], 5.0)
        self.assert_allclose(actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, -1, :, :], 0.0)

    def test_one_device_conducting_fold_keeps_physical_reflection_sign(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        tiles = tiles.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)

        actual = jax.jit(lambda field: ghost_cells.fold_tiled_ghost_cells(field, static_parameters, 1))(tiles)

        self.assert_allclose(actual, _reference_fold(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 1, 1:-1, 1:-1], -3.0)
        self.assert_allclose(actual[0, 0, 0, -2, 1:-1, 1:-1], -5.0)
        self.assert_allclose(actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, -1, :, :], 0.0)

    def test_reduced_axis_on_one_device_does_not_use_periodic_self_exchange(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (1, 3, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        tiles = tiles.at[0, 0, 0, 0, :, :].set(-100.0)
        tiles = tiles.at[0, 0, 0, -1, :, :].set(100.0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)

        actual = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 0, :, :], actual[0, 0, 0, 1, :, :])
        self.assert_allclose(actual[0, 0, 0, -1, :, :], actual[0, 0, 0, 1, :, :])

    def test_public_axis_zero_boundary_on_one_device_runs_inside_mapped_path(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)
        field = jnp.ones(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)

        sharded_field = jax.device_put(field, sharding)
        actual = jax.jit(lambda tiles: ghost_cells.apply_tiled_zero_boundary(tiles, static_parameters, 0, 1))(
            sharded_field
        )

        self.assertEqual(actual.sharding, sharding)
        self.assert_allclose(actual[0, 0, 0, 1, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, -2, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, 2, :, :], 1.0)

    def test_scalar_halo_refresh_matches_reference_on_2x2x2_periodic_mesh(self):
        mesh_shape = (2, 2, 2)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 0, 0, 0], tiles[1, 1, 1, -2, -2, -2])

    def test_scalar_halo_refresh_matches_reference_on_mixed_2x2x1_mesh(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_CONDUCTING, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, :, 0, :], jnp.zeros_like(actual[0, 0, 0, :, 0, :]))

    def test_one_device_uses_same_mapped_periodic_and_conducting_paths(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        for bcs in (
            (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            (BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
        ):
            updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
            actual = jax.jit(updater)(tiles)
            self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))

    def test_reduced_physical_axis_uses_single_interior_cell_not_neighbor_exchange(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 1)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[:, :, :, :, :, 0], actual[:, :, :, :, :, 1])
        self.assert_allclose(actual[:, :, :, :, :, -1], actual[:, :, :, :, :, 1])

    def test_public_update_bc_type_selects_field_or_particle_boundaries(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (2, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            particle_boundary_conditions=particle_bcs,
        )

        field_bc_actual = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1, bc_type=0)
        particle_bc_actual = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1, bc_type=1)

        self.assert_allclose(field_bc_actual, _reference_update(tiles, field_bcs, tile_shape))
        self.assert_allclose(particle_bc_actual, _reference_update(tiles, particle_bcs, tile_shape))
        self.assert_allclose(field_bc_actual[0, 0, 0, 0, 1:-1, 1:-1], tiles[0, 0, 0, -2, 1:-1, 1:-1])
        self.assert_allclose(particle_bc_actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, -1, :, :], 0.0)

    def test_public_fold_bc_type_selects_absorbing_particle_boundaries(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (2, BC_PERIODIC, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        tiles = tiles.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            particle_boundary_conditions=particle_bcs,
        )

        field_bc_actual = ghost_cells.fold_tiled_ghost_cells(tiles, static_parameters, 1, bc_type=0)
        particle_bc_actual = ghost_cells.fold_tiled_ghost_cells(tiles, static_parameters, 1, bc_type=1)

        self.assert_allclose(field_bc_actual, _reference_fold(tiles, field_bcs, tile_shape))
        self.assert_allclose(particle_bc_actual, _reference_fold(tiles, particle_bcs, tile_shape))
        self.assert_allclose(field_bc_actual[0, 0, 0, -2, 1:-1, 1:-1], 3.0)
        self.assert_allclose(field_bc_actual[0, 0, 0, 1, 1:-1, 1:-1], 5.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, 1:-1, 1:-1, 1:-1], 0.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, -1, :, :], 0.0)

    def test_reduced_absorbing_particle_axis_discards_ghosts(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (1, 2, 2)
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (2, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            particle_boundary_conditions=particle_bcs,
        )

        refreshed = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1, bc_type=1)
        self.assert_allclose(refreshed, _reference_update(tiles, particle_bcs, tile_shape))
        self.assert_allclose(refreshed[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(refreshed[0, 0, 0, -1, :, :], 0.0)

        deposits = jnp.zeros(mesh_shape + (3, 4, 4), dtype=jnp.float64)
        deposits = deposits.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        deposits = deposits.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        folded = ghost_cells.fold_tiled_ghost_cells(deposits, static_parameters, 1, bc_type=1)

        self.assert_allclose(folded, _reference_fold(deposits, particle_bcs, tile_shape))
        self.assert_allclose(folded[0, 0, 0, 1, :, :], 0.0)
        self.assert_allclose(folded[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(folded[0, 0, 0, -1, :, :], 0.0)

    def test_stacked_and_tuple_vector_halo_refresh_preserve_layouts(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_CONDUCTING, BC_CONDUCTING)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        stacked = jnp.stack((tiles, tiles + 1000.0, tiles + 2000.0), axis=0)
        updater = ghost_cells.make_distributed_vector_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)

        stacked_actual = jax.jit(updater)(stacked)
        tuple_actual = updater((stacked[0], stacked[1], stacked[2]))

        self.assertEqual(stacked_actual.shape, stacked.shape)
        self.assertIsInstance(tuple_actual, tuple)
        for i in range(3):
            self.assert_allclose(stacked_actual[i], _reference_update(stacked[i], bcs, tile_shape))
            self.assert_allclose(tuple_actual[i], stacked_actual[i])

    def test_axis_zero_boundary_builds_conducting_electric_bc_on_global_walls(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_CONDUCTING, BC_CONDUCTING, BC_PERIODIC)
        E = tuple(jnp.ones((mesh_shape + (4, 4, 4)), dtype=jnp.float64) * (i + 1.0) for i in range(3))
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)

        def apply_electric_bc(Ex, Ey, Ez):
            Ey = ghost_cells.apply_tiled_zero_boundary(Ey, static_parameters, axis=0, num_guard_cells=1)
            Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=0, num_guard_cells=1)
            Ex = ghost_cells.apply_tiled_zero_boundary(Ex, static_parameters, axis=1, num_guard_cells=1)
            Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=1, num_guard_cells=1)
            return Ex, Ey, Ez

        Ex, Ey, Ez = jax.jit(apply_electric_bc)(*E)

        self.assert_allclose(Ey[0, :, :, 1, :, :], 0.0)
        self.assert_allclose(Ez[-1, :, :, -2, :, :], 0.0)
        self.assert_allclose(Ex[:, 0, :, :, 1, :], 0.0)
        self.assert_allclose(Ez[:, -1, :, :, -2, :], 0.0)
        self.assert_allclose(Ey[0, :, :, 2, 1:-1, 1:-1], 2.0)
        self.assert_allclose(Ex[:, 0, :, 1:-1, 2, 1:-1], 1.0)

    def test_scalar_and_vector_folding_match_reference_and_clear_ghosts(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_CONDUCTING, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (4, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[:, :, :, 0, :, :].set(1.0)
        tiles = tiles.at[:, :, :, -1, :, :].set(2.0)
        tiles = tiles.at[:, :, :, :, 0, :].set(3.0)
        tiles = tiles.at[:, :, :, :, -1, :].set(4.0)
        tiles = tiles.at[:, :, :, :, :, 0].set(5.0)
        tiles = tiles.at[:, :, :, :, :, -1].set(6.0)
        folder = ghost_cells.make_distributed_ghost_folder(_mesh(mesh_shape), tile_shape, bcs, 1)
        vector_folder = ghost_cells.make_distributed_vector_ghost_folder(_mesh(mesh_shape), tile_shape, bcs, 1)

        actual = jax.jit(folder)(tiles)
        vector_actual = jax.jit(vector_folder)((tiles, tiles + 1.0, tiles + 2.0))

        self.assert_allclose(actual, _reference_fold(tiles, bcs, tile_shape))
        for component, base in zip(vector_actual, (tiles, tiles + 1.0, tiles + 2.0)):
            self.assert_allclose(component, _reference_fold(base, bcs, tile_shape))
            self.assert_allclose(component[:, :, :, 0, :, :], 0.0)
            self.assert_allclose(component[:, :, :, -1, :, :], 0.0)

    def test_standard_refresh_uses_initialization_owned_nonwrapping_axis(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 2, 2)
        static_parameters = _static_parameters((BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC), tile_shape, mesh_shape)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        actual = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1)
        expected = _reference_update(tiles, (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC), tile_shape)

        self.assert_allclose(actual, expected)

    def test_validation_rejects_multiple_logical_tiles_per_device(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        tiles = _coordinate_tiles((2, 1, 1), tile_shape)

        with self.assertRaisesRegex(ValueError, "one logical tile per device"):
            updater(tiles)


if __name__ == "__main__":
    unittest.main()
