import unittest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_PERIODIC,
)
from PyPIC3D.boundary_conditions import ghost_cells
from tests.kernel_fixtures import kernel_parameters


def _mesh(mesh_shape):
    n_devices = int(jnp.prod(jnp.asarray(mesh_shape)))
    devices = jax.devices()
    if len(devices) < n_devices:
        raise RuntimeError(f"Need {n_devices} JAX devices, got {len(devices)}")
    return jax.make_mesh(
        mesh_shape,
        ghost_cells.MESH_AXES,
        devices=devices[:n_devices],
        axis_types=(jax.sharding.AxisType.Auto,) * len(ghost_cells.MESH_AXES),
    )


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
    n = tuple(int(width) * int(count) for width, count in zip(tile_shape, mesh_shape))
    return kernel_parameters(
        Nx=n[0], Ny=n[1], Nz=n[2],
        tile_shape=tuple(int(width) for width in tile_shape),
        guard_cells=int(g),
        boundary_conditions=tuple(int(bc) for bc in boundary_conditions),
        particle_boundary_conditions=tuple(int(bc) for bc in particle_boundary_conditions),
    )[0]


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


class DistributedGhostCellsFixtures:
    def assert_allclose(self, actual, expected):
        self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12), msg=f"\n{actual}\n!=\n{expected}")
