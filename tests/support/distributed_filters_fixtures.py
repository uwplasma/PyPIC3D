import unittest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from PyPIC3D.boundary_conditions import ghost_cells
from tests.kernel_fixtures import kernel_parameters
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.utilities.filters import (
    bilinear_filter,
    digital_filter,
    tiled_bilinear_filter,
    tiled_bilinear_filter_vector,
    tiled_digital_filter,
    tiled_digital_filter_vector,
)


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


def _static_parameters(mesh_shape, tile_shape, g=2, particle_boundary_conditions=(BC_PERIODIC,) * 3):
    n = tuple(int(width) * int(count) for width, count in zip(tile_shape, mesh_shape))
    return kernel_parameters(
        Nx=n[0], Ny=n[1], Nz=n[2],
        tile_shape=tuple(int(width) for width in tile_shape),
        guard_cells=int(g),
        particle_boundary_conditions=particle_boundary_conditions,
    )[0]


def _tile_interior(interior, mesh_shape, tile_shape, g):
    ntx, nty, ntz = mesh_shape
    tile_nx, tile_ny, tile_nz = tile_shape

    interior_tiles = interior.reshape(
        ntx,
        tile_nx,
        nty,
        tile_ny,
        ntz,
        tile_nz,
    ).transpose(0, 2, 4, 1, 3, 5)

    tiles = jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=interior.dtype,
    )
    return tiles.at[:, :, :, g:-g, g:-g, g:-g].set(interior_tiles)


def _assemble_interior(field_tiles, g):
    ntx, nty, ntz, local_nx, local_ny, local_nz = field_tiles.shape
    tile_nx = local_nx - 2 * g
    tile_ny = local_ny - 2 * g
    tile_nz = local_nz - 2 * g

    interior_tiles = field_tiles[:, :, :, g:-g, g:-g, g:-g]
    return interior_tiles.transpose(0, 3, 1, 4, 2, 5).reshape(
        ntx * tile_nx,
        nty * tile_ny,
        ntz * tile_nz,
    )


def _periodic_field(interior, g):
    return jnp.pad(interior, ((g, g), (g, g), (g, g)), mode="wrap")


class DistributedFiltersFixtures:
    def assert_allclose(self, actual, expected):
        self.assertTrue(
            jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12),
            msg=f"\n{actual}\n!=\n{expected}",
        )

    def _scalar_filter_case(self, tiled_filter, local_filter, alpha=None):
        mesh_shape = (2, 2, 2)
        tile_shape = (3, 3, 3)
        g = 2
        static_parameters = _static_parameters(mesh_shape, tile_shape, g)

        global_shape = tuple(mesh_size * tile_size for mesh_size, tile_size in zip(mesh_shape, tile_shape))
        values = jnp.arange(jnp.prod(jnp.asarray(global_shape)), dtype=jnp.float64).reshape(global_shape)
        interior = jnp.sin(values / 11.0) + 0.01 * values

        tiles = _tile_interior(interior, mesh_shape, tile_shape, g)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)
        sharded_tiles = jax.device_put(tiles, sharding)

        if alpha is None:
            actual = tiled_filter(sharded_tiles, static_parameters)
            expected = local_filter(_periodic_field(interior, g), num_guard_cells=g)
        else:
            actual = tiled_filter(sharded_tiles, alpha, static_parameters)
            expected = local_filter(_periodic_field(interior, g), alpha, num_guard_cells=g)

        actual.block_until_ready()
        self.assertEqual(actual.sharding, sharding)
        self.assertEqual(
            {tuple(shard.data.shape) for shard in actual.addressable_shards},
            {(1, 1, 1, 7, 7, 7)},
        )
        self.assert_allclose(_assemble_interior(actual, g), expected[g:-g, g:-g, g:-g])

        # The post-filter refresh must populate the tile interface from the
        # filtered owner interior rather than leaving the input guards stale.
        self.assert_allclose(
            actual[0, 0, 0, -g, g:-g, g:-g],
            actual[1, 0, 0, g, g:-g, g:-g],
        )
