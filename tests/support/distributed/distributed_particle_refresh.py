import unittest
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.particles import particle_tile_communication as particle_comm
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.grids import build_yee_grid


class _HashableStaticParameters(NamedTuple):
    tile_shape: tuple
    guard_cells: int
    particle_boundary_conditions: tuple
    field_mesh: object


def _mesh(mesh_shape):
    n_devices = int(np.prod(mesh_shape))
    devices = jax.devices()
    if len(devices) < n_devices:
        raise RuntimeError(f"Need {n_devices} JAX devices, got {len(devices)}")
    return Mesh(np.asarray(devices[:n_devices]).reshape(mesh_shape), ghost_cells.MESH_AXES)


def _static_parameters(mesh_shape, tile_shape, particle_bcs=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)):
    return SimpleNamespace(
        tile_shape=tuple(int(width) for width in tile_shape),
        guard_cells=2,
        particle_boundary_conditions=tuple(int(bc) for bc in particle_bcs),
        field_mesh=_mesh(mesh_shape),
    )


def _hashable_static_parameters(mesh_shape, tile_shape, particle_bcs=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)):
    return _HashableStaticParameters(
        tile_shape=tuple(int(width) for width in tile_shape),
        guard_cells=2,
        particle_boundary_conditions=tuple(int(bc) for bc in particle_bcs),
        field_mesh=_mesh(mesh_shape),
    )


def _dynamic_parameters(mesh_shape, tile_shape):
    nx = int(mesh_shape[0]) * int(tile_shape[0])
    ny = int(mesh_shape[1]) * int(tile_shape[1])
    nz = int(mesh_shape[2]) * int(tile_shape[2])
    dynamic_parameters = SimpleNamespace(
        dx=jnp.asarray(1.0),
        dy=jnp.asarray(1.0),
        dz=jnp.asarray(1.0),
        Nx=jnp.asarray(nx),
        Ny=jnp.asarray(ny),
        Nz=jnp.asarray(nz),
        x_wind=jnp.asarray(float(nx)),
        y_wind=jnp.asarray(float(ny)),
        z_wind=jnp.asarray(float(nz)),
    )
    grid_parameters = SimpleNamespace(
        **vars(dynamic_parameters),
        x_min=jnp.asarray(-0.5 * nx),
        y_min=jnp.asarray(-0.5 * ny),
        z_min=jnp.asarray(-0.5 * nz),
    )
    center_grid, vertex_grid = build_yee_grid(grid_parameters)
    dynamic_parameters.grids = SimpleNamespace(
        center=center_grid,
        vertex=vertex_grid,
    )
    return dynamic_parameters


def _empty_particles(mesh_shape, n_slots=2):
    return TiledParticles(
        x=jnp.zeros(mesh_shape + (1, n_slots, 3), dtype=jnp.float64),
        u=jnp.zeros(mesh_shape + (1, n_slots, 3), dtype=jnp.float64),
        active=jnp.zeros(mesh_shape + (1, n_slots), dtype=bool),
    )


def _put_particle(particles, tile, slot, x, u=(0.0, 0.0, 0.0)):
    tx, ty, tz = tile
    particles = particles._replace(
        x=particles.x.at[tx, ty, tz, 0, slot].set(jnp.asarray(x, dtype=jnp.float64)),
        u=particles.u.at[tx, ty, tz, 0, slot].set(jnp.asarray(u, dtype=jnp.float64)),
        active=particles.active.at[tx, ty, tz, 0, slot].set(True),
    )
    return particles


def _shard_particles(particles, static_parameters):
    x_sharding = NamedSharding(static_parameters.field_mesh, particle_comm.PARTICLE_STATE_TILE_SPEC)
    active_sharding = NamedSharding(static_parameters.field_mesh, particle_comm.PARTICLE_ACTIVE_TILE_SPEC)
    return TiledParticles(
        x=jax.device_put(particles.x, x_sharding),
        u=jax.device_put(particles.u, x_sharding),
        active=jax.device_put(particles.active, active_sharding),
    )


class DistributedParticleRefreshFixtures:
    @classmethod
    def setUpClass(cls):
        particle_comm._cached_distributed_particle_refresher.cache_clear()

    @classmethod
    def tearDownClass(cls):
        particle_comm._cached_distributed_particle_refresher.cache_clear()

    def assert_allclose(self, actual, expected):
        self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12), msg=f"\n{actual}\n!=\n{expected}")
