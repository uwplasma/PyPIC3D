import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.particles import particle_tile_communication as particle_comm
from PyPIC3D.particles.particle_class import TiledParticles


jax.config.update("jax_enable_x64", True)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _mesh(mesh_shape):
    n_devices = int(np.prod(mesh_shape))
    devices = jax.devices()
    if len(devices) < n_devices:
        raise unittest.SkipTest(f"Need {n_devices} JAX devices, got {len(devices)}")
    return Mesh(np.asarray(devices[:n_devices]).reshape(mesh_shape), ghost_cells.MESH_AXES)


def _static_parameters(mesh_shape, tile_shape, particle_bcs=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)):
    return SimpleNamespace(
        tile_shape=tuple(int(width) for width in tile_shape),
        guard_cells=2,
        particle_boundary_conditions=tuple(int(bc) for bc in particle_bcs),
        field_mesh=_mesh(mesh_shape),
    )


def _dynamic_parameters(mesh_shape, tile_shape):
    nx = int(mesh_shape[0]) * int(tile_shape[0])
    ny = int(mesh_shape[1]) * int(tile_shape[1])
    nz = int(mesh_shape[2]) * int(tile_shape[2])
    return SimpleNamespace(
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


class TestDistributedParticleRefresh(unittest.TestCase):
    def assert_allclose(self, actual, expected):
        self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12), msg=f"\n{actual}\n!=\n{expected}")

    def test_no_roll_based_tile_migration_remains(self):
        source = (REPO_ROOT / "PyPIC3D" / "particles" / "particle_tile_communication.py").read_text()

        self.assertNotIn("jnp.roll", source)

    def test_particle_sharding_places_one_logical_tile_on_each_device(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 1, 1)
        static_parameters = _static_parameters(mesh_shape, tile_shape)
        particles = _put_particle(_empty_particles(mesh_shape), (0, 0, 0), 0, (0.25, 0.0, 0.0))

        sharded = particle_comm.shard_tiled_particles(particles, static_parameters)

        self.assertEqual(sharded.x.sharding, NamedSharding(static_parameters.field_mesh, particle_comm.PARTICLE_STATE_TILE_SPEC))
        self.assertEqual(sharded.active.sharding, NamedSharding(static_parameters.field_mesh, particle_comm.PARTICLE_ACTIVE_TILE_SPEC))

    def test_refresh_moves_particle_to_x_neighbor_and_preserves_sharding(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 1, 1)
        static_parameters = _static_parameters(mesh_shape, tile_shape)
        dynamic_parameters = _dynamic_parameters(mesh_shape, tile_shape)
        particles = _put_particle(_empty_particles(mesh_shape), (0, 0, 0), 0, (0.25, 0.0, 0.0))
        particles = _shard_particles(particles, static_parameters)

        refreshed, overflow = jax.jit(
            lambda p: particle_comm.refresh_tiled_particle_tiles(p, static_parameters, dynamic_parameters)
        )(particles)

        self.assertFalse(bool(overflow))
        self.assertEqual(refreshed.x.sharding, particles.x.sharding)
        self.assertEqual(refreshed.active.sharding.mesh, static_parameters.field_mesh)
        self.assertEqual(len(refreshed.active.addressable_shards), int(np.prod(mesh_shape)))
        self.assertEqual(int(jnp.sum(refreshed.active[0, 0, 0, 0])), 0)
        self.assertEqual(int(jnp.sum(refreshed.active[1, 0, 0, 0])), 1)
        self.assert_allclose(refreshed.x[1, 0, 0, 0, 0, 0], 0.25)

    def test_refresh_wraps_periodic_edge_with_ppermute(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 1, 1)
        static_parameters = _static_parameters(mesh_shape, tile_shape)
        dynamic_parameters = _dynamic_parameters(mesh_shape, tile_shape)
        particles = _put_particle(_empty_particles(mesh_shape), (1, 0, 0), 0, (2.25, 0.0, 0.0))
        particles = _shard_particles(particles, static_parameters)

        refreshed, overflow = particle_comm.refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)

        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active[1, 0, 0, 0])), 0)
        self.assertEqual(int(jnp.sum(refreshed.active[0, 0, 0, 0])), 1)
        self.assert_allclose(refreshed.x[0, 0, 0, 0, 0, 0], -1.75)

    def test_refresh_moves_diagonal_particle_through_two_axis_permute(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 1)
        static_parameters = _static_parameters(mesh_shape, tile_shape)
        dynamic_parameters = _dynamic_parameters(mesh_shape, tile_shape)
        particles = _put_particle(_empty_particles(mesh_shape), (0, 0, 0), 0, (0.25, 0.25, 0.0))
        particles = _shard_particles(particles, static_parameters)

        refreshed, overflow = particle_comm.refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)

        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active[0, 0, 0, 0])), 0)
        self.assertEqual(int(jnp.sum(refreshed.active[1, 1, 0, 0])), 1)
        self.assert_allclose(refreshed.x[1, 1, 0, 0, 0, :2], jnp.asarray([0.25, 0.25]))

    def test_refresh_reports_destination_capacity_overflow(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 1, 1)
        static_parameters = _static_parameters(mesh_shape, tile_shape)
        dynamic_parameters = _dynamic_parameters(mesh_shape, tile_shape)
        particles = _empty_particles(mesh_shape, n_slots=1)
        particles = _put_particle(particles, (0, 0, 0), 0, (0.25, 0.0, 0.0))
        particles = _put_particle(particles, (1, 0, 0), 0, (1.25, 0.0, 0.0))
        particles = _shard_particles(particles, static_parameters)

        refreshed, overflow = particle_comm.refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)

        self.assertTrue(bool(overflow))
        self.assertEqual(refreshed.x.shape, particles.x.shape)
        self.assertEqual(int(jnp.sum(refreshed.active)), 1)

    def test_refresh_rejects_particle_topology_that_does_not_match_mesh(self):
        static_parameters = _static_parameters((1, 1, 1), (2, 1, 1))
        dynamic_parameters = _dynamic_parameters((1, 1, 1), (2, 1, 1))
        particles = _empty_particles((2, 1, 1), n_slots=1)

        with self.assertRaisesRegex(ValueError, "one logical particle tile per device"):
            particle_comm.refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)


if __name__ == "__main__":
    unittest.main()
