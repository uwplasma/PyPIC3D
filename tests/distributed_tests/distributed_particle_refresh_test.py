"""Focused cross-device regression tests."""

from tests.support.distributed_particle_refresh_fixtures import (
    DistributedParticleRefreshFixtures,
    NamedSharding,
    _dynamic_parameters,
    _empty_particles,
    _hashable_static_parameters,
    _put_particle,
    _shard_particles,
    _static_parameters,
    jax,
    jnp,
    np,
    particle_comm,
    unittest,
)


class TestDistributedParticleRefresh(DistributedParticleRefreshFixtures, unittest.TestCase):
    def test_cached_refresher_uses_current_particle_values(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 1, 1)
        static_parameters = _hashable_static_parameters(mesh_shape, tile_shape)
        dynamic_parameters = _dynamic_parameters(mesh_shape, tile_shape)
        first_particles = _put_particle(
            _empty_particles(mesh_shape),
            (0, 0, 0),
            0,
            (0.25, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
        )
        second_particles = _put_particle(
            _empty_particles(mesh_shape),
            (0, 0, 0),
            0,
            (0.75, 0.0, 0.0),
            u=(2.0, 0.0, 0.0),
        )
        first_particles = _shard_particles(first_particles, static_parameters)
        second_particles = _shard_particles(second_particles, static_parameters)
        refresher = particle_comm.make_distributed_particle_refresher(static_parameters)

        first, first_overflow = refresher(first_particles, dynamic_parameters)
        second, second_overflow = refresher(second_particles, dynamic_parameters)

        self.assertFalse(bool(first_overflow))
        self.assertFalse(bool(second_overflow))
        self.assert_allclose(first.x[1, 0, 0, 0, 0, 0], 0.25)
        self.assert_allclose(second.x[1, 0, 0, 0, 0, 0], 0.75)
        self.assert_allclose(first.u[1, 0, 0, 0, 0, 0], 1.0)
        self.assert_allclose(second.u[1, 0, 0, 0, 0, 0], 2.0)


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
        self.assertTrue(refreshed.x.sharding.is_equivalent_to(particles.x.sharding, refreshed.x.ndim))
        self.assertTrue(refreshed.u.sharding.is_equivalent_to(particles.u.sharding, refreshed.u.ndim))
        self.assertTrue(refreshed.active.sharding.is_equivalent_to(particles.active.sharding, refreshed.active.ndim))
        self.assertEqual(len(refreshed.active.addressable_shards), int(np.prod(mesh_shape)))
        self.assertEqual(int(jnp.sum(refreshed.active)), int(jnp.sum(particles.active)))
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


if __name__ == "__main__":
    unittest.main()
