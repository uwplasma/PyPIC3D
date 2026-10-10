"""Single-device numerical tests."""

from tests.support.distributed_particle_refresh_fixtures import (
    DistributedParticleRefreshFixtures,
    _dynamic_parameters,
    _empty_particles,
    _hashable_static_parameters,
    _put_particle,
    _shard_particles,
    _static_parameters,
    jnp,
    particle_comm,
    unittest,
)


class TestDistributedParticleRefresh(DistributedParticleRefreshFixtures, unittest.TestCase):
    def test_hashable_static_parameters_reuse_identical_refresher(self):
        static_parameters = _hashable_static_parameters((1, 1, 1), (2, 1, 1))

        first = particle_comm.make_distributed_particle_refresher(static_parameters)
        second = particle_comm.make_distributed_particle_refresher(static_parameters)

        self.assertIs(first, second)


    def test_unhashable_static_parameters_use_working_uncached_fallback(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (2, 1, 1)
        static_parameters = _static_parameters(mesh_shape, tile_shape)
        dynamic_parameters = _dynamic_parameters(mesh_shape, tile_shape)
        particles = _put_particle(_empty_particles(mesh_shape), (0, 0, 0), 0, (0.25, 0.0, 0.0))
        particles = _shard_particles(particles, static_parameters)

        first = particle_comm.make_distributed_particle_refresher(static_parameters)
        second = particle_comm.make_distributed_particle_refresher(static_parameters)
        refreshed, overflow = first(particles, dynamic_parameters)

        self.assertIsNot(first, second)
        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active)), 1)


    def test_refresh_rejects_particle_topology_that_does_not_match_mesh(self):
        static_parameters = _static_parameters((1, 1, 1), (2, 1, 1))
        dynamic_parameters = _dynamic_parameters((1, 1, 1), (2, 1, 1))
        particles = _empty_particles((2, 1, 1), n_slots=1)

        with self.assertRaisesRegex(ValueError, "one logical particle tile per device"):
            particle_comm.refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)



if __name__ == "__main__":
    unittest.main()
