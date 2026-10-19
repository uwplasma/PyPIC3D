"""Run with JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 (or four GPUs)."""
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
from PyPIC3D.relativity.pair_injection import insert_pairs
from tests.code_tests.pair_injection_test import setup


@unittest.skipUnless(jax.device_count() >= 4, 'requires four devices')
class TestPairInjectionTiles(unittest.TestCase):
    def test_y_and_z_tiles(self):
        s, d, p, species, f, x, u = setup(cells=(4, 8, 8))
        p = shard_tiled_particles(p, s)
        layout = NamedSharding(s.field_mesh, PartitionSpec('tile_x', 'tile_y', 'tile_z'))
        x, u = jax.device_put((x, u), (layout, layout))
        result, counts = jax.jit(lambda pts, x, u: insert_pairs(
            pts, species, x, u, jnp.ones((1, 2, 2)), f[0], f[1], f[6], s, d,
            species_indices=(0, 2)))(p, x, u)
        self.assertTrue(bool(jnp.array_equal(counts[1], jnp.ones((1, 2, 2)))))
        self.assertTrue(bool(jnp.array_equal(result.x[..., 0, 0, :], x[..., 0, :])))
        rho = compute_rho(result, species, f[3], s, d)
        self.assertTrue(bool(jnp.allclose(rho, 0, atol=1e-13, rtol=1e-7, equal_nan=True)))
        for value in result:
            self.assertTrue(value.sharding.is_equivalent_to(layout, value.ndim))


if __name__ == '__main__':
    unittest.main()
