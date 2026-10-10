import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.pusher.particle_push import particle_push
from tests.kernel_fixtures import (
    build_tiled_particles,
    empty_tiled_scalar,
    empty_tiled_vector,
    kernel_parameters,
    particle_species,
)


class TestParticleBatching(unittest.TestCase):
    def assert_tree_allclose(self, actual, expected):
        actual_leaves = jax.tree_util.tree_leaves(actual)
        expected_leaves = jax.tree_util.tree_leaves(expected)
        self.assertEqual(len(actual_leaves), len(expected_leaves))
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
            self.assertTrue(
                jnp.allclose(actual_leaf, expected_leaf, rtol=1.0e-12, atol=1.0e-12),
                msg=f"\n{actual_leaf}\n!=\n{expected_leaf}",
            )

    def test_small_batches_match_one_full_active_batch_for_sparse_particles(self):
        batched, dynamic = kernel_parameters(
            Nx=4,
            Ny=1,
            Nz=1,
            tile_shape=(4, 1, 1),
            dt=0.01,
            x_wind=1.0,
            y_wind=1.0,
            z_wind=1.0,
            relativistic=False,
            particle_batch_size=2,
        )
        full_batch = batched._replace(particle_batch_size=4)
        positions = np.asarray([-0.4, -0.1, 0.15, 0.4])
        species = [
            particle_species(
                "negative",
                -1.0,
                1.0,
                weight=0.75,
                x1=positions,
                u1=np.asarray([0.01, 0.02, 0.03, 0.04]),
                active_mask=np.asarray([True, False, True, False]),
            ),
            particle_species(
                "positive",
                2.0,
                3.0,
                weight=0.25,
                x1=positions[::-1],
                u1=np.asarray([-0.02, -0.04, -0.06, -0.08]),
                active_mask=np.asarray([False, True, False, True]),
            ),
        ]
        particles, species_config = build_tiled_particles(
            species,
            batched,
            dynamic,
            capacity_factor=1.0,
        )
        E = tuple(
            jnp.ones_like(component) * (axis + 1)
            for axis, component in enumerate(empty_tiled_vector(batched, dynamic))
        )
        B = empty_tiled_vector(batched, dynamic)
        J = empty_tiled_vector(batched, dynamic)
        rho = empty_tiled_scalar(batched, dynamic)

        pushed_batched = particle_push(particles, species_config, E, B, batched, dynamic)
        pushed_full = particle_push(particles, species_config, E, B, full_batch, dynamic)
        current_batched = J_from_rhov(particles, species_config, J, batched, dynamic)
        current_full = J_from_rhov(particles, species_config, J, full_batch, dynamic)
        rho_batched = compute_rho(particles, species_config, rho, batched, dynamic)
        rho_full = compute_rho(particles, species_config, rho, full_batch, dynamic)

        self.assert_tree_allclose(pushed_batched, pushed_full)
        self.assert_tree_allclose(current_batched, current_full)
        self.assert_tree_allclose(rho_batched, rho_full)


if __name__ == "__main__":
    unittest.main()
