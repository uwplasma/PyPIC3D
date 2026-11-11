"""Initial particle histograms keep the simulation's local device shards."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from PyPIC3D.diagnostics import plotting
from PyPIC3D.particles.particle_class import TiledParticles


class TestShardedInitialHistograms(unittest.TestCase):
    def setUp(self):
        if jax.local_device_count() < 2:
            self.skipTest("Run with JAX_NUM_CPU_DEVICES=2")
        mesh = Mesh(jax.local_devices()[:2], ("tile",))
        sharding = NamedSharding(mesh, P("tile"))
        x = jnp.full((2, 1, 1, 3, 4, 3), jnp.nan, dtype=jnp.float64)
        u = jnp.full_like(x, jnp.nan)
        active = jnp.zeros(x.shape[:-1], dtype=bool)
        # The range spans both shards, including values exactly on bin edges.
        positions = jnp.asarray([[-2.0, 3.0, -0.5], [-1.0, 4.0, 0.0],
                                 [0.0, 5.0, 0.5], [2.0, 6.0, 1.5]])
        velocities = 0.5 * positions + 1.0
        x = x.at[0, 0, 0, 0, :2].set(positions[:2])
        x = x.at[1, 0, 0, 0, :2].set(positions[2:])
        u = u.at[0, 0, 0, 0, :2].set(velocities[:2])
        u = u.at[1, 0, 0, 0, :2].set(velocities[2:])
        active = active.at[:, 0, 0, 0, :2].set(True)
        # A constant species and an empty species exercise automatic ranges.
        x = x.at[:, 0, 0, 1, :1].set(jnp.asarray([0.0, -4.0, 7.0]))
        u = u.at[:, 0, 0, 1, :1].set(jnp.asarray([-1.0, 0.0, 3.0]))
        active = active.at[:, 0, 0, 1, :1].set(True)
        self.particles = TiledParticles(*(jax.device_put(value, sharding) for value in (x, u, active)))
        self.names = ("live ions", "constant", "empty")
        grid = jnp.linspace(-3.0, 3.0, 7)
        self.dynamic = SimpleNamespace(grids=SimpleNamespace(center=(grid, grid, grid)))

    def test_shard_reductions_match_matplotlib_histograms_and_plot_metadata(self):
        expected_histograms = []
        actual_histograms = []
        original_hist = plotting.plt.hist

        def capture(histograms):
            def histogram(*args, **kwargs):
                count, edges, bars = original_hist(*args, **kwargs)
                histograms.append((jnp.asarray(count), jnp.asarray(edges)))
                return count, edges, bars
            return histogram

        with patch.object(plotting.plt, "savefig") as expected_save, \
             patch.object(plotting.plt, "xlabel") as expected_label, \
             patch.object(plotting.plt, "ylabel"), \
             patch.object(plotting.plt, "xlim") as expected_limits, \
             patch.object(plotting.plt, "title") as expected_titles, \
             patch.object(plotting.plt, "hist", side_effect=capture(expected_histograms)):
            for species, name in enumerate(self.names):
                mask = self.particles.active[:, :, :, species, :].reshape(-1)
                record = SimpleNamespace(
                    x_diagnostic=self.particles.x[:, :, :, species, :, :].reshape(-1, 3)[mask],
                    u=self.particles.u[:, :, :, species, :, :].reshape(-1, 3)[mask])
                plotting.plot_initial_histograms(record, self.dynamic, name.replace(" ", "_"), "/unused")

        with patch.object(plotting.plt, "savefig") as actual_save, \
             patch.object(plotting.plt, "xlabel") as actual_label, \
             patch.object(plotting.plt, "ylabel"), \
             patch.object(plotting.plt, "xlim") as actual_limits, \
             patch.object(plotting.plt, "title") as actual_titles, \
             patch.object(plotting.plt, "hist", side_effect=capture(actual_histograms)):
            plotting.plot_initial_tiled_histograms(self.particles, self.names, self.dynamic, "/unused")

        self.assertEqual(len(actual_histograms), 18)
        for actual, expected in zip(actual_histograms, expected_histograms):
            self.assertTrue(bool(jnp.array_equal(actual[0], expected[0])))
            self.assertTrue(bool(jnp.array_equal(actual[1], expected[1])))
        self.assertEqual(actual_save.call_args_list, expected_save.call_args_list)
        self.assertEqual(actual_label.call_args_list, expected_label.call_args_list)
        self.assertEqual(actual_limits.call_args_list, expected_limits.call_args_list)
        self.assertEqual(actual_titles.call_args_list, expected_titles.call_args_list)

    def test_only_addressable_particle_data_are_copied_to_host(self):
        copied_shapes = []
        device_get = jax.device_get

        def copy_local(value):
            if any(value is array for array in self.particles):
                raise AssertionError("The global particle array was copied to the host")
            copied_shapes.append(value.shape)
            return device_get(value)

        with patch.object(plotting.jax, "device_get", side_effect=copy_local), \
             patch.object(plotting, "particles_for_output", side_effect=AssertionError("global particle flattening")):
            counts, edges = plotting.initial_tiled_histogram_data(self.particles)
        self.assertEqual(len(copied_shapes), 12)
        self.assertTrue(all(shape[0] == 1 for shape in copied_shapes))
        self.assertTrue(bool(jnp.array_equal(jnp.sum(jnp.asarray(counts), axis=-1),
                                           jnp.asarray([[4]*6, [2]*6, [0]*6]))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.asarray(edges)))))

    def test_no_species_produces_no_histogram_files(self):
        particles = TiledParticles(self.particles.x[:, :, :, :0], self.particles.u[:, :, :, :0],
                                   self.particles.active[:, :, :, :0])
        with patch.object(plotting.plt, "savefig") as savefig:
            plotting.plot_initial_tiled_histograms(particles, (), self.dynamic, "/unused")
        savefig.assert_not_called()


if __name__ == "__main__":
    unittest.main()
