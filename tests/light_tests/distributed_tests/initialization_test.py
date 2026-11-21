"""Focused cross-device regression tests."""

from types import (
    SimpleNamespace,
)
from tests.support.initialization_fixtures import (
    _resolve_particle_batch_size,
)
from tests.support.initialization_fixtures import (
    InitializationFunctionsFixtures,
    TiledParticles,
    build_yee_grid,
    initialize_simulation,
    jnp,
    np,
    os,
    patch,
    tempfile,
    time_loop_electrodynamic,
    toml,
    unittest,
)


class TestInitializationFunctions(InitializationFunctionsFixtures, unittest.TestCase):


    def test_automatic_particle_batch_size_counts_only_local_mesh_devices(self):
        particles = self._particles_with_active(jnp.ones((3, 1, 1, 1, 900), dtype=bool))
        devices = tuple(SimpleNamespace(process_index=process) for process in (0, 1, 0))
        mesh = SimpleNamespace(devices=SimpleNamespace(flat=devices))
        with patch("PyPIC3D.initialization.jax.process_index", return_value=0):
            batch_size, reason = _resolve_particle_batch_size(
                particles, None, field_mesh=mesh, platform="cpu", available_cpu_threads=8,
            )
        self.assertEqual(batch_size, 512)
        self.assertIn("2 local devices", reason)

    def test_automatic_particle_batch_size_uses_largest_active_tile(self):
        active = np.zeros((2, 1, 1, 1, 32), dtype=bool)
        active[0, 0, 0, 0, :7] = True
        active[1, 0, 0, 0, :13] = True
        particles = self._particles_with_active(active)

        batch_size, _ = _resolve_particle_batch_size(
            particles,
            None,
            platform="cpu",
            available_cpu_threads=64,
            local_device_count=2,
        )

        self.assertEqual(batch_size, 13)

    def test_automatic_particle_batch_size_uses_cpu_threads_per_local_device(self):
        active = np.zeros((2, 2, 1, 1, 1000), dtype=bool)
        active[..., :900] = True
        particles = self._particles_with_active(active)

        wider, _ = _resolve_particle_batch_size(
            particles,
            None,
            platform="cpu",
            available_cpu_threads=16,
            local_device_count=4,
        )
        narrower, _ = _resolve_particle_batch_size(
            particles,
            None,
            platform="cpu",
            available_cpu_threads=8,
            local_device_count=4,
        )

        self.assertEqual(wider, 512)
        self.assertEqual(narrower, 256)

    def test_automatic_particle_batch_size_uses_accelerator_fallback(self):
        active = np.zeros((2, 1, 1, 1, 2048), dtype=bool)
        active[..., :1500] = True
        particles = self._particles_with_active(active)

        batch_size, reason = _resolve_particle_batch_size(particles, None, platform="gpu")

        self.assertEqual(batch_size, 1024)
        self.assertIn("gpu", reason)


if __name__ == "__main__":
    unittest.main()
