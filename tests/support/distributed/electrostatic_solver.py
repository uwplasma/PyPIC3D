from tests.support.electrostatic_solver_fixtures import (
    TiledLocalSchwarzFixtures as SharedFixtures,
    jax,
)


class TiledLocalSchwarzFixtures(SharedFixtures):
    def _require_devices(self, count):
        if jax.device_count() < count:
            raise RuntimeError(f"Need {count} JAX devices, got {jax.device_count()}")
