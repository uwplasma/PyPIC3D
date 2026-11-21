from tests.support.distributed.sharded_openpmd import (
    ExitStack,
    Path,
    SimpleNamespace,
    StringIO,
    TiledParticles,
    async_writer,
    field_with_guard_sentinels,
    io,
    jax,
    jnp,
    openPMD,
    output_adapters,
    parameters,
    particle_fixture,
    patch,
    read_particle_records,
    redirect_stderr,
    redirect_stdout,
    replace,
    shard,
    simulation_config,
    tempfile,
    unittest,
)


class ShardedOpenPMDTestsFixtures:
    def require_two_devices(self):
        if jax.local_device_count() < 2:
            self.skipTest("Run with JAX_NUM_CPU_DEVICES=2")
    def assert_close(self, actual, expected):
        self.assertTrue(bool(jnp.allclose(jnp.asarray(actual), jnp.asarray(expected), rtol=1.e-12, atol=1.e-12)))
