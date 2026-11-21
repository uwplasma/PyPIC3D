from tests.support.distributed_sharded_openpmd_fixtures import (
    Path,
    ShardedOpenPMDTestsFixtures,
    StringIO,
    io,
    jax,
    jnp,
    read_particle_records,
    redirect_stderr,
    redirect_stdout,
    simulation_config,
    tempfile,
    unittest,
)


class ShardedOpenPMDTests(ShardedOpenPMDTestsFixtures, unittest.TestCase):


    def test_short_simulation_is_identical_with_output_enabled_and_disabled(self):
        self.require_two_devices()
        from PyPIC3D.__main__ import run_PyPIC3D

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            folder = Path(folder)
            position_file = folder / "positions.npy"
            jnp.save(position_file, jnp.array([-.8, .8]))
            outcomes = []
            for output in (False, True):
                config = simulation_config(folder / str(output), position_file, output=output)
                outcomes.append(run_PyPIC3D(config, field_initializers={"Ex": lambda x, y, z: .02 + 0. * (x+y+z)}))
            for expected, actual in zip(jax.tree.leaves(outcomes[0][4:]), jax.tree.leaves(outcomes[1][4:])):
                self.assertTrue(bool(jnp.array_equal(actual, expected)))
            for filename in ("fields.h5", "particles.h5"):
                self.assertFalse((folder / "False" / "data" / filename).exists())
                series = io.Series(str(folder / "True" / "data" / filename), io.Access.read_only)
                try:
                    self.assertEqual(list(series.iterations), [0, 1, 2])
                    self.assertAlmostEqual(series.iterations[2].time, .04)
                finally:
                    series.close()

    def test_initial_only_simulation_writes_initial_paths_without_runtime_files(self):
        self.require_two_devices()
        from PyPIC3D.__main__ import run_PyPIC3D

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            folder = Path(folder)
            position_file = folder / "positions.npy"
            positions = jnp.array([-.8, .8])
            jnp.save(position_file, positions)
            output_dir = folder / "initial_only"
            config = simulation_config(output_dir, position_file, output=True, initial=True)
            run_PyPIC3D(config, field_initializers={"Ex": lambda x, y, z: .02 + 0. * (x+y+z)})
            data = output_dir / "data"
            self.assertFalse((data / "fields.h5").exists())
            self.assertFalse((data / "particles.h5").exists())
            self.assertTrue((data / "initial_fields" / "initial_fields.h5").is_file())
            records, time = read_particle_records(data / "initial_particles" / "test_beam_initial_particles.h5")
            self.assertEqual(time, 0.)
            self.assert_close(records["test_beam"]["position/x"], positions)
            for suffix in ("x", "y", "z", "x_velocity", "y_velocity", "z_velocity"):
                self.assertTrue((data / ("test_beam_initial_" + suffix + "_histogram.png")).is_file())


if __name__ == "__main__":
    unittest.main()
