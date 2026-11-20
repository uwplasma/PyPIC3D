from tests.distributed_tests.fixtures.sharded_openpmd import (
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


class ShardedOpenPMDTests(unittest.TestCase):
    def require_two_devices(self):
        if jax.local_device_count() < 2:
            self.skipTest("Run with JAX_NUM_CPU_DEVICES=2")

    def assert_close(self, actual, expected):
        self.assertTrue(bool(jnp.allclose(jnp.asarray(actual), jnp.asarray(expected), rtol=1.e-12, atol=1.e-12)))

    def test_tile_chunks_strip_scalar_and_per_axis_guards(self):
        tile_shape = (2, 3, 1)
        reference = jnp.arange(12, dtype=jnp.float64).reshape((2, 1, 1) + tile_shape)
        shard_index = (slice(2, 4), slice(0, 1), slice(0, 1)) + (slice(None),) * 3
        for guards, widths in ((0, (0, 0, 0)), (2, (2, 2, 2)),
                               ((0, 1, 2), (0, 1, 2)), ([2, 0, 1], (2, 0, 1))):
            with self.subTest(guards=guards):
                data = jnp.pad(reference, ((0, 0),) * 3 + tuple((g, g) for g in widths),
                               constant_values=-9999.)
                layout = SimpleNamespace(tile_shape=tile_shape, guard_cells=guards)
                chunks = list(openPMD._iter_tile_chunks_from_host_shard(shard_index, data, layout=layout))

                self.assertEqual(len(chunks), 2)
                for local_tile, (offset, interior) in enumerate(chunks):
                    self.assertEqual(offset, [(2 + local_tile) * tile_shape[0], 0, 0])
                    self.assertEqual(interior.shape, tile_shape)
                    self.assertTrue(jnp.array_equal(interior, reference[local_tile, 0, 0]))

    def test_mesh_chunks_cover_periodic_mixed_and_singleton_grids(self):
        for num_tiles in (1, 2):
            if num_tiles > jax.local_device_count():
                continue
            for shape, boundaries in (((8, 2, 1), (0, 0, 0)), ((8, 1, 3), (1, 0, 1))):
                with self.subTest(devices=num_tiles, shape=shape, boundaries=boundaries):
                    static, dynamic = parameters(num_tiles, shape, boundaries)
                    scalar, reference = field_with_guard_sentinels(static, shape)
                    fields = {"E": (scalar, scalar + 100., scalar + 200.), "rho": scalar + 300.}
                    with tempfile.TemporaryDirectory() as folder:
                        writer = async_writer.create_async_tiled_openpmd_field_writer(static, dynamic, folder, queue_size=1)
                        try:
                            for step in (0, 3):
                                self.assertTrue(async_writer.enqueue_openpmd_field_output(writer, fields, dynamic, step, step))
                        finally:
                            writer.close()
                        series = io.Series(str(Path(folder) / "fields.h5"), io.Access.read_only)
                        try:
                            self.assertEqual(series.get_attribute("software"), "PyPIC3D")
                            self.assertTrue(series.get_attribute("softwareVersion"))
                            self.assertEqual(list(series.iterations), [0, 3])
                            iteration = series.iterations[3]
                            self.assertAlmostEqual(iteration.time, .6)
                            self.assertAlmostEqual(iteration.dt, .2)
                            self.assertEqual(tuple(iteration.meshes["E"].axis_labels), ("x", "y", "z"))
                            self.assert_close(iteration.meshes["E"].grid_global_offset, (-2., -1., -.5))
                            pending = []
                            for component, offset in zip("xyz", (0., 100., 200.)):
                                record = iteration.meshes["E"][component]
                                self.assertEqual(tuple(record.shape), shape)
                                pending.append((record.load_chunk(), reference + offset))
                            pending.append((iteration.meshes["rho"][io.Mesh_Record_Component.SCALAR].load_chunk(), reference + 300.))
                            series.flush()
                            for actual, expected in pending:
                                self.assert_close(actual, expected)
                        finally:
                            series.close()

    def test_particle_order_and_records_survive_independently_reordered_shards(self):
        self.require_two_devices()
        static, dynamic = parameters()
        particles, species, names = particle_fixture(static)
        snapshot = async_writer.make_tiled_particle_snapshot(particles, step=2, time=.4, species_names=names, species_config=species)
        # Each array's shard list may have a different enumeration order.
        snapshot = replace(snapshot, x_shards=snapshot.x_shards[::-1], active_shards=snapshot.active_shards[::-1])
        with tempfile.TemporaryDirectory() as folder:
            openPMD.write_tiled_particle_snapshot_openpmd(snapshot, output_dir=folder, filename="particles",
                static_parameters=static, dynamic_parameters=dynamic, file_extension=".h5")
            records, time = read_particle_records(Path(folder) / "particles.h5", step=2)
        self.assertAlmostEqual(time, .4)
        self.assertEqual(set(records), {name.replace(" ", "_") for name in names})
        for index, name in enumerate(names):
            mask = particles.active[:, :, :, index, :].reshape(-1)
            positions = particles.x[:, :, :, index, :, :].reshape(-1, 3)[mask]
            velocity = particles.u[:, :, :, index, :, :].reshape(-1, 3)[mask]
            expected_positions = positions - velocity * dynamic.dt / 2
            expected_positions = expected_positions.at[:, 0].set((expected_positions[:, 0] + 2.) % 4. - 2.)
            gamma = 1. / jnp.sqrt(1. - jnp.sum(velocity**2, axis=1) / dynamic.C**2)
            result = records[name.replace(" ", "_")]
            self.assertEqual(result["gamma"].shape, (int(jnp.count_nonzero(mask)),))
            for axis, component in enumerate("xyz"):
                self.assert_close(result["position/" + component], expected_positions[:, axis])
                self.assert_close(result["momentum/" + component], species.mass[index] * gamma * velocity[:, axis])
                self.assert_close(result["positionOffset/" + component], jnp.zeros_like(gamma))
            self.assert_close(result["gamma"], gamma)
            for record_name, values in (("charge", species.charge), ("mass", species.mass), ("weighting", species.weight)):
                self.assert_close(result[record_name], jnp.full(gamma.shape, values[index]))

    def test_initial_particles_keep_stored_positions_and_separate_species_paths(self):
        self.require_two_devices()
        static, dynamic = parameters()
        particles, species, names = particle_fixture(static)
        with tempfile.TemporaryDirectory() as folder:
            openPMD.write_openpmd_initial_particles(particles, static, dynamic, folder,
                species_config=species, species_names=names)
            snapshot = async_writer.make_tiled_particle_snapshot(particles, step=0, time=0.,
                species_config=species, species_names=names, initial=True)
            openPMD.write_tiled_particle_snapshot_openpmd(snapshot, output_dir=folder, filename="explicit_initial",
                static_parameters=static, dynamic_parameters=dynamic, file_extension=".h5")
            explicit, _ = read_particle_records(Path(folder) / "explicit_initial.h5")
            for index, name in enumerate(names):
                name = name.replace(" ", "_")
                records, time = read_particle_records(Path(folder) / "data" / "initial_particles" / (name + "_initial_particles.h5"))
                self.assertEqual(set(records), {name})
                self.assertEqual(time, 0.)
                mask = particles.active[:, :, :, index, :].reshape(-1)
                expected = particles.x[:, :, :, index, :, 0].reshape(-1)[mask]
                self.assert_close(records[name]["position/x"], expected)
                for record_name, values in records[name].items():
                    self.assert_close(values, explicit[name][record_name])

    def test_bounded_async_particle_queue_keeps_each_iteration_snapshot(self):
        self.require_two_devices()
        static, dynamic = parameters()
        particles, species, names = particle_fixture(static)
        with tempfile.TemporaryDirectory() as folder:
            writer = async_writer.create_async_tiled_openpmd_particle_writer(static, dynamic, folder, queue_size=1)
            try:
                for step in range(3):
                    state = particles._replace(x=particles.x + step * .01)
                    self.assertTrue(async_writer.enqueue_openpmd_particle_output(writer, state, dynamic, step, step,
                        species_config=species, species_names=names))
            finally:
                writer.close()
            for step in range(3):
                records, time = read_particle_records(Path(folder) / "particles.h5", step)
                self.assertAlmostEqual(time, step * .2)
                expected = jnp.array([-1.95, -.7, .8]) + step * .01 - jnp.array([.8, .2, -.4]) * .1
                expected = (expected + 2.) % 4. - 2.
                self.assert_close(records["beam_electrons"]["position/x"], expected)

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

    def test_initial_histograms_match_50_bin_reference_without_global_transfer(self):
        self.require_two_devices()
        from PyPIC3D.diagnostics import plotting

        static, _ = parameters()
        particles, _, _ = particle_fixture(static)
        device_get = jax.device_get

        def local_device_get(value):
            if any(value is array for array in (particles.x, particles.u, particles.active)):
                raise AssertionError("Histogram transferred a complete distributed array")
            return device_get(value)

        with patch.object(jax, "device_get", side_effect=local_device_get), \
             patch.object(plotting, "particles_for_output", side_effect=AssertionError("Histogram flattened particles")):
            counts, edges = plotting.initial_tiled_histogram_data(particles)
        self.assertEqual(counts.shape, (3, 6, 50))
        self.assertEqual(edges.shape, (3, 6, 51))
        for species in range(3):
            mask = particles.active[:, :, :, species, :].reshape(-1)
            for component in range(6):
                values = particles.x if component < 3 else particles.u
                selected = values[:, :, :, species, :, component % 3].reshape(-1)[mask]
                # Use the persisted edges for bin membership: JAX and NumPy
                # can round a generated edge to opposite sides of a particle.
                reference_counts, _ = jnp.histogram(selected, bins=jnp.asarray(edges[species, component]))
                lower = float(jnp.min(selected)) if selected.size else 0.
                upper = float(jnp.max(selected)) if selected.size else 1.
                if lower == upper:
                    lower, upper = lower - .5, upper + .5
                reference_edges = jnp.linspace(lower, upper, 51)
                self.assert_close(counts[species, component], reference_counts)
                self.assert_close(edges[species, component], reference_edges)

    def test_covariant_momenta_and_supplied_gamma_remain_paired_with_their_tiles(self):
        self.require_two_devices()
        static, dynamic = parameters()
        static = static._replace(solver="static_metric")
        particles, species, names = particle_fixture(static)
        # A supplied nontrivial metric Gamma must not be replaced with the
        # special-relativistic three-velocity expression.
        gamma = shard(jnp.arange(particles.active.size, dtype=jnp.float64).reshape(particles.active.shape) / 20. + 1., static)
        snapshot = async_writer.make_tiled_particle_snapshot(particles, step=0, time=0., species_names=names, species_config=species, gamma=gamma)
        snapshot = replace(snapshot, x_shards=snapshot.x_shards[::-1], gamma_shards=snapshot.gamma_shards[::-1])
        with tempfile.TemporaryDirectory() as folder:
            openPMD.write_tiled_particle_snapshot_openpmd(snapshot, output_dir=folder, filename="particles",
                static_parameters=static, dynamic_parameters=dynamic, file_extension=".h5")
            records, _ = read_particle_records(Path(folder) / "particles.h5")
        for index, name in enumerate(names):
            mask = particles.active[:, :, :, index, :].reshape(-1)
            result = records[name.replace(" ", "_")]
            self.assert_close(result["gamma"], gamma[:, :, :, index, :].reshape(-1)[mask])
            for axis, component in enumerate("xyz"):
                self.assert_close(result["position/" + component], particles.x[:, :, :, index, :, axis].reshape(-1)[mask])
                self.assert_close(result["momentum/" + component], species.mass[index] * particles.u[:, :, :, index, :, axis].reshape(-1)[mask])

    def test_dynamic_and_dark_metadata_keep_native_positions_and_sample_times(self):
        static, dynamic = parameters(num_tiles=1)
        scalar, _ = field_with_guard_sentinels(static, (8, 2, 1))
        for is_dynamic in (False, True):
            with self.subTest(dynamic=is_dynamic):
                state = dynamic._replace(bssn_parameters=SimpleNamespace(time=jnp.asarray(2.75))) if is_dynamic else dynamic
                names = ("E", "B", "J") if is_dynamic else ("dark_E", "dark_B", "dark_A")
                fields = {name: (scalar, scalar, scalar) for name in names}
                fields["lapse" if is_dynamic else "dark_phi"] = scalar
                with tempfile.TemporaryDirectory() as folder:
                    writer = async_writer.create_async_tiled_openpmd_field_writer(static, state, folder)
                    try:
                        async_writer.enqueue_openpmd_field_output(writer, fields, state, 0, 0)
                        async_writer.enqueue_openpmd_field_output(writer, fields, state, 2, 99)
                    finally:
                        writer.close()
                    series = io.Series(str(Path(folder) / "fields.h5"), io.Access.read_only)
                    try:
                        iteration = series.iterations[2]
                        electric, magnetic = iteration.meshes[names[0]], iteration.meshes[names[1]]
                        self.assert_close(electric["x"].position, (.5, 0., 0.))
                        self.assert_close(magnetic["x"].position, (0., .5, .5))
                        self.assertEqual(electric.time_offset, 0.)
                        self.assertEqual(magnetic.time_offset, 0.)
                        if is_dynamic:
                            self.assertAlmostEqual(iteration.time, 2.75)
                            self.assertEqual(series.iterations[0].meshes["J"].time_offset, 0.)
                            self.assertAlmostEqual(iteration.meshes["J"].time_offset, -.05)
                            self.assertEqual(electric.get_attribute("normalization"), "Gaussian, c=1")
                        else:
                            self.assertAlmostEqual(iteration.time, 99 * .2)
                            self.assert_close(electric.unit_dimension, (1., 1., -3., -1., 0., 0., 0.))
                            self.assert_close(iteration.meshes["dark_phi"][io.Mesh_Record_Component.SCALAR].position, (0., 0., 0.))
                    finally:
                        series.close()

    def test_initial_and_runtime_output_never_gather_global_arrays(self):
        self.require_two_devices()
        static, dynamic = parameters()
        scalar, reference = field_with_guard_sentinels(static, (8, 2, 1))
        particles, species, names = particle_fixture(static)
        protected = (scalar, particles.x, particles.u, particles.active)
        device_get = jax.device_get

        def local_device_get(value):
            if any(value is array for array in protected):
                raise AssertionError("Writer transferred a complete distributed array")
            return device_get(value)

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(jax, "device_get", side_effect=local_device_get))
            for name in ("assemble_tiled_scalar_field", "assemble_tiled_vector_field", "particles_for_output"):
                stack.enter_context(patch.object(output_adapters, name, side_effect=AssertionError("Global output assembly")))
            # Also guard imported aliases in the initial-output module.
            stack.enter_context(patch.object(openPMD, "particles_for_output", create=True, side_effect=AssertionError("Global particle flattening")))
            stack.enter_context(patch.object(openPMD, "field_map_for_output", create=True, side_effect=AssertionError("Global field assembly")))
            openPMD.write_openpmd_initial_fields({"rho": scalar}, static, dynamic, folder)
            openPMD.write_openpmd_initial_particles(particles, static, dynamic, folder, species_config=species, species_names=names)
            field_snapshot = async_writer.make_tiled_field_snapshot({"rho": scalar}, step=1, time=.2)
            particle_snapshot = async_writer.make_tiled_particle_snapshot(particles, step=1, time=.2, species_config=species, species_names=names)
            layout = openPMD.TiledMeshLayout((8, 2, 1), static.tile_shape, static.guard_cells)
            openPMD.write_tiled_field_snapshot_openpmd(field_snapshot, output_dir=folder, filename="fields", dynamic_parameters=dynamic, layout=layout, file_extension=".h5")
            openPMD.write_tiled_particle_snapshot_openpmd(particle_snapshot, output_dir=folder, filename="particles", static_parameters=static, dynamic_parameters=dynamic, file_extension=".h5")
            series = io.Series(str(Path(folder) / "data" / "initial_fields" / "initial_fields.h5"), io.Access.read_only)
            try:
                actual = series.iterations[0].meshes["rho"][io.Mesh_Record_Component.SCALAR].load_chunk()
                series.flush()
                self.assert_close(actual, reference)
            finally:
                series.close()

    def test_host_snapshot_storage_is_bounded_by_addressable_shards(self):
        self.require_two_devices()
        static, _ = parameters()
        scalar, _ = field_with_guard_sentinels(static, (8, 2, 1))
        particles, species, names = particle_fixture(static)

        def local_view(array):
            return SimpleNamespace(shape=array.shape, addressable_shards=array.addressable_shards[:1])

        field_snapshot = async_writer.make_tiled_field_snapshot({"rho": local_view(scalar)}, step=0, time=0.)
        viewed = TiledParticles(*(local_view(array) for array in
                                  (particles.x, particles.u, particles.active)))
        particle_snapshot = async_writer.make_tiled_particle_snapshot(viewed,
            step=0, time=0., species_names=names, species_config=species)
        for array, host_shards in ((scalar, field_snapshot.fields["rho"]),
                                  (particles.x, particle_snapshot.x_shards),
                                  (particles.u, particle_snapshot.u_shards),
                                  (particles.active, particle_snapshot.active_shards)):
            self.assertEqual(len(host_shards), 1)
            self.assertEqual(sum(chunk.nbytes for _, chunk in host_shards), array.nbytes // 2)
            self.assertTrue(host_shards[0][1].flags.writeable)
            self.assertTrue(host_shards[0][1].flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
