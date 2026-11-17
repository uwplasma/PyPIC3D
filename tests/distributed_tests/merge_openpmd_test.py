"""Per-process openPMD naming, ownership metadata, and the merge round trip."""

from dataclasses import replace
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import openpmd_api as io

from PyPIC3D.diagnostics import async_writer, merge_openpmd, openPMD
from tests.code_tests.sharded_openpmd_test import (
    field_with_guard_sentinels,
    parameters,
    particle_fixture,
    read_particle_records,
)


def _tile_of(index):
    return openPMD._slice_start(index[0])


def _split(host_shards, process):
    """Keep the shards whose global tile belongs to one process."""
    return [(index, data) for index, data in host_shards if _tile_of(index) == process]


def _split_field_snapshot(snapshot, process):
    fields = {}
    for name, value in snapshot.fields.items():
        if isinstance(value, tuple) and len(value) == 3:
            fields[name] = tuple(_split(component, process) for component in value)
        else:
            fields[name] = _split(value, process)
    return replace(snapshot, fields=fields)


def _split_particle_snapshot(snapshot, process):
    changed = {name: _split(getattr(snapshot, name), process)
               for name in ("x_shards", "u_shards", "active_shards")}
    if snapshot.gamma_shards is not None:
        changed["gamma_shards"] = _split(snapshot.gamma_shards, process)
    return replace(snapshot, **changed)


class SeriesDiscoveryTests(unittest.TestCase):
    def test_directory_member_and_pointer_discover_the_same_group(self):
        for extension in (".h5", ".bp"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                data = folder / "data"
                data.mkdir()
                members = [data / f"fields_p{process:03d}{extension}" for process in range(2)]
                for member in members:
                    if extension == ".bp":
                        member.mkdir()
                        (member / "md.idx").touch()
                    else:
                        member.touch()
                pointer = folder / "fields.pmd"
                pointer.write_text(f"data/{members[0].name}\n")
                expected = [(str(data), "fields", extension, tuple(str(path) for path in members))]
                for entry in (data, members[0], pointer):
                    self.assertEqual(merge_openpmd.discover_series_groups([entry]), expected)

    def test_missing_member_and_pointer_target_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for extension in (".h5", ".bp"):
                missing = folder / f"fields_p000{extension}"
                pointer = folder / "fields.pmd"
                pointer.write_text(missing.name + "\n")
                for entry in (missing, pointer):
                    with self.subTest(entry=entry, extension=extension):
                        with self.assertRaises(FileNotFoundError):
                            merge_openpmd.discover_series_groups([entry])


class MergeOpenPMDTests(unittest.TestCase):
    def setUp(self):
        if jax.device_count() < 2:
            self.skipTest("Run with JAX_NUM_CPU_DEVICES>=2")
        self.static, self.dynamic = parameters(num_tiles=2, shape=(8, 2, 1))
        self.layout = openPMD.TiledMeshLayout(
            global_shape=(8, 2, 1),
            tile_shape=tuple(int(width) for width in self.static.tile_shape),
            guard_cells=int(self.static.guard_cells),
        )

    def test_process_suffix_only_appears_for_multiprocess_runs(self):
        self.assertEqual(openPMD.process_series_filename("fields.h5", 0, 1), "fields.h5")
        self.assertEqual(openPMD.process_series_filename("fields.h5", 3, 4), "fields_p003.h5")
        self.assertEqual(
            openPMD.process_series_filename("ion_initial_particles.h5", 2, 16),
            "ion_initial_particles_p002.h5")

    def _write_per_process(self, folder, snapshot, kind, extension=".h5"):
        """Write one series per process the way a real multiprocess run does."""
        for process in range(2):
            with patch("jax.process_index", return_value=process), \
                 patch("jax.process_count", return_value=2):
                if kind == "fields":
                    openPMD.write_tiled_field_snapshot_openpmd(
                        _split_field_snapshot(snapshot, process),
                        output_dir=str(folder), filename="fields",
                        dynamic_parameters=self.dynamic, layout=self.layout,
                        file_extension=extension)
                else:
                    openPMD.write_tiled_particle_snapshot_openpmd(
                        _split_particle_snapshot(snapshot, process),
                        output_dir=str(folder), filename="particles",
                        static_parameters=self.static, dynamic_parameters=self.dynamic,
                        file_extension=extension)

    def test_custom_output_pointer_lives_beside_the_merged_series(self):
        field, reference = field_with_guard_sentinels(self.static, (8, 2, 1))
        snapshot = async_writer.make_tiled_field_snapshot({"lapse": field}, step=4, time=0.2)
        for location in ("same", "other", "relative"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                source = folder / "data"
                self._write_per_process(source, snapshot, "fields")
                source_pointer = source / "fields.pmd"
                source_pointer.write_text("previous.h5\n")
                destination = {"same": source, "other": folder / "results", "relative": folder}[location]
                destination.mkdir(exist_ok=True)
                target = destination / "custom.h5"
                previous_directory = os.getcwd()
                try:
                    os.chdir(folder)
                    merge_openpmd.merge_paths([source], output="custom.h5" if location == "relative" else str(target))
                finally:
                    os.chdir(previous_directory)

                pointer = target.with_suffix(".pmd")
                self.assertEqual(pointer.read_text(), "custom.h5\n")
                self.assertEqual(source_pointer.read_text(), "previous.h5\n")
                series = io.Series(merge_openpmd._resolve_pmd(pointer), io.Access.read_only)
                try:
                    iteration = series.iterations[4]
                    self.assertAlmostEqual(iteration.time, 0.2)
                    record = iteration.meshes["lapse"][io.Mesh_Record_Component.SCALAR]
                    values = record.load_chunk()
                    series.flush()
                    self.assertTrue(bool(jnp.array_equal(jnp.asarray(values), reference)))
                finally:
                    series.close()

    def test_no_pmd_preserves_existing_pointers(self):
        field, _ = field_with_guard_sentinels(self.static, (8, 2, 1))
        snapshot = async_writer.make_tiled_field_snapshot({"lapse": field}, step=0, time=0.)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self._write_per_process(folder, snapshot, "fields")
            pointer = folder / "fields.pmd"
            pointer.write_text("previous.h5\n")
            for output in (None, str(folder / "custom.h5")):
                merge_openpmd.merge_paths([folder], output=output, write_pmd=False)
                self.assertTrue(Path(output or folder / "fields.h5").exists())
                self.assertEqual(pointer.read_text(), "previous.h5\n")
                self.assertFalse((folder / "custom.pmd").exists())

    @unittest.skipUnless(io.variants.get("adios2", False), "ADIOS2 backend is unavailable")
    def test_bp_member_and_pointer_merge_preserve_values_and_metadata(self):
        field, reference = field_with_guard_sentinels(self.static, (8, 2, 1))
        snapshot = async_writer.make_tiled_field_snapshot(
            {"J": (field, field, field)}, step=4, time=0.2, current_time_offset=-0.05)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self._write_per_process(folder, snapshot, "fields", extension=".bp")
            member = folder / "fields_p000.bp"
            pointer = folder / "input.pmd"
            pointer.write_text(member.name + "\n")
            source = io.Series(str(member), io.Access.read_only)
            try:
                mesh = source.iterations[4].meshes["J"]
                position = tuple(mesh["x"].position)
                dimensions = tuple(mesh.unit_dimension)
            finally:
                source.close()
            for entry, name in ((member, "direct"), (pointer, "via_pointer")):
                with self.subTest(entry=entry):
                    target = folder / f"{name}.bp"
                    merge_openpmd.merge_paths([entry], output=str(target))
                    merged_pointer = target.with_suffix(".pmd")
                    self.assertEqual(merged_pointer.read_text(), target.name + "\n")
                    series = io.Series(merge_openpmd._resolve_pmd(merged_pointer), io.Access.read_only)
                    try:
                        iteration = series.iterations[4]
                        self.assertAlmostEqual(iteration.time, 0.2)
                        self.assertAlmostEqual(iteration.dt, float(self.dynamic.dt))
                        mesh = iteration.meshes["J"]
                        self.assertAlmostEqual(mesh.time_offset, -0.05)
                        self.assertEqual(tuple(mesh.unit_dimension), dimensions)
                        record = mesh["x"]
                        self.assertEqual(tuple(record.shape), (8, 2, 1))
                        self.assertEqual(tuple(record.position), position)
                        values = record.load_chunk()
                        series.flush()
                        self.assertTrue(bool(jnp.array_equal(jnp.asarray(values), reference)))
                    finally:
                        series.close()

    def test_merged_fields_reproduce_the_whole_domain_mesh(self):
        field, reference = field_with_guard_sentinels(self.static, (8, 2, 1))
        snapshot = async_writer.make_tiled_field_snapshot(
            {"lapse": field, "E": (field, field, field)}, step=4, time=0.2)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self._write_per_process(folder, snapshot, "fields")
            self.assertTrue((folder/"fields_p000.h5").exists())
            self.assertTrue((folder/"fields_p001.h5").exists())
            self.assertFalse((folder/"fields.h5").exists())

            merge_openpmd.merge_paths([str(folder)])
            self.assertTrue((folder/"fields.h5").exists())
            # The merge restores the pointer a single-process run writes itself.
            self.assertEqual((folder/"fields.pmd").read_text().strip(), "fields.h5")

            series = io.Series(str(folder/"fields.h5"), io.Access.read_only)
            try:
                iteration = series.iterations[4]
                self.assertAlmostEqual(float(iteration.time), 0.2)
                scalar = iteration.meshes["lapse"][io.Mesh_Record_Component.SCALAR]
                electric = iteration.meshes["E"]["x"]
                self.assertEqual(tuple(scalar.shape), (8, 2, 1))
                values = scalar.load_chunk()
                vector = electric.load_chunk()
                series.flush()
                self.assertTrue(bool(jnp.allclose(jnp.asarray(values), reference)))
                self.assertTrue(bool(jnp.allclose(jnp.asarray(vector), reference)))
            finally:
                series.close()

    def test_merged_particles_match_a_single_process_write(self):
        particles, species, names = particle_fixture(self.static)
        snapshot = async_writer.make_tiled_particle_snapshot(
            particles, step=0, time=0.5, species_names=names, species_config=species)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            openPMD.write_tiled_particle_snapshot_openpmd(
                snapshot, output_dir=str(folder), filename="reference",
                static_parameters=self.static, dynamic_parameters=self.dynamic,
                file_extension=".h5")
            expected, _ = read_particle_records(folder/"reference.h5")

            self._write_per_process(folder, snapshot, "particles")
            merge_openpmd.merge_paths([str(folder/"particles_p000.h5")])
            merged, time = read_particle_records(folder/"particles.h5")

        self.assertAlmostEqual(float(time), 0.5)
        self.assertEqual(set(merged), set(expected))
        for name, records in expected.items():
            for record_name, values in records.items():
                # Global tile order must survive the split, independent of which
                # process held which tile.
                self.assertTrue(bool(jnp.array_equal(merged[name][record_name], values)),
                                f"{name}/{record_name} differs after merge")

    def test_merged_fields_survive_batched_draining(self):
        field, reference = field_with_guard_sentinels(self.static, (8, 2, 1))
        snapshot = async_writer.make_tiled_field_snapshot({"lapse": field}, step=0, time=0.)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self._write_per_process(folder, snapshot, "fields")
            # One tile per batch forces more than one load/store/flush cycle.
            merge_openpmd.merge_paths([str(folder)], max_pending_bytes=1)
            series = io.Series(str(folder/"fields.h5"), io.Access.read_only)
            try:
                record = series.iterations[0].meshes["lapse"][io.Mesh_Record_Component.SCALAR]
                values = record.load_chunk()
                series.flush()
                self.assertTrue(bool(jnp.allclose(jnp.asarray(values), reference)))
            finally:
                series.close()

    def test_merge_rejects_an_incomplete_group(self):
        field, _ = field_with_guard_sentinels(self.static, (8, 2, 1))
        snapshot = async_writer.make_tiled_field_snapshot({"lapse": field}, step=0, time=0.)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self._write_per_process(folder, snapshot, "fields")
            (folder/"fields_p001.h5").unlink()
            with self.assertRaisesRegex(ValueError, "Expected processes 0..1"):
                merge_openpmd.merge_paths([str(folder)])

    def test_ownership_metadata_partitions_the_tile_grid(self):
        field, _ = field_with_guard_sentinels(self.static, (8, 2, 1))
        snapshot = async_writer.make_tiled_field_snapshot({"lapse": field}, step=0, time=0.)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self._write_per_process(folder, snapshot, "fields")
            sources = [io.Series(str(folder/f"fields_p00{index}.h5"), io.Access.read_only)
                       for index in range(2)]
            try:
                geometry = merge_openpmd.read_tile_geometry(
                    [series.iterations[0] for series in sources])
                self.assertEqual(geometry.global_shape, (8, 2, 1))
                self.assertEqual(geometry.tile_counts, (2, 1, 1))
                merge_openpmd.validate_tile_coverage(geometry)
                self.assertEqual([ids.tolist() for ids in geometry.owned], [[0], [1]])
            finally:
                for series in sources:
                    series.close()


if __name__ == "__main__":
    unittest.main()
