"""Per-process openPMD naming, ownership metadata, and the merge round trip."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
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

    def _write_per_process(self, folder, snapshot, kind):
        """Write one series per process the way a real multiprocess run does."""
        for process in range(2):
            with patch("jax.process_index", return_value=process), \
                 patch("jax.process_count", return_value=2):
                if kind == "fields":
                    openPMD.write_tiled_field_snapshot_openpmd(
                        _split_field_snapshot(snapshot, process),
                        output_dir=str(folder), filename="fields",
                        dynamic_parameters=self.dynamic, layout=self.layout,
                        file_extension=".h5")
                else:
                    openPMD.write_tiled_particle_snapshot_openpmd(
                        _split_particle_snapshot(snapshot, process),
                        output_dir=str(folder), filename="particles",
                        static_parameters=self.static, dynamic_parameters=self.dynamic,
                        file_extension=".h5")

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
