"""Native input and owned-chunk tests; run with two logical CPU devices."""

import os
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import openpmd_api as io

from tests.kernel_fixtures import kernel_parameters
from PyPIC3D.distributed import tiled_field_from_initializer
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.utilities.toml_helpers import load_external_fields_from_toml
from PyPIC3D.particles.distributed_initialization import load_distributed_particles
from PyPIC3D.particles.particle_initialization import load_particles_from_toml
from PyPIC3D.diagnostics.async_writer import make_tiled_field_snapshot
from PyPIC3D.diagnostics.parallel_openpmd import write_parallel_snapshot
from PyPIC3D.diagnostics.openPMD import TiledMeshLayout, _iter_tile_chunks_from_host_shard


class TestDistributedInputOutput(unittest.TestCase):
    def setUp(self):
        if jax.device_count() < 2:
            self.skipTest("Run with JAX_NUM_CPU_DEVICES=2")
        self.static, self.dynamic = kernel_parameters(
            Nx=8, Ny=4, Nz=4, tile_shape=(4, 4, 4), guard_cells=2)

    def test_zero_fields_and_coordinate_lines_keep_the_shared_mesh(self):
        fields = initialize_fields(self.static, self.dynamic)
        for field in fields[0] + fields[1]:
            self.assertEqual(field.sharding.mesh, self.static.field_mesh)
            self.assertTrue(bool(jnp.all(field == 0)))
        for coordinates in self.dynamic.grids.tiled_center_grid:
            self.assertEqual(coordinates.sharding.mesh, self.static.field_mesh)

    def test_native_field_reads_exact_owned_blocks_without_reading_halos(self):
        values = jnp.arange(8*4*4, dtype=jnp.float64).reshape((8, 4, 4))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "field.npy")
            jnp.save(path, values)
            field = tiled_field_from_initializer(path, self.static, self.dynamic)
        self.assertTrue(bool(jnp.array_equal(field[0, 0, 0, 2:-2, 2:-2, 2:-2], values[:4])))
        self.assertTrue(bool(jnp.array_equal(field[1, 0, 0, 2:-2, 2:-2, 2:-2], values[4:])))
        self.assertEqual(len(field.addressable_shards), 2)

    def test_local_callback_observes_the_native_yee_half_cell_offset(self):
        center = tiled_field_from_initializer(lambda x, y, z: x+2*y+3*z, self.static, self.dynamic)
        vertex = tiled_field_from_initializer(lambda x, y, z: x+2*y+3*z, self.static, self.dynamic,
                                              location=("V", "C", "C"))
        self.assertTrue(bool(jnp.allclose(vertex-center, self.dynamic.dx/2, rtol=0, atol=1.e-14)))

    def test_native_file_loader_avoids_global_device_load_on_a_shared_mesh(self):
        E, B, J, *_ = initialize_fields(self.static, self.dynamic)
        values = jnp.arange(8*4*4, dtype=jnp.float64).reshape((8, 4, 4))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "Ex.npy")
            jnp.save(path, values)
            config = {"field1": dict(name="Ex", type=0, path=path)}
            with patch("jax.numpy.load", side_effect=AssertionError("global device load")):
                fields, _ = load_external_fields_from_toml(list(E+B+J), (E, B), config,
                                                           self.static, self.dynamic)
        self.assertEqual(fields[0].sharding.mesh, self.static.field_mesh)
        self.assertTrue(bool(jnp.array_equal(fields[0][1, 0, 0, 2:-2, 2:-2, 2:-2], values[4:])))

    def test_bounded_particle_input_preserves_serial_order_and_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            data = (jnp.asarray([1.2, -1.2, 0.7, -0.4, 1.8]), jnp.zeros(5), jnp.zeros(5),
                    jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5]), jnp.zeros(5), jnp.zeros(5))
            block = dict(name="ions", N_particles=5, charge=1.0, mass=2.0, temperature=0.0, weight=3.0)
            for name, values in zip(("x", "y", "z", "vx", "vy", "vz"), data):
                path = os.path.join(folder, name + ".npy")
                jnp.save(path, values)
                block["initial_"+name] = path
            config = {"particle1": block}
            serial = load_particles_from_toml(config, self.static, self.dynamic)
            distributed = load_distributed_particles(config, self.static, self.dynamic, chunk_size=2)
        for reference, actual in zip(serial[0], distributed[0]):
            self.assertTrue(bool(jnp.array_equal(reference, actual)))
        self.assertEqual(serial[0].active.shape, distributed[0].active.shape)
        self.assertTrue(bool(jnp.array_equal(serial[1].weight, distributed[1].weight)))

    def test_distributed_random_particles_are_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "random particle generation"):
            load_distributed_particles({"particle1": dict(name="ions", N_particles=4, mass=1, charge=1)},
                                       self.static, self.dynamic)

    def test_empty_particle_storage_keeps_addressable_tile_ownership(self):
        particles, species, names, metadata = load_distributed_particles({}, self.static, self.dynamic)
        self.assertEqual(particles.x.shape, (2, 1, 1, 0, 0, 3))
        self.assertEqual(len(particles.x.addressable_shards), 2)
        self.assertEqual(species.update_x.shape, (0, 3))
        self.assertEqual((names, metadata), ((), ()))

    def test_owned_chunks_cover_the_output_mesh_without_guards_or_endpoints(self):
        field = tiled_field_from_initializer(lambda x, y, z: x+2*y+3*z, self.static, self.dynamic)
        snapshot = make_tiled_field_snapshot({"lapse": field}, step=0, time=0)
        layout = TiledMeshLayout((8, 4, 4), (4, 4, 4), guard_cells=2)
        chunks = [chunk for index, data in snapshot.fields["lapse"]
                  for chunk in _iter_tile_chunks_from_host_shard(index, data, layout=layout)]
        self.assertEqual([block.shape for _, block in chunks], [(4, 4, 4), (4, 4, 4)])
        coverage = jnp.zeros((8, 4, 4), dtype=jnp.int32)
        for offset, block in chunks:
            region = tuple(slice(start, start+n) for start, n in zip(offset, block.shape))
            coverage = coverage.at[region].add(1)
        self.assertTrue(bool(jnp.all(coverage == 1)))

    def test_parallel_writer_preserves_uniform_record_shapes_for_default_readers(self):
        static = self.static._replace(boundary_conditions=(1, 1, 1))
        field = tiled_field_from_initializer(lambda x, y, z: x+2*y+3*z, static, self.dynamic)
        snapshot = make_tiled_field_snapshot({"lapse": field, "E": (field, field, field)}, step=4, time=0.2)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "shared.h5")
            series = io.Series(path, io.Access.create)
            write_parallel_snapshot(series, None, static, self.dynamic, fields=snapshot)
            series.close()
            series = io.Series(path, io.Access.read_only)
            scalar = series.iterations[4].meshes["lapse"][io.Mesh_Record_Component.SCALAR]
            electric = series.iterations[4].meshes["E"]["x"]
            self.assertEqual(tuple(scalar.shape), (8, 4, 4))
            self.assertEqual(tuple(electric.shape), (8, 4, 4))
            values = scalar.load_chunk()
            series.flush()
            self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.asarray(values)))))
            self.assertAlmostEqual(float(values[4, 0, 0]), float(field[1, 0, 0, 2, 2, 2]))
            series.close()


if __name__ == "__main__":
    unittest.main()
