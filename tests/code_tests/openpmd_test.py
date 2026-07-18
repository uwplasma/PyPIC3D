import unittest
import threading
import time
from unittest.mock import patch

import jax.numpy as jnp

from PyPIC3D.diagnostics import async_writer
from PyPIC3D.diagnostics import openPMD
from PyPIC3D.diagnostics.openPMD import _ensure_openpmd_array
from tests.kernel_fixtures import (
    build_tiled_particles,
    field_tiles_from_global,
    kernel_parameters,
    kernel_parameters_from_values,
    particle_parameters_from_values,
    particle_species,
    species_names as names_for_species,
    vector_tiles_from_global,
)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, static_parameters, dynamic_parameters, num_guard_cells=None):
    return field_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells)


def tile_vector_field(field, static_parameters, dynamic_parameters, num_guard_cells=None):
    return vector_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells)


class FakeRecord:
    def __init__(self):
        self.shape = None
        self.dataset_shape = None
        self.unit_SI = None
        self.data = None
        self.chunks = []

    def reset_dataset(self, dataset):
        extent = getattr(dataset, "extent", None)
        if extent is not None:
            self.dataset_shape = tuple(extent)

    def store_chunk(self, array, offset, extent):
        self.shape = tuple(extent)
        self.data = jnp.asarray(array)
        self.chunks.append((tuple(offset), tuple(extent), jnp.asarray(array)))


class FakeMesh:
    def __init__(self):
        self.records = {}
        self.axis_labels = None
        self.grid_spacing = None
        self.grid_global_offset = None

    def __getitem__(self, component):
        if component not in self.records:
            self.records[component] = FakeRecord()
        return self.records[component]


class FakeMeshes(dict):
    def __getitem__(self, name):
        if name not in self:
            self[name] = FakeMesh()
        return dict.__getitem__(self, name)


class FakeParticleRecord(dict):
    def __getitem__(self, component):
        if component not in self:
            self[component] = FakeRecord()
        return dict.__getitem__(self, component)


class FakeParticleSpecies(dict):
    def __getitem__(self, record_name):
        if record_name not in self:
            if record_name in ("position", "positionOffset", "momentum"):
                self[record_name] = FakeParticleRecord()
            else:
                self[record_name] = FakeRecord()
        return dict.__getitem__(self, record_name)


class FakeParticles(dict):
    def __getitem__(self, species_name):
        if species_name not in self:
            self[species_name] = FakeParticleSpecies()
        return dict.__getitem__(self, species_name)


class FakeIteration:
    def __init__(self):
        self.meshes = FakeMeshes()
        self.particles = FakeParticles()
        self.time = None
        self.dt = None
        self.time_unit_SI = None


class FakeIterations(dict):
    def __getitem__(self, iteration):
        if iteration not in self:
            self[iteration] = FakeIteration()
        return dict.__getitem__(self, iteration)


class FakeSeries:
    def __init__(self):
        self.iterations = FakeIterations()

    def set_attribute(self, name, value):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _zero_field(shape):
    return tuple(jnp.zeros(shape) for _ in range(3))


def _parameter_values():
    return {
        "dt": 0.2,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "x_wind": 4.0,
        "y_wind": 2.0,
        "z_wind": 1.0,
        "Nx": 4,
        "Ny": 2,
        "Nz": 1,
        "tile_shape": (2, 1, 1),
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }


def _record_data(record):
    if len(record.chunks) <= 1:
        return record.data

    n = sum(int(extent[0]) for _offset, extent, _data in record.chunks)
    out = jnp.zeros((n,), dtype=record.chunks[0][2].dtype)
    for offset, extent, data in record.chunks:
        start = int(offset[0])
        stop = start + int(extent[0])
        out = out.at[start:stop].set(data)
    return out


def _species(name, charge, mass, weight, x1):
    return particle_species(
        name=name,
        charge=charge,
        mass=mass,
        weight=weight,
        x1=jnp.asarray(x1),
        u1=jnp.ones_like(x1) * 0.1,
    )


class OpenPMDDiagnosticsTests(unittest.TestCase):

    def test_openpmd_field_array_preserves_thin_y_axis(self):
        field_component = jnp.ones((4, 1, 6))

        array = _ensure_openpmd_array(field_component)

        self.assertEqual(array.shape, (4, 1, 6))

    def _expected_tiled_scalar_from_interior(self, field, static_parameters, dynamic_parameters, num_guard_cells):
        return field_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells)

    def _field_with_stale_global_ghosts(self, dynamic_parameters, value_offset=0.0):
        shape = (
            int(dynamic_parameters.Nx) + 2,
            int(dynamic_parameters.Ny) + 2,
            int(dynamic_parameters.Nz) + 2,
        )
        field = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float64).reshape(shape)
        field = field + value_offset
        field = field.at[0, :, :].set(-1000.0)
        field = field.at[-1, :, :].set(-2000.0)
        field = field.at[:, 0, :].set(-3000.0)
        field = field.at[:, -1, :].set(-4000.0)
        field = field.at[:, :, 0].set(-5000.0)
        field = field.at[:, :, -1].set(-6000.0)
        return field

    def test_tile_scalar_field_one_guard_rebuilds_periodic_halos_from_interiors(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(_parameter_values())
        field = self._field_with_stale_global_ghosts(dynamic_parameters)

        tiles = tile_scalar_field(field, static_parameters, dynamic_parameters, num_guard_cells=1)
        expected = self._expected_tiled_scalar_from_interior(
            field,
            static_parameters,
            dynamic_parameters,
            num_guard_cells=1,
        )

        self.assertTrue(jnp.allclose(tiles, expected))

    def test_tile_scalar_field_two_guards_rebuilds_periodic_halos_from_interiors(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(_parameter_values())
        field = self._field_with_stale_global_ghosts(dynamic_parameters, value_offset=10.0)

        tiles = tile_scalar_field(field, static_parameters, dynamic_parameters, num_guard_cells=2)
        expected = self._expected_tiled_scalar_from_interior(
            field,
            static_parameters,
            dynamic_parameters,
            num_guard_cells=2,
        )

        self.assertTrue(jnp.allclose(tiles, expected))

    def test_write_openpmd_fields_preserves_thin_y_mesh_metadata(self):
        shape_with_ghosts = (6, 3, 8)
        E = _zero_field(shape_with_ghosts)
        B = _zero_field(shape_with_ghosts)
        J = _zero_field(shape_with_ghosts)
        rho = jnp.zeros(shape_with_ghosts)
        phi = jnp.zeros(shape_with_ghosts)
        external_fields = _zero_field(shape_with_ghosts), _zero_field(shape_with_ghosts)
        fields = (E, B, J, rho, phi, external_fields)
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=4,
            Ny=1,
            Nz=6,
            x_wind=1.0,
            y_wind=2.0,
            z_wind=3.0,
            dx=0.25,
            dy=0.5,
            dz=0.75,
            dt=1.0,
            tile_shape=(4, 1, 6),
            guard_cells=1,
        )
        series = FakeSeries()

        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_openpmd_fields(fields, static_parameters, dynamic_parameters, "/tmp", plot_t=0, t=0)

        B_mesh = series.iterations[0].meshes["B"]
        self.assertEqual(B_mesh.axis_labels, ["x", "y", "z"])
        self.assertEqual(B_mesh.grid_spacing, [0.25, 0.5, 0.75])
        self.assertEqual(B_mesh.grid_global_offset, [-0.5, -1.0, -1.5])
        self.assertEqual(B_mesh.records["x"].shape, (4, 1, 6))

    def test_write_openpmd_fields_assembles_tiled_fields_before_output(self):
        shape_with_ghosts = (6, 4, 4)
        E = _zero_field(shape_with_ghosts)
        B = _zero_field(shape_with_ghosts)
        J = _zero_field(shape_with_ghosts)
        rho = jnp.zeros(shape_with_ghosts)
        phi = jnp.zeros(shape_with_ghosts)
        external_fields = _zero_field(shape_with_ghosts), _zero_field(shape_with_ghosts)
        parameter_values = {
            "dt": 1.0,
            "dx": 0.25,
            "dy": 0.5,
            "dz": 0.75,
            "x_wind": 1.0,
            "y_wind": 2.0,
            "z_wind": 3.0,
            "Nx": 4,
            "Ny": 2,
            "Nz": 2,
            "tile_shape": (2, 1, 1),
            "guard_cells": 2,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_values)
        tiled_fields = (
            tile_vector_field(E, static_parameters, dynamic_parameters),
            tile_vector_field(B, static_parameters, dynamic_parameters),
            tile_vector_field(J, static_parameters, dynamic_parameters),
            tile_scalar_field(rho, static_parameters, dynamic_parameters),
            tile_scalar_field(phi, static_parameters, dynamic_parameters),
            (
                tile_vector_field(external_fields[0], static_parameters, dynamic_parameters),
                tile_vector_field(external_fields[1], static_parameters, dynamic_parameters),
            ),
            None,
        )
        series = FakeSeries()

        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_openpmd_fields(tiled_fields, static_parameters, dynamic_parameters, "/tmp", plot_t=0, t=0)

        E_mesh = series.iterations[0].meshes["E"]
        rho_mesh = series.iterations[0].meshes["rho"]
        self.assertEqual(E_mesh.records["x"].shape, (4, 2, 2))
        self.assertEqual(rho_mesh.records[openPMD.io.Mesh_Record_Component.SCALAR].shape, (4, 2, 2))

    def test_tiled_field_snapshot_writes_tile_chunks_without_global_assembly(self):
        parameter_values = _parameter_values()
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_values)
        shape_with_ghosts = (6, 4, 3)
        rho_global = jnp.arange(6 * 4 * 3, dtype=jnp.float64).reshape(shape_with_ghosts)
        phi_global = rho_global + 100.0
        E_global = tuple(rho_global + offset for offset in (10.0, 20.0, 30.0))
        tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
        field_map = {
            "E": tile_vector_field(E_global, static_parameters, dynamic_parameters),
            "rho": tile_scalar_field(rho_global, static_parameters, dynamic_parameters),
            "phi": tile_scalar_field(phi_global, static_parameters, dynamic_parameters),
        }
        snapshot = async_writer.make_tiled_field_snapshot(
            field_map,
            step=3,
            time=0.6,
        )
        layout = openPMD.TiledMeshLayout(
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tile_shape,
            guard_cells=int(static_parameters.guard_cells),
        )
        series = FakeSeries()

        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_tiled_field_snapshot_openpmd(
                snapshot,
                output_dir="/tmp",
                filename="fields",
                dynamic_parameters=dynamic_parameters,
                layout=layout,
                file_extension=".h5",
            )

        iteration = series.iterations[3]
        rho_record = iteration.meshes["rho"].records[openPMD.io.Mesh_Record_Component.SCALAR]
        E_record = iteration.meshes["E"].records["x"]

        self.assertEqual(iteration.time, 0.6)
        self.assertEqual(iteration.dt, float(dynamic_parameters.dt))
        self.assertEqual(iteration.meshes["rho"].axis_labels, ["x", "y", "z"])
        self.assertEqual(len(rho_record.chunks), 4)
        self.assertEqual(rho_record.chunks[0][0], (0, 0, 0))
        self.assertEqual(rho_record.chunks[0][1], tile_shape)
        self.assertEqual(len(E_record.chunks), 4)
        self.assertEqual(E_record.chunks[0][0], (0, 0, 0))

    def test_async_field_writer_queue_size_caps_pending_snapshots(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(_parameter_values())
        writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
            guard_cells=int(static_parameters.guard_cells),
            queue_size=1,
        )
        snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})

        self.assertTrue(writer.enqueue(snapshot, block=False))
        self.assertFalse(writer.enqueue(snapshot, block=False))
        writer.close(raise_errors=False)

    def test_async_field_writer_raises_worker_errors_on_close(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(_parameter_values())
        writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
            guard_cells=int(static_parameters.guard_cells),
            queue_size=1,
        )
        snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})

        with patch.object(async_writer, "write_tiled_field_snapshot_openpmd", side_effect=RuntimeError("disk failed")):
            writer.start()
            self.assertTrue(writer.enqueue(snapshot))
            with self.assertRaisesRegex(RuntimeError, "Async openPMD writer failed"):
                writer.close()

    def test_tiled_particle_snapshot_writes_same_records_as_synchronous_output(self):
        parameter_values = _parameter_values()
        dynamic_values = {"C": 10.0}
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            parameter_values,
            dynamic_values=dynamic_values,
        )
        species = [
            _species("beam electrons", -1.0, 2.0, 3.0, jnp.array([-1.5, 0.5])),
            _species("background ions", 1.0, 4.0, 5.0, jnp.array([1.5, -0.5])),
        ]
        tiled_particles, species_config = build_tiled_particles(species, static_parameters, dynamic_parameters)
        species_names = names_for_species(species)

        sync_series = FakeSeries()
        with patch.object(openPMD, "_open_openpmd_series", return_value=sync_series):
            openPMD.write_openpmd_particles(
                tiled_particles,
                static_parameters,
                dynamic_parameters,
                "/tmp",
                plot_t=2,
                t=3,
                species_config=species_config,
                species_names=species_names,
            )

        snapshot = async_writer.make_tiled_particle_snapshot(
            tiled_particles,
            step=2,
            time=3 * float(dynamic_parameters.dt),
            species_names=species_names,
            species_config=species_config,
        )
        async_series = FakeSeries()
        with patch.object(openPMD, "_open_openpmd_series", return_value=async_series):
            openPMD.write_tiled_particle_snapshot_openpmd(
                snapshot,
                output_dir="/tmp",
                filename="particles",
                static_parameters=static_parameters,
                dynamic_parameters=dynamic_parameters,
                file_extension=".h5",
            )

        sync_electrons = sync_series.iterations[2].particles["beam_electrons"]
        async_electrons = async_series.iterations[2].particles["beam_electrons"]
        sync_ions = sync_series.iterations[2].particles["background_ions"]
        async_ions = async_series.iterations[2].particles["background_ions"]

        self.assertEqual(async_series.iterations[2].time, 3 * float(dynamic_parameters.dt))
        self.assertEqual(_record_data(async_electrons["position"]["x"]).shape, _record_data(sync_electrons["position"]["x"]).shape)
        self.assertEqual(_record_data(async_ions["position"]["x"]).shape, _record_data(sync_ions["position"]["x"]).shape)
        for group_async, group_sync in ((async_electrons, sync_electrons), (async_ions, sync_ions)):
            for record_name in ("position", "positionOffset", "momentum"):
                for component in ("x", "y", "z"):
                    self.assertTrue(
                        jnp.allclose(
                            _record_data(group_async[record_name][component]),
                            _record_data(group_sync[record_name][component]),
                        )
                    )
            for record_name in ("weighting", "charge", "mass"):
                self.assertTrue(
                    jnp.allclose(
                        _record_data(group_async[record_name]),
                        _record_data(group_sync[record_name]),
                    )
                )

    def test_async_particle_writer_queue_size_caps_pending_snapshots(self):
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            queue_size=1,
        )
        snapshot = async_writer.TiledParticleSnapshot(
            step=0,
            time=0.0,
            species_names=("electrons",),
            x_shards=[],
            u_shards=[],
            active_shards=[],
            species_charge=jnp.array([-1.0]),
            species_mass=jnp.array([1.0]),
            species_weight=jnp.array([1.0]),
        )

        self.assertTrue(writer.enqueue(snapshot, block=False))
        self.assertFalse(writer.enqueue(snapshot, block=False))
        writer.close(raise_errors=False)

    def test_async_particle_writer_raises_worker_errors_on_close(self):
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            queue_size=1,
        )
        snapshot = async_writer.TiledParticleSnapshot(
            step=0,
            time=0.0,
            species_names=("electrons",),
            x_shards=[],
            u_shards=[],
            active_shards=[],
            species_charge=jnp.array([-1.0]),
            species_mass=jnp.array([1.0]),
            species_weight=jnp.array([1.0]),
        )

        with patch.object(async_writer, "write_tiled_particle_snapshot_openpmd", side_effect=RuntimeError("disk failed")):
            writer.start()
            self.assertTrue(writer.enqueue(snapshot))
            with self.assertRaisesRegex(RuntimeError, "Async openPMD particle writer failed"):
                writer.close()

    def test_field_and_particle_async_writers_serialize_openpmd_calls(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        field_writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
            guard_cells=int(static_parameters.guard_cells),
            queue_size=1,
        )
        particle_writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            queue_size=1,
        )
        field_snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})
        particle_snapshot = async_writer.TiledParticleSnapshot(
            step=0,
            time=0.0,
            species_names=("electrons",),
            x_shards=[],
            u_shards=[],
            active_shards=[],
            species_charge=jnp.array([-1.0]),
            species_mass=jnp.array([1.0]),
            species_weight=jnp.array([1.0]),
        )

        active_writes = 0
        overlaps = []
        counter_lock = threading.Lock()
        field_write_started = threading.Event()

        def write_snapshot(*args, **kwargs):
            nonlocal active_writes
            with counter_lock:
                active_writes += 1
                if active_writes > 1:
                    overlaps.append(active_writes)
            field_write_started.set()
            time.sleep(0.05)
            with counter_lock:
                active_writes -= 1

        with (
            patch.object(async_writer, "write_tiled_field_snapshot_openpmd", side_effect=write_snapshot),
            patch.object(async_writer, "write_tiled_particle_snapshot_openpmd", side_effect=write_snapshot),
        ):
            field_writer.start()
            particle_writer.start()
            self.assertTrue(field_writer.enqueue(field_snapshot))
            self.assertTrue(field_write_started.wait(timeout=1.0))
            self.assertTrue(particle_writer.enqueue(particle_snapshot))
            field_writer.close()
            particle_writer.close()

        self.assertEqual(overlaps, [])

    def test_write_openpmd_initial_particles_flattens_tiled_particles_and_preserves_names(self):
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        species = [
            _species("beam electrons", -1.0, 2.0, 3.0, jnp.array([-1.5, 0.5])),
            _species("background ions", 1.0, 4.0, 5.0, jnp.array([1.5])),
        ]
        tiled_particles, species_config = build_tiled_particles(species, static_parameters, dynamic_parameters)
        species_names = names_for_species(species)
        series = FakeSeries()

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_particles(
                tiled_particles,
                static_parameters,
                dynamic_parameters,
                "/tmp",
                species_config=species_config,
                species_names=species_names,
            )

        iteration = series.iterations[0]
        electron_group = iteration.particles["beam_electrons"]
        ion_group = iteration.particles["background_ions"]
        self.assertEqual(electron_group["position"]["x"].shape, (2,))
        self.assertEqual(ion_group["position"]["x"].shape, (1,))
        self.assertTrue(jnp.allclose(electron_group["weighting"].data, jnp.array([3.0, 3.0])))
        self.assertTrue(jnp.allclose(ion_group["charge"].data, jnp.array([1.0])))
        expected_gamma = 1.0 / jnp.sqrt(1.0 - 0.1**2 / 10.0**2)
        self.assertTrue(jnp.allclose(electron_group["momentum"]["x"].data, jnp.ones(2) * 0.1 * 6.0 * expected_gamma))


if __name__ == "__main__":
    unittest.main()
