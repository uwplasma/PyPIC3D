import unittest
import math
import threading
import time
from unittest.mock import patch

import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.diagnostics import async_writer
from PyPIC3D.diagnostics import openPMD
from PyPIC3D.diagnostics.openPMD import _ensure_openpmd_array
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, world, tile_shape, num_guard_cells=2):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

    interior_tiles = field[1:-1, 1:-1, 1:-1]
    interior_tiles = interior_tiles.reshape(ntx, tile_nx, nty, tile_ny, ntz, tile_nz)
    interior_tiles = interior_tiles.transpose(0, 2, 4, 1, 3, 5)

    field_tiles = jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=field.dtype,
    )
    field_tiles = field_tiles.at[:, :, :, g:-g, g:-g, g:-g].set(interior_tiles)

    return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g, tile_shape)


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


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


def _world():
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
    return {
        "name": name,
        "charge": charge,
        "mass": mass,
        "weight": weight,
        "x1": jnp.asarray(x1),
        "u1": jnp.ones_like(x1) * 0.1,
    }


def _particle_tile_index(x, y, z, world, tile_shape):
    tile_nx, tile_ny, tile_nz = tile_shape

    x_cell = math.floor((float(x) + float(world["x_wind"]) / 2.0) / float(world["dx"]))
    y_cell = math.floor((float(y) + float(world["y_wind"]) / 2.0) / float(world["dy"]))
    z_cell = math.floor((float(z) + float(world["z_wind"]) / 2.0) / float(world["dz"]))

    x_cell = min(max(x_cell, 0), int(world["Nx"]) - 1)
    y_cell = min(max(y_cell, 0), int(world["Ny"]) - 1)
    z_cell = min(max(z_cell, 0), int(world["Nz"]) - 1)

    return x_cell // tile_nx, y_cell // tile_ny, z_cell // tile_nz


def _make_tiled_particles(species, world):
    tile_shape = tuple(int(width) for width in world["tile_shape"])
    tile_nx, tile_ny, tile_nz = tile_shape
    ntx = _tile_axis_count(world["Nx"], tile_nx)
    nty = _tile_axis_count(world["Ny"], tile_ny)
    ntz = _tile_axis_count(world["Nz"], tile_nz)

    placements = []
    tile_counts = {}

    for species_index, species_data in enumerate(species):
        x1 = species_data["x1"]
        u1 = species_data["u1"]
        zeros = jnp.zeros_like(x1)
        x = jnp.stack((x1, zeros, zeros), axis=-1)
        u = jnp.stack((u1, zeros, zeros), axis=-1)

        for particle_index in range(x1.shape[0]):
            tx, ty, tz = _particle_tile_index(
                x[particle_index, 0],
                x[particle_index, 1],
                x[particle_index, 2],
                world,
                tile_shape,
            )
            tile_key = (tx, ty, tz, species_index)
            tile_counts[tile_key] = tile_counts.get(tile_key, 0) + 1
            placements.append((tile_key, x[particle_index], u[particle_index]))

    max_particles_per_tile = max(1, max(tile_counts.values()))
    n_species = len(species)

    x_tiles = jnp.zeros((ntx, nty, ntz, n_species, max_particles_per_tile, 3))
    u_tiles = jnp.zeros_like(x_tiles)
    active_tiles = jnp.zeros((ntx, nty, ntz, n_species, max_particles_per_tile), dtype=bool)
    write_counts = {}

    for tile_key, x_particle, u_particle in placements:
        slot = write_counts.get(tile_key, 0)
        write_counts[tile_key] = slot + 1

        x_tiles = x_tiles.at[tile_key + (slot, slice(None))].set(x_particle)
        u_tiles = u_tiles.at[tile_key + (slot, slice(None))].set(u_particle)
        active_tiles = active_tiles.at[tile_key + (slot,)].set(True)

    particles = TiledParticles(x=x_tiles, u=u_tiles, active=active_tiles)
    species_config = SpeciesConfig(
        charge=jnp.asarray([species_data["charge"] for species_data in species]),
        mass=jnp.asarray([species_data["mass"] for species_data in species]),
        weight=jnp.asarray([species_data["weight"] for species_data in species]),
        update_x=jnp.ones((n_species, 3), dtype=bool),
        update_u=jnp.ones((n_species, 3), dtype=bool),
    )
    species_names = tuple(species_data["name"] for species_data in species)

    return particles, species_config, species_names


class OpenPMDDiagnosticsTests(unittest.TestCase):

    def test_openpmd_field_array_preserves_thin_y_axis(self):
        field_component = jnp.ones((4, 1, 6))

        array = _ensure_openpmd_array(field_component)

        self.assertEqual(array.shape, (4, 1, 6))

    def _expected_tiled_scalar_from_interior(self, field, world, num_guard_cells):
        tile_shape = world["tile_shape"]
        tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
        g = int(num_guard_cells)
        Nx = int(field.shape[0]) - 2
        Ny = int(field.shape[1]) - 2
        Nz = int(field.shape[2]) - 2
        ntx = _tile_axis_count(Nx, tile_nx)
        nty = _tile_axis_count(Ny, tile_ny)
        ntz = _tile_axis_count(Nz, tile_nz)

        interior_tiles = field[1:-1, 1:-1, 1:-1]
        interior_tiles = interior_tiles.reshape(ntx, tile_nx, nty, tile_ny, ntz, tile_nz)
        interior_tiles = interior_tiles.transpose(0, 2, 4, 1, 3, 5)

        field_tiles = jnp.zeros(
            (
                ntx,
                nty,
                ntz,
                tile_nx + 2 * g,
                tile_ny + 2 * g,
                tile_nz + 2 * g,
            ),
            dtype=field.dtype,
        )
        field_tiles = field_tiles.at[:, :, :, g:-g, g:-g, g:-g].set(interior_tiles)

        return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g, tile_shape)

    def _field_with_stale_global_ghosts(self, world, value_offset=0.0):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
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
        world = _world()
        field = self._field_with_stale_global_ghosts(world)

        tiles = tile_scalar_field(field, world, world["tile_shape"], num_guard_cells=1)
        expected = self._expected_tiled_scalar_from_interior(field, world, num_guard_cells=1)

        self.assertTrue(jnp.allclose(tiles, expected))

    def test_tile_scalar_field_two_guards_rebuilds_periodic_halos_from_interiors(self):
        world = _world()
        field = self._field_with_stale_global_ghosts(world, value_offset=10.0)

        tiles = tile_scalar_field(field, world, world["tile_shape"], num_guard_cells=2)
        expected = self._expected_tiled_scalar_from_interior(field, world, num_guard_cells=2)

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
        world = {
            "dt": 1.0,
            "dx": 0.25,
            "dy": 0.5,
            "dz": 0.75,
            "x_wind": 1.0,
            "y_wind": 2.0,
            "z_wind": 3.0,
        }
        series = FakeSeries()

        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_openpmd_fields(fields, world, "/tmp", plot_t=0, t=0)

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
        world = {
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
        tiled_fields = (
            tile_vector_field(E, world, world["tile_shape"]),
            tile_vector_field(B, world, world["tile_shape"]),
            tile_vector_field(J, world, world["tile_shape"]),
            tile_scalar_field(rho, world, world["tile_shape"]),
            tile_scalar_field(phi, world, world["tile_shape"]),
            (
                tile_vector_field(external_fields[0], world, world["tile_shape"]),
                tile_vector_field(external_fields[1], world, world["tile_shape"]),
            ),
            None,
        )
        series = FakeSeries()

        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_openpmd_fields(tiled_fields, world, "/tmp", plot_t=0, t=0)

        E_mesh = series.iterations[0].meshes["E"]
        rho_mesh = series.iterations[0].meshes["rho"]
        self.assertEqual(E_mesh.records["x"].shape, (4, 2, 2))
        self.assertEqual(rho_mesh.records[openPMD.io.Mesh_Record_Component.SCALAR].shape, (4, 2, 2))

    def test_tiled_field_snapshot_writes_tile_chunks_without_global_assembly(self):
        world = _world()
        shape_with_ghosts = (6, 4, 3)
        rho_global = jnp.arange(6 * 4 * 3, dtype=jnp.float64).reshape(shape_with_ghosts)
        phi_global = rho_global + 100.0
        E_global = tuple(rho_global + offset for offset in (10.0, 20.0, 30.0))
        tile_shape = world["tile_shape"]
        field_map = {
            "E": tile_vector_field(E_global, world, tile_shape),
            "rho": tile_scalar_field(rho_global, world, tile_shape),
            "phi": tile_scalar_field(phi_global, world, tile_shape),
        }
        snapshot = async_writer.make_tiled_field_snapshot(
            field_map,
            step=3,
            time=0.6,
        )
        layout = openPMD.TiledMeshLayout(
            global_shape=(world["Nx"], world["Ny"], world["Nz"]),
            tile_shape=tile_shape,
            guard_cells=world["guard_cells"],
        )
        series = FakeSeries()

        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_tiled_field_snapshot_openpmd(
                snapshot,
                output_dir="/tmp",
                filename="fields",
                world=world,
                layout=layout,
                file_extension=".h5",
            )

        iteration = series.iterations[3]
        rho_record = iteration.meshes["rho"].records[openPMD.io.Mesh_Record_Component.SCALAR]
        E_record = iteration.meshes["E"].records["x"]

        self.assertEqual(iteration.time, 0.6)
        self.assertEqual(iteration.dt, world["dt"])
        self.assertEqual(iteration.meshes["rho"].axis_labels, ["x", "y", "z"])
        self.assertEqual(len(rho_record.chunks), 4)
        self.assertEqual(rho_record.chunks[0][0], (0, 0, 0))
        self.assertEqual(rho_record.chunks[0][1], tile_shape)
        self.assertEqual(len(E_record.chunks), 4)
        self.assertEqual(E_record.chunks[0][0], (0, 0, 0))

    def test_async_field_writer_queue_size_caps_pending_snapshots(self):
        world = _world()
        writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            world=world,
            global_shape=(world["Nx"], world["Ny"], world["Nz"]),
            tile_shape=world["tile_shape"],
            guard_cells=world["guard_cells"],
            queue_size=1,
        )
        snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})

        self.assertTrue(writer.enqueue(snapshot, block=False))
        self.assertFalse(writer.enqueue(snapshot, block=False))
        writer.close(raise_errors=False)

    def test_async_field_writer_raises_worker_errors_on_close(self):
        world = _world()
        writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            world=world,
            global_shape=(world["Nx"], world["Ny"], world["Nz"]),
            tile_shape=world["tile_shape"],
            guard_cells=world["guard_cells"],
            queue_size=1,
        )
        snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})

        with patch.object(async_writer, "write_tiled_field_snapshot_openpmd", side_effect=RuntimeError("disk failed")):
            writer.start()
            self.assertTrue(writer.enqueue(snapshot))
            with self.assertRaisesRegex(RuntimeError, "Async openPMD writer failed"):
                writer.close()

    def test_tiled_particle_snapshot_writes_same_records_as_synchronous_output(self):
        world = _world()
        species = [
            _species("beam electrons", -1.0, 2.0, 3.0, jnp.array([-1.5, 0.5])),
            _species("background ions", 1.0, 4.0, 5.0, jnp.array([1.5, -0.5])),
        ]
        tiled_particles, species_config, species_names = _make_tiled_particles(species, world)
        constants = {"C": 10.0}

        sync_series = FakeSeries()
        with patch.object(openPMD, "_open_openpmd_series", return_value=sync_series):
            openPMD.write_openpmd_particles(
                tiled_particles,
                world,
                constants,
                "/tmp",
                plot_t=2,
                t=3,
                species_config=species_config,
                species_names=species_names,
            )

        snapshot = async_writer.make_tiled_particle_snapshot(
            tiled_particles,
            step=2,
            time=3 * world["dt"],
            species_names=species_names,
            species_config=species_config,
        )
        async_series = FakeSeries()
        with patch.object(openPMD, "_open_openpmd_series", return_value=async_series):
            openPMD.write_tiled_particle_snapshot_openpmd(
                snapshot,
                output_dir="/tmp",
                filename="particles",
                world=world,
                constants=constants,
                file_extension=".h5",
            )

        sync_electrons = sync_series.iterations[2].particles["beam_electrons"]
        async_electrons = async_series.iterations[2].particles["beam_electrons"]
        sync_ions = sync_series.iterations[2].particles["background_ions"]
        async_ions = async_series.iterations[2].particles["background_ions"]

        self.assertEqual(async_series.iterations[2].time, 3 * world["dt"])
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
        world = _world()
        writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            world=world,
            constants={"C": 10.0},
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
        world = _world()
        writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            world=world,
            constants={"C": 10.0},
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
        world = _world()
        field_writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            world=world,
            global_shape=(world["Nx"], world["Ny"], world["Nz"]),
            tile_shape=world["tile_shape"],
            guard_cells=world["guard_cells"],
            queue_size=1,
        )
        particle_writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            world=world,
            constants={"C": 10.0},
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
        world = _world()
        species = [
            _species("beam electrons", -1.0, 2.0, 3.0, jnp.array([-1.5, 0.5])),
            _species("background ions", 1.0, 4.0, 5.0, jnp.array([1.5])),
        ]
        tiled_particles, species_config, species_names = _make_tiled_particles(species, world)
        series = FakeSeries()

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_particles(
                tiled_particles,
                world,
                {"C": 10.0},
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
