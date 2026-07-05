import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.diagnostics import openPMD
from PyPIC3D.diagnostics.openPMD import _ensure_openpmd_array
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles


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

    if g != 1:
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
        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    ix = 1 + tx * tile_nx
                    iy = 1 + ty * tile_ny
                    iz = 1 + tz * tile_nz
                    interior = field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                    field_tiles = field_tiles.at[tx, ty, tz, g:-g, g:-g, g:-g].set(interior)
        return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g, tile_shape)

    def tile_at(tx, ty, tz):
        start = (tx * tile_nx, ty * tile_ny, tz * tile_nz)
        size = (tile_nx + 2, tile_ny + 2, tile_nz + 2)
        return jax.lax.dynamic_slice(field, start, size)

    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack([tile_at(tx, ty, tz) for tz in range(ntz)], axis=0)
                    for ty in range(nty)
                ],
                axis=0,
            )
            for tx in range(ntx)
        ],
        axis=0,
    )


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


class FakeRecord:
    def __init__(self):
        self.shape = None
        self.unit_SI = None
        self.data = None

    def reset_dataset(self, dataset):
        pass

    def store_chunk(self, array, offset, extent):
        self.shape = tuple(extent)
        self.data = jnp.asarray(array)


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


def _simulation_parameters():
    return {
        "particle_tile_nx": 2,
        "particle_tile_ny": 1,
        "particle_tile_nz": 1,
    }


def _species(name, charge, mass, weight, x1):
    world = _world()
    return particle_species(
        name=name,
        N_particles=x1.shape[0],
        charge=charge,
        mass=mass,
        weight=weight,
        T=0.0,
        x1=x1,
        x2=jnp.zeros_like(x1),
        x3=jnp.zeros_like(x1),
        v1=jnp.ones_like(x1) * 0.1,
        v2=jnp.zeros_like(x1),
        v3=jnp.zeros_like(x1),
        xwind=world["x_wind"],
        ywind=world["y_wind"],
        zwind=world["z_wind"],
        dx=world["dx"],
        dy=world["dy"],
        dz=world["dz"],
        active_mask=jnp.ones(x1.shape[0], dtype=bool),
        dt=world["dt"],
    )


class OpenPMDDiagnosticsTests(unittest.TestCase):

    def test_openpmd_field_array_preserves_thin_y_axis(self):
        field_component = jnp.ones((4, 1, 6))

        array = _ensure_openpmd_array(field_component)

        self.assertEqual(array.shape, (4, 1, 6))

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

    def test_write_openpmd_initial_particles_flattens_tiled_particles_and_preserves_names(self):
        world = _world()
        species = [
            _species("beam electrons", -1.0, 2.0, 3.0, jnp.array([-1.5, 0.5])),
            _species("background ions", 1.0, 4.0, 5.0, jnp.array([1.5])),
        ]
        tiled_particles, species_config = to_tiled_particles(species, world, _simulation_parameters())
        species_names = tuple(s.get_name() for s in species)
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
