import unittest
from unittest.mock import patch

import jax.numpy as jnp

from PyPIC3D.diagnostics import openPMD
from PyPIC3D.diagnostics.openPMD import _ensure_openpmd_array
from PyPIC3D.solvers.yee_tiled import tile_scalar_field, tile_vector_field


class FakeRecord:
    def __init__(self):
        self.shape = None
        self.unit_SI = None

    def reset_dataset(self, dataset):
        pass

    def store_chunk(self, array, offset, extent):
        self.shape = tuple(extent)


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


class FakeIteration:
    def __init__(self):
        self.meshes = FakeMeshes()
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

    def flush(self):
        pass

    def close(self):
        pass


def _zero_field(shape):
    return tuple(jnp.zeros(shape) for _ in range(3))


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


if __name__ == "__main__":
    unittest.main()
