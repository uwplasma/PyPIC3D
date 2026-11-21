"""Focused cross-device regression tests."""

from tests.support.openpmd_fixtures import (
    SimpleNamespace,
    jax,
    particle_parameters_from_values,
)
from tests.support.openpmd_fixtures import (
    FakeIteration,
    FakeSeries,
    OpenPMDDiagnosticsTestsFixtures,
    SpeciesConfig,
    TiledParticles,
    _parameter_values,
    _zero_field,
    async_writer,
    build_yee_metric,
    field_tiles_from_global,
    jnp,
    kernel_parameters,
    kernel_parameters_from_values,
    np,
    openPMD,
    particle_lorentz_factor,
    patch,
    unittest,
    vector_tiles_from_global,
)


class OpenPMDDiagnosticsTests(OpenPMDDiagnosticsTestsFixtures, unittest.TestCase):
    def test_initial_fields_write_tile_chunks_into_the_same_global_mesh(self):
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
            vector_tiles_from_global(E, static_parameters, dynamic_parameters),
            vector_tiles_from_global(B, static_parameters, dynamic_parameters),
            vector_tiles_from_global(J, static_parameters, dynamic_parameters),
            field_tiles_from_global(rho, static_parameters, dynamic_parameters),
            field_tiles_from_global(phi, static_parameters, dynamic_parameters),
            (
                vector_tiles_from_global(external_fields[0], static_parameters, dynamic_parameters),
                vector_tiles_from_global(external_fields[1], static_parameters, dynamic_parameters),
            ),
            None,
        )
        series = FakeSeries()
        field_map = {
            "E": tiled_fields[0],
            "rho": tiled_fields[3],
        }

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_fields(field_map, static_parameters, dynamic_parameters, "/tmp")

        E_mesh = series.iterations[0].meshes["E"]
        rho_mesh = series.iterations[0].meshes["rho"]
        self.assertEqual(E_mesh.records["x"].shape, (4, 2, 2))
        self.assertEqual(rho_mesh.records[openPMD.io.Mesh_Record_Component.SCALAR].shape, (4, 2, 2))


    def test_tiled_field_snapshot_writes_tile_chunks_without_global_assembly(self):
        parameter_values = _parameter_values(tile_shape=(2, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_values)
        shape_with_ghosts = (6, 4, 3)
        rho_global = jnp.arange(6 * 4 * 3, dtype=jnp.float64).reshape(shape_with_ghosts)
        phi_global = rho_global + 100.0
        E_global = tuple(rho_global + offset for offset in (10.0, 20.0, 30.0))
        tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
        field_map = {
            "E": vector_tiles_from_global(E_global, static_parameters, dynamic_parameters),
            "rho": field_tiles_from_global(rho_global, static_parameters, dynamic_parameters),
            "phi": field_tiles_from_global(phi_global, static_parameters, dynamic_parameters),
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
        dimensions = openPMD.io.Unit_Dimension
        self.assertEqual(
            iteration.meshes["E"].unit_dimension,
            {
                dimensions.L: 1.0,
                dimensions.M: 1.0,
                dimensions.T: -3.0,
                dimensions.I: -1.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
