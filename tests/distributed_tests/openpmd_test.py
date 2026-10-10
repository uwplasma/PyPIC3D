"""Focused cross-device regression tests."""

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
    def test_initial_fields_assemble_tiled_fields_before_output(self):
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


    def test_static_metric_snapshot_writes_covariant_gamma_momentum_and_stored_position(self):
        static, dynamic = kernel_parameters(
            Nx=8, Ny=8, Nz=1, x_min=1., y_min=.5, z_min=0.,
            x_wind=1., y_wind=1., z_wind=1., dt=.005,
            solver='static_metric', particle_pusher='hybrid_boris_geodesic',
            metric='numerical', tile_shape=(4, 8, 1),
        )
        gamma_metric = jnp.array([[2., .2, .1], [.2, 3., -.1], [.1, -.1, 1.5]])
        # a uniform metric is reproduced exactly by the Hermite interpolant

        def provider(position):
            return 1., jnp.zeros(3), gamma_metric, jnp.linalg.inv(gamma_metric), jnp.sqrt(jnp.linalg.det(gamma_metric))

        metric = build_yee_metric(dynamic, provider)
        # one live particle in each of the two x tiles, plus an inactive slot
        x = jnp.array([[[1.2, .8, 0.], [1.3, .9, 0.]], [[1.7, 1.1, 0.], [0., 0., 0.]]]).reshape(2, 1, 1, 1, 2, 3)
        u = jnp.array([[[.6, .2, -.1], [0., 0., 0.]], [[-2., 1., 3.], [5., 5., 5.]]]).reshape(2, 1, 1, 1, 2, 3)
        active = jnp.array([[True, False], [True, False]]).reshape(2, 1, 1, 1, 2)
        particles = TiledParticles(x, u, active)
        species = SpeciesConfig(jnp.array([-1.]), jnp.array([2.]), jnp.array([3.]), jnp.ones((1, 3), bool))

        gamma = particle_lorentz_factor(particles, metric, static, dynamic)
        snapshot = async_writer.make_tiled_particle_snapshot(
            particles, step=0, time=0., species_names=('electrons',), species_config=species, gamma=gamma)
        iteration = FakeIteration()
        openPMD.write_tiled_particle_snapshot_to_iteration(iteration, snapshot, static, dynamic)

        group = iteration.particles['electrons']
        x_live = np.array([[1.2, .8, 0.], [1.7, 1.1, 0.]])
        u_live = np.array([[.6, .2, -.1], [-2., 1., 3.]])
        expected_gamma = np.sqrt(1. + np.einsum('ni,ij,nj->n', u_live, np.linalg.inv(gamma_metric), u_live))
        written_gamma = np.concatenate([data for _, _, data in group['gamma'].chunks])
        np.testing.assert_allclose(written_gamma, expected_gamma, rtol=1e-6)
        self.assertTrue(np.all(written_gamma > 1.))
        for axis, component in enumerate(('x', 'y', 'z')):
            momentum = np.concatenate([data for _, _, data in group['momentum'][component].chunks])
            position = np.concatenate([data for _, _, data in group['position'][component].chunks])
            np.testing.assert_allclose(momentum, 2. * u_live[:, axis], rtol=1e-6)
            np.testing.assert_allclose(position, x_live[:, axis], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
