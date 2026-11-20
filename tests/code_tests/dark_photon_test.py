"""One-tile API contracts."""

from tests.support.dark_photon_fixtures import (
    DarkPhotonFixtures,
    dark_photon_courant_condition,
    build_field_output_map,
    build_static_parameters,
    compute_dark_energy,
    contextlib,
    default_parameters,
    field_map_for_output,
    initialize_dark_photon_fields,
    initialize_simulation,
    io,
    jnp,
    load_dark_fields_from_toml,
    math,
    np,
    os,
    parameters,
    pmd,
    run_PyPIC3D,
    static_parameters_for_output,
    tempfile,
    time_loop_dark_photon,
    toml,
    unittest,
)


class TestDarkPhoton(DarkPhotonFixtures, unittest.TestCase):
    def test_toml_initialization_seed_output_energy_and_driver(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = self.config(directory)
            config["plotting"] = dict(dump_fields=True, plot_openpmd_fields=True)
            for field_type, amplitude in ((0, .25), (3, .5), (6, .1)):
                path = os.path.join(directory, f"initial_{field_type}.npy")
                np.save(path, np.full((8, 1, 1), amplitude))
                config[f"dark_field{field_type}"] = dict(name="dark", type=field_type, path=path)
            config = toml.loads(toml.dumps(config))
            loop, particles, fields, s, d, plotting, _, species = initialize_simulation(config)
            self.assertIs(loop, time_loop_dark_photon)
            self.assertEqual(len(fields), 9)
            self.assertEqual(s.sin_chi, .2)
            self.assertEqual(s.dark_mu, .7)
            self.assertEqual(static_parameters_for_output(s)["dark_mu"], .7)
            self.assertEqual(build_static_parameters(s._asdict()), s)
            self.assertEqual(toml.loads(toml.dumps(static_parameters_for_output(s)))["sin_chi"], .2)
            E, A, phi = fields[7]
            np.testing.assert_allclose(A[0], .5+.25*float(d.dt)/2, atol=1e-15)
            snapshot = build_field_output_map(fields, particles, species, s, d)
            np.testing.assert_allclose(snapshot["dark_A"][0], .5, atol=1e-15)
            self.assertEqual(set(snapshot), {"E", "B", "J", "dark_E", "dark_A", "dark_phi", "dark_B"})
            np.testing.assert_allclose(field_map_for_output(snapshot, s)["dark_A"][0], .5, atol=1e-15)
            expected_energy = .5*2*(2*math.pi)*(.25**2+1.3**2*.7**2*.5**2+.7**2*.1**2)
            self.assertAlmostEqual(float(compute_dark_energy(fields[7], s, d)), expected_energy, places=12)
            config["plotting"]["dump_fields"] = False
            result = run_PyPIC3D(config)
            self.assertEqual(len(result[5]), 9)
            recorded = np.loadtxt(os.path.join(directory, "data", "total_energy.txt"), delimiter=",")
            self.assertAlmostEqual(float(np.ravel(recorded)[1]), expected_energy, places=10)
            for filename in (os.path.join("initial_fields", "initial_fields.h5"), "fields.h5"):
                series = pmd.Series(os.path.join(directory, "data", filename), pmd.Access.read_only)
                iteration = series.iterations[0]
                for name in ("dark_E", "dark_A", "dark_phi", "dark_B"):
                    self.assertIn(name, iteration.meshes)
                    self.assertEqual(iteration.meshes[name].time_offset, 0.)
                np.testing.assert_allclose(iteration.meshes["dark_A"]["x"].position, [.5, 0., 0.])
                np.testing.assert_allclose(iteration.meshes["dark_B"]["x"].position, [0., .5, .5])
                values = iteration.meshes["dark_A"]["x"].load_chunk()
                series.flush()
                np.testing.assert_allclose(values, .5, atol=1e-15)
                series.close()

    def test_configuration_errors_and_timestep_bound(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for updates in (dict(sin_chi=1.1), dict(sin_chi=float("nan")), dict(dark_mu=-1.),
                            dict(dark_mu=float("inf")), dict(y_bc="constant"),
                            dict(dt=-1.), dict(dt=float("nan")), dict(dt=100.), dict(C=0.), dict(eps=0.)):
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    initialize_simulation(self.config(directory, **updates))
            for absorber in ("supergaussian",):
                config = self.config(directory)
                config[absorber] = [{"axis": "x"}]
                with self.assertRaisesRegex(ValueError, "does not support"):
                    initialize_simulation(config)
            for updates in (dict(sin_chi=2.), dict(dark_mu=-1.),
                            dict(supergaussian_active=True), dict(boundary_conditions=(2, 0, 0))):
                s, _ = parameters()
                with self.assertRaises(ValueError):
                    build_static_parameters({**s._asdict(), **updates})
        _, static, dynamic = default_parameters()
        dynamic.update(Nx=8, Ny=1, Nz=1, dx=.2, dy=1., dz=1., C=1.3, dt=None)
        static.update(dark_mu=20., sin_chi=.1)
        limit = 2/(1.3*math.sqrt(20**2+4/.2**2))
        self.assertAlmostEqual(dark_photon_courant_condition(static, dynamic), .99*limit)
        dynamic["dt"] = limit
        with self.assertRaisesRegex(ValueError, "stability limit"):
            dark_photon_courant_condition(static, dynamic)
        dynamic.update(Nx=1, dt=None)
        self.assertAlmostEqual(dark_photon_courant_condition(static, dynamic), .99*2/(1.3*20))
        static["dark_mu"] = 0.
        with self.assertRaisesRegex(ValueError, "Specify dt"):
            dark_photon_courant_condition(static, dynamic)
        dynamic["dt"] = .1
        self.assertEqual(dark_photon_courant_condition(static, dynamic), .1)

    def test_invalid_initial_field_shapes_and_types(self):
        s, d = parameters()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "field.npy")
            for values in (np.zeros((3, 1, 1)), np.full((32, 1, 1), np.nan),
                           np.ones((32, 1, 1), dtype=complex)):
                np.save(path, values)
                with self.assertRaises(ValueError):
                    load_dark_fields_from_toml({"dark_field0": dict(type=0, path=path)}, s, d)
            np.save(path, np.zeros((32, 1, 1)))
            for field_type in (-1, 7, True, 1.5):
                with self.assertRaises(ValueError):
                    load_dark_fields_from_toml({"dark_field0": dict(type=field_type, path=path)}, s, d)
            with self.assertRaisesRegex(ValueError, "evolved"):
                load_dark_fields_from_toml({"dark_field0": dict(type=0, path=path, evolve=False)}, s, d)
        with self.assertRaisesRegex(ValueError, "shape"):
            initialize_dark_photon_fields(s, d, phi=jnp.zeros((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, "three components"):
            initialize_dark_photon_fields(s, d, E=(jnp.zeros(1),))


if __name__ == "__main__":
    unittest.main()
