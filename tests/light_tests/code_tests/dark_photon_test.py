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
