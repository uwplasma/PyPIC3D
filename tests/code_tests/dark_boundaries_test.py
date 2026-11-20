"""One-tile API contracts."""

from tests.support.dark_boundaries_fixtures import (
    DarkBoundariesFixtures,
    build_field_output_map,
    build_static_parameters,
    contextlib,
    initialize_dark_photon_fields,
    initialize_dark_pml,
    initialize_simulation,
    io,
    jax,
    jnp,
    load_pml_from_toml,
    np,
    periodic_tests,
    pmd,
    run_PyPIC3D,
    synchronized_dark_fields,
    tempfile,
    time_loop_dark_photon,
    time_loop_electrodynamic,
    unittest,
)


class TestDarkBoundaries(DarkBoundariesFixtures, unittest.TestCase):
    def test_public_initialization_and_sector_history_independence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = periodic_tests.DarkPhotonFixtures().config(directory, x_bc="conducting", sin_chi=0.)
            config["pml"] = [dict(wall="+x", thickness=3)]
            path = directory + "/dark_A.npy"
            np.save(path, np.sin(np.pi*np.arange(8)/8)[:, None, None])
            config["dark_field4"] = dict(type=4, path=path)
            loop, particles, fields, s, d, _, _, species = initialize_simulation(config)
            self.assertIs(loop, time_loop_dark_photon)
            self.assertEqual(len(fields), 9)
            self.assertEqual(s.boundary_conditions, (1, 0, 0))
            self.assertEqual(build_static_parameters(s._asdict()), s)
            eager = loop(particles, species, fields, s, d)
            compiled = jax.jit(lambda p, f: loop(p, species, f, s, d))(particles, fields)
            self.assert_tree_close(eager, compiled)
            fields = compiled[1]
            for history in jax.tree.leaves(fields[6][0][:2]):
                np.testing.assert_array_equal(history, 0.)
            # A starts nonzero but E starts zero: Ampere memory is excited on
            # this first kick; Faraday memory is first driven on the next drift.
            self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in fields[6][1][4]), 0.)
            output = build_field_output_map(fields, compiled[0], species, s, d)
            expected = synchronized_dark_fields(fields[7], s, d, fields[6][1])
            self.assert_tree_close(output["dark_B"], expected[3])

            # Excite only Maxwell and compare the complete coupled production
            # step against the ordinary Yee loop, including its PML histories.
            zero_dark = initialize_dark_photon_fields(s, d)
            profiles = load_pml_from_toml(config["pml"], s, d)[4]
            zero_dark, dark_pml = initialize_dark_pml(zero_dark, s, d, profiles)
            ordinary = (expected[1], expected[3], *fields[2:6], fields[6][0], fields[8])
            coupled = ordinary[:6] + ((ordinary[6], dark_pml), zero_dark, ordinary[7])
            yee = jax.jit(lambda p, f: time_loop_electrodynamic(p, species, f, s, d))(particles, ordinary)
            coupled = jax.jit(lambda p, f: loop(p, species, f, s, d))(particles, coupled)
            actual = coupled[1]
            self.assert_tree_close(yee, (coupled[0], actual[:6] + (actual[6][0], actual[8])))
            for history in jax.tree.leaves(actual[6][1][1:5]):
                np.testing.assert_array_equal(history, 0.)

    def test_pml_driver_particle_seed_and_openpmd(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = periodic_tests.DarkPhotonFixtures().config(
                directory, relativistic=False, current_calculation="esirkepov",
            )
            config["pml"] = [dict(wall="+x", thickness=3)]
            config["plotting"] = dict(dump_fields=False, plot_openpmd_fields=True)
            path = directory + "/dark_E.npy"
            np.save(path, np.full((8, 1, 1), .75))
            config["dark_field0"] = dict(type=0, path=path)
            config["particle1"] = dict(name="p", N_particles=1, mass=4., charge=2., temperature=0.,
                                       initial_x=0., initial_y=0., initial_z=0.,
                                       initial_vx=0., initial_vy=0., initial_vz=0.)
            _, particles, fields, s, d, *_ = initialize_simulation(config)
            np.testing.assert_allclose(np.asarray(particles.u)[np.asarray(particles.active), 0],
                                       (2/4)*s.sin_chi*.75*float(d.dt)/2, atol=1e-15)
            result = run_PyPIC3D(config)
            self.assertEqual(len(result[5]), 9)
            self.assertIsNotNone(result[5][6])
            for value in jax.tree.leaves(result[5]):
                self.assertTrue(np.all(np.isfinite(value)))
            series = pmd.Series(directory + "/data/fields.h5", pmd.Access.read_only)
            for iteration in series.iterations:
                for name in ("dark_E", "dark_A", "dark_phi", "dark_B"):
                    self.assertEqual(series.iterations[iteration].meshes[name].time_offset, 0.)
            values = series.iterations[0].meshes["dark_E"]["x"].load_chunk()
            series.flush()
            np.testing.assert_allclose(values, .75, atol=1e-14)
            series.close()


if __name__ == "__main__":
    unittest.main()
