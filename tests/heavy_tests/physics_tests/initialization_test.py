"""One-tile numerical and physical checks."""


from tests.support.initialization_fixtures import (
    InitializationFunctionsFixtures,
    contextlib,
    initialize_simulation,
    io,
    jax,
    np,
    os,
    tempfile,
    unittest,
)


class TestInitializationFunctions(InitializationFunctionsFixtures, unittest.TestCase):
    def test_filtered_leapfrog_seed_matches_force_and_zero_dark_coupling(self):
        # Use Ey on center-grid nodes: the independent gather is the node value.
        n, dt, alpha, charge, mass, c = 8, .01, .4, 2., 4., 3.
        x = -0.5 + np.arange(n) / n
        evolved = .6 * np.cos(2 * np.pi * np.arange(n) / n)
        external = .3 * np.sin(4 * np.pi * np.arange(n) / n) + .1
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            def save(name, data):
                path = os.path.join(directory, name + ".npy")
                np.save(path, data)
                return path

            x_path = save("x", x)
            evolved_path = save("evolved", evolved.reshape(n, 1, 1))
            external_path = save("external", external.reshape(n, 1, 1))
            for pusher in ("boris", "higuera_cary"):
                for filter_name in ("none", "digital", "bilinear"):
                    if filter_name == "bilinear":
                        filtered = (np.roll(evolved, 1) + 2 * evolved + np.roll(evolved, -1)) / 4
                    elif filter_name == "digital":
                        filtered = alpha * evolved + (1 - alpha) / 6 * (
                            np.roll(evolved, 1) + np.roll(evolved, -1) + 4 * evolved
                        )
                    else:
                        filtered = evolved
                    momentum = -(charge / mass) * (filtered + external) * dt / 2
                    expected = momentum if pusher == "boris" else momentum / np.sqrt(1 + (momentum / c)**2)
                    states = []
                    for solver in ("electrodynamic_yee", "dark_matter_yee"):
                        with self.subTest(pusher=pusher, filter=filter_name, solver=solver):
                            config = {
                                "simulation_parameters": dict(
                                    name="filtered leapfrog seed", output_dir=directory,
                                    solver=solver, Nx=n, Ny=1, Nz=1,
                                    x_wind=1., y_wind=1., z_wind=1., dt=dt, Nt=1,
                                    particle_tile_nx=n, particle_tile_ny=1, particle_tile_nz=1,
                                    shape_factor=1, particle_pusher=pusher, relativistic=False,
                                    current_calculation="j_from_rhov", filter_j=filter_name,
                                    alpha=alpha, C=c, eps=1., mu=1/c**2, sin_chi=0., dark_mu=0.,
                                ),
                                "particle1": dict(
                                    name="p", N_particles=n, charge=charge, mass=mass, temperature=0.,
                                    initial_x=x_path, initial_y=0., initial_z=0.,
                                    initial_vx=0., initial_vy=0., initial_vz=0.,
                                ),
                                "field1": dict(name="evolved Ey", type=1, path=evolved_path, evolve=True),
                                "field2": dict(name="external Ey", type=1, path=external_path, evolve=False),
                            }
                            _, particles, *_ = initialize_simulation(config)
                            positions = np.asarray(particles.x)[np.asarray(particles.active)]
                            velocities = np.asarray(particles.u)[np.asarray(particles.active)]
                            order = np.argsort(positions[:, 0])
                            np.testing.assert_allclose(positions[order, 0], x, atol=1e-15)
                            np.testing.assert_array_equal(positions[:, 1:], 0.)
                            np.testing.assert_allclose(velocities[order, 1], expected, rtol=1e-11, atol=1e-14)
                            np.testing.assert_array_equal(velocities[:, (0, 2)], 0.)
                            states.append(velocities[order])
                    np.testing.assert_array_equal(states[0], states[1])

    def test_initialize_simulation_offsets_particle_velocity_to_the_half_step(self):
        """
        The time loops advance ``u^{n-1/2}`` to ``u^{n+1/2}`` with the force at
        ``x^n``, so the run has to start from ``u(-dt/2)``.  Handing the
        configured ``u(0)`` straight in leaves an O(dt) error in the initial
        state and drops the whole simulation to first order.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            Nx, Ny, Nz = 4, 1, 1
            dt = 1.0e-3
            charge, mass, Ex = -2.0, 4.0, 0.75

            zeros_path = os.path.join(tmpdir, "zeros.npy")
            x_path = os.path.join(tmpdir, "x.npy")
            field_path = os.path.join(tmpdir, "ex.npy")
            np.save(x_path, np.array([-0.375, -0.125, 0.125, 0.375]))
            np.save(zeros_path, np.zeros(4))
            np.save(field_path, np.full((Nx, Ny, Nz), Ex))

            config = {
                "simulation_parameters": {
                    "name": "leapfrog velocity seed",
                    "output_dir": tmpdir,
                    "Nx": Nx,
                    "Ny": Ny,
                    "Nz": Nz,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "Nt": 1,
                    "dt": dt,
                    "particle_tile_nx": Nx,
                    "particle_tile_ny": Ny,
                    "particle_tile_nz": Nz,
                    "filter_j": "none",
                },
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": charge,
                    "mass": mass,
                    "temperature": 0.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
                "field1": {
                    "name": "uniform Ex",
                    "type": 0,
                    "path": field_path,
                    "evolve": False,
                },
            }

            _loop, particles, *_rest = initialize_simulation(config)

            u = np.asarray(jax.device_get(particles.u))[particles.active]
            expected_vx = -(charge / mass) * Ex * 0.5 * dt
            # a particle at rest feels only the electric kick over -dt/2
            np.testing.assert_allclose(
                u[:, 0], expected_vx, rtol=1.0e-6,
                err_msg="initialize_simulation did not offset u onto the leapfrog half step",
            )
            np.testing.assert_allclose(u[:, 1], 0.0, atol=1.0e-15)
            np.testing.assert_allclose(u[:, 2], 0.0, atol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
