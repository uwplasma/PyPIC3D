from tests.support.utils_runtime_fixtures import (
    build_tiled_particles,
    jnp,
    os,
    particle_parameters_from_tile_values,
    particle_species,
    plotting,
    tempfile,
    unittest,
)


class TestUtilsFunctions(unittest.TestCase):
    def setUp(self):
        self.parameter_set = {
            'Nx': 4,
            'Ny': 4,
            'Nz': 4,
            'dx': 0.1,
            'dy': 0.1,
            'dz': 0.1,
            'dt': 0.01,
            'x_wind': 1.0,
            'y_wind': 1.0,
            'z_wind': 1.0
        }
        self.plasma_parameters = {
            "Theoretical Plasma Frequency": 1.0,
            "Debye Length": 0.01,
            "Thermal Velocity": 1.0,
            "Number of Electrons": 10,
            "dx per debye length": 2.0
        }

    def test_plot_positions_flattens_tiled_particles_and_preserves_species_names(self):
        species = particle_species(
            name="beam electrons",
            charge=-1.0,
            mass=1.0,
            weight=2.0,
            x1=jnp.array([-0.25, 0.25]),
            x2=jnp.array([0.0, 0.0]),
            x3=jnp.array([0.0, 0.0]),
            u1=jnp.array([0.1, 0.2]),
            u2=jnp.array([0.0, 0.0]),
            u3=jnp.array([0.0, 0.0]),
        )
        parameter_set = {
            "Nx": 4,
            "Ny": 2,
            "Nz": 1,
            "dx": 0.25,
            "dy": 0.5,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "tile_shape": (2, 1, 1),
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        static_parameters, dynamic_parameters = particle_parameters_from_tile_values(parameter_set, simulation_parameters)
        tiled_particles, species_config = build_tiled_particles([species], static_parameters, dynamic_parameters)

        class FakeFigure:
            def __init__(self):
                self.trace_names = []

            def add_trace(self, trace):
                self.trace_names.append(trace.name)

            def update_layout(self, *args, **kwargs):
                pass

            def write_html(self, *args, **kwargs):
                pass

        figure = FakeFigure()
        with tempfile.TemporaryDirectory() as tmpdir:
            with unittest.mock.patch.object(plotting.go, "Figure", return_value=figure):
                plotting.plot_positions(
                    tiled_particles,
                    0,
                    static_parameters,
                    dynamic_parameters,
                    tmpdir,
                    species_config=species_config,
                    species_names=("beam electrons",),
                )

        self.assertEqual(figure.trace_names, ["beam electrons"])

    def test_particles_phase_space_flattens_tiled_particles(self):
        species = particle_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=1.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            u1=jnp.array([0.2]),
            u2=jnp.array([0.0]),
            u3=jnp.array([0.0]),
        )
        parameter_set = {
            "Nx": 4,
            "Ny": 2,
            "Nz": 1,
            "dx": 0.25,
            "dy": 0.5,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "tile_shape": (2, 1, 1),
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        static_parameters, dynamic_parameters = particle_parameters_from_tile_values(parameter_set, simulation_parameters)
        tiled_particles, species_config = build_tiled_particles([species], static_parameters, dynamic_parameters)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "data/phase_space/x"))
            os.makedirs(os.path.join(tmpdir, "data/phase_space/y"))
            os.makedirs(os.path.join(tmpdir, "data/phase_space/z"))
            with unittest.mock.patch.object(plotting.plt, "scatter") as scatter:
                with unittest.mock.patch.object(plotting.plt, "savefig"):
                    plotting.particles_phase_space(
                        tiled_particles,
                        static_parameters,
                        dynamic_parameters,
                        0,
                        "electrons",
                        tmpdir,
                        species_config=species_config,
                        species_names=("electrons",),
                    )

        self.assertEqual(scatter.call_count, 3)


if __name__ == "__main__":
    unittest.main()
