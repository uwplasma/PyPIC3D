import unittest
import os
import tempfile
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import toml
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from tests.kernel_fixtures import (
    build_tiled_particles,
    particle_parameters_from_tile_values,
    particle_species,
)
from PyPIC3D.diagnostics import plotting
from PyPIC3D.utilities.parameters import build_dynamic_parameters, build_static_parameters
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.utilities.grids import build_yee_grid
from PyPIC3D.utilities.plasma_quantities import (
    T_to_vth,
    build_plasma_parameters_dict,
    check_stability,
    vth_to_T,
)
from PyPIC3D.utilities.simulation_helpers import (
    courant_condition,
    particle_sanity_check,
)
from PyPIC3D.utilities.toml_helpers import (
    dump_parameters_to_toml,
    load_external_fields_from_toml,
)
from tests.kernel_fixtures import kernel_parameters_from_values


def _field_parameter_values(Nx, Ny, Nz):
        return {
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "tile_shape": (Nx, Ny, Nz),
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }


def _active_interior(parameter_set):
    g = int(parameter_set["guard_cells"])
    return (
        0,
        0,
        0,
        slice(g, g + int(parameter_set["Nx"])),
        slice(g, g + int(parameter_set["Ny"])),
        slice(g, g + int(parameter_set["Nz"])),
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

    def test_build_yee_grid(self):
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**self.parameter_set))
        self.assertEqual(len(center_grid), 3)
        self.assertEqual(len(vertex_grid), 3)
        self.assertEqual(len(center_grid[0]), self.parameter_set['Nx'] + 2)
        self.assertEqual(len(center_grid[1]), self.parameter_set['Ny'] + 2)
        self.assertEqual(len(center_grid[2]), self.parameter_set['Nz'] + 2)
        self.assertEqual(len(vertex_grid[0]), self.parameter_set['Nx'] + 2)
        self.assertEqual(len(vertex_grid[1]), self.parameter_set['Ny'] + 2)
        self.assertEqual(len(vertex_grid[2]), self.parameter_set['Nz'] + 2)
        #  Check that the center and vertex arrays have the expected lengths

        self.assertTrue(
            jnp.allclose(
                center_grid[0][1:-1],
                -self.parameter_set["x_wind"] / 2
                + self.parameter_set["dx"] * jnp.arange(self.parameter_set["Nx"]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                vertex_grid[0][1:-1],
                center_grid[0][1:-1] + 0.5 * self.parameter_set["dx"],
            )
        )
        # Legacy PyPIC3D names: center is the collocated/base grid, and
        # vertex is the staggered half-cell grid.

    def test_check_stability(self):
        # Should not raise
        check_stability(self.plasma_parameters, 0.01)
        # Check that the stability check does not raise an error

    def test_temperature_thermal_velocity_conversion_round_trip(self):
        temperature = 8.0
        mass = 2.0
        kb = 4.0

        thermal_velocity = T_to_vth(temperature, mass, kb)

        self.assertTrue(jnp.allclose(thermal_velocity, 4.0))
        self.assertTrue(jnp.allclose(vth_to_T(thermal_velocity, mass, kb), temperature))

    def test_build_plasma_parameters_dict(self):
        dynamic_parameters = SimpleNamespace(
            kb=3.0,
            eps=2.0,
            dx=0.5,
            dy=1.0,
            dz=1.0,
            grids=SimpleNamespace(
                center=(
                    jnp.asarray([-1.0, 0.0, 1.0, 2.0]),
                    jnp.asarray([-1.0, 0.0, 1.0]),
                    jnp.asarray([-1.0, 0.0, 1.0]),
                ),
            ),
        )
        electrons = {
            "mass": 2.0,
            "temperature": 4.0,
            "N_particles": 10,
            "charge": -2.0,
            "weight": 0.5,
        }

        plasma_parameters = build_plasma_parameters_dict(
            SimpleNamespace(),
            dynamic_parameters,
            electrons,
        )

        self.assertTrue(
            jnp.allclose(plasma_parameters["Theoretical Plasma Frequency"], jnp.sqrt(2.5))
        )
        self.assertTrue(jnp.allclose(plasma_parameters["Debye Length"], jnp.sqrt(2.4)))
        self.assertTrue(jnp.allclose(plasma_parameters["Thermal Velocity"], jnp.sqrt(18.0)))
        self.assertTrue(
            jnp.allclose(plasma_parameters["dx per debye length"], 2.0 * jnp.sqrt(2.4))
        )

    def test_courant_condition_uses_only_active_dimensions(self):
        dynamic_parameters = SimpleNamespace(C=2.0, Nx=4, Ny=1, Nz=1)

        dt = courant_condition(0.8, 0.5, 1.0, 1.0, dynamic_parameters)

        self.assertAlmostEqual(dt, 0.2)

    def test_particle_sanity_check(self):
        particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 1, 5, 3)),
            u=jnp.zeros((1, 1, 1, 1, 5, 3)),
            active=jnp.ones((1, 1, 1, 1, 5), dtype=bool),
        )
        # Should not raise
        particle_sanity_check(particles)
        # Check that the particle sanity check does not raise an error

    def test_add_external_fields_adds_components(self):
        E = (jnp.ones((2, 2, 2)), jnp.ones((2, 2, 2)) * 2, jnp.ones((2, 2, 2)) * 3)
        B = (jnp.ones((2, 2, 2)) * 4, jnp.ones((2, 2, 2)) * 5, jnp.ones((2, 2, 2)) * 6)
        external_E = (jnp.ones((2, 2, 2)) * 10, jnp.ones((2, 2, 2)) * 20, jnp.ones((2, 2, 2)) * 30)
        external_B = (jnp.ones((2, 2, 2)) * 40, jnp.ones((2, 2, 2)) * 50, jnp.ones((2, 2, 2)) * 60)

        total_E, total_B = add_external_fields(E, B, (external_E, external_B))

        self.assertTrue(jnp.allclose(total_E[0], 11.0))
        self.assertTrue(jnp.allclose(total_E[1], 22.0))
        self.assertTrue(jnp.allclose(total_E[2], 33.0))
        self.assertTrue(jnp.allclose(total_B[0], 44.0))
        self.assertTrue(jnp.allclose(total_B[1], 55.0))
        self.assertTrue(jnp.allclose(total_B[2], 66.0))

    def test_load_external_fields_defaults_to_evolved_fields(self):
        parameter_set = _field_parameter_values(2, 2, 2)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ex.npy")
            np.save(path, np.ones((2, 2, 2)))
            config = {"field1": {"name": "Ex", "type": 0, "path": path}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

        interior = _active_interior(parameter_set)
        self.assertTrue(jnp.allclose(fields[0][interior], 1.0))
        self.assertTrue(jnp.allclose(external_fields[0][0], 0.0))

    def test_load_external_fields_evolve_true_uses_evolved_fields(self):
        parameter_set = _field_parameter_values(2, 2, 2)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "by.npy")
            np.save(path, np.ones((2, 2, 2)) * 3)
            config = {"field1": {"name": "By", "type": 4, "path": path, "evolve": True}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

        interior = _active_interior(parameter_set)
        self.assertTrue(jnp.allclose(fields[4][interior], 3.0))
        self.assertTrue(jnp.allclose(external_fields[1][1], 0.0))

    def test_load_external_fields_evolve_false_uses_external_fields(self):
        parameter_set = _field_parameter_values(2, 2, 2)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bz.npy")
            np.save(path, np.ones((2, 2, 2)) * 5)
            config = {"field1": {"name": "external Bz", "type": 5, "path": path, "evolve": False}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

        interior = _active_interior(parameter_set)
        self.assertTrue(jnp.allclose(fields[5][interior], 0.0))
        self.assertTrue(jnp.allclose(external_fields[1][2][interior], 5.0))

    def test_load_external_fields_rejects_external_current(self):
        parameter_set = _field_parameter_values(2, 2, 2)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "jx.npy")
            np.save(path, np.ones((2, 2, 2)))
            config = {"field1": {"name": "external Jx", "type": 6, "path": path, "evolve": False}}

            with self.assertRaisesRegex(ValueError, "External-only fields must be electric or magnetic"):
                load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

    def test_load_external_fields_preserves_shape_validation(self):
        parameter_set = _field_parameter_values(2, 2, 2)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wrong.npy")
            np.save(path, np.ones((3, 2, 2)))
            config = {"field1": {"name": "wrong Ex", "type": 0, "path": path, "evolve": False}}

            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

    def test_energy_can_include_external_fields(self):
        parameter_set = _field_parameter_values(1, 1, 1)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        external_E = tuple(jnp.zeros_like(comp) for comp in E)
        external_B = tuple(jnp.zeros_like(comp) for comp in B)
        interior = _active_interior(parameter_set)
        external_E = (external_E[0].at[interior].set(2.0), external_E[1], external_E[2])
        external_B = (external_B[0], external_B[1].at[interior].set(3.0), external_B[2])
        total_E, total_B = add_external_fields(E, B, (external_E, external_B))

        parameter_set = parameter_set | {"dx": 1.0, "dy": 1.0, "dz": 1.0}
        dynamic_values = {"eps": 2.0, "mu": 4.0, "C": 10.0}
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 1, 0, 3)),
            u=jnp.zeros((1, 1, 1, 1, 0, 3)),
            active=jnp.zeros((1, 1, 1, 1, 0), dtype=bool),
        )
        species_config = SpeciesConfig(
            charge=jnp.asarray([1.0]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.0]),
            update_x=jnp.ones((1, 3), dtype=bool),
        )
        e_energy, b_energy, kinetic_energy = compute_energy(particles, total_E, total_B, static_parameters, dynamic_parameters, species_config=species_config)

        self.assertTrue(jnp.allclose(e_energy, 4.0))
        self.assertTrue(jnp.allclose(b_energy, 1.125))
        self.assertEqual(kinetic_energy, 0.0)

    def test_compute_energy_ignores_inactive_particles(self):
        parameter_set = _field_parameter_values(1, 1, 1) | {
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
        }
        dynamic_values = {"eps": 1.0, "mu": 1.0, "C": 10.0}
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        tiled_particles = TiledParticles(
            x=jnp.asarray([[[[[[0.6, 0.0, 0.0]]]]]], dtype=float),
            u=jnp.asarray([[[[[[1.0, 0.0, 0.0]]]]]], dtype=float),
            active=jnp.asarray([[[[[False]]]]], dtype=bool),
        )
        species_config = SpeciesConfig(
            charge=jnp.asarray([1.0]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.0]),
            update_x=jnp.ones((1, 3), dtype=bool),
        )
        # Keep a nonzero inactive velocity so this checks the kinetic-energy mask.

        _, _, kinetic_energy = compute_energy(tiled_particles, E, B, static_parameters, dynamic_parameters, species_config=species_config)

        self.assertTrue(jnp.allclose(kinetic_energy, 0.0))

    def test_dump_parameters_to_toml_writes_tiled_species_summaries(self):
        active = jnp.array([[[[[True, False, True], [False, True, False]]]]])
        zeros3 = jnp.zeros(active.shape + (3,))
        particles = TiledParticles(
            x=zeros3,
            u=zeros3,
            active=active,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "data"))
            static_parameters = build_static_parameters({
                "name": "dump test",
                "output_dir": tmpdir,
                "shape_factor": 1,
                "guard_cells": 2,
                "tile_shape": (2, 1, 1),
                "boundary_conditions": {"x": 0, "y": 0, "z": 0},
                "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
                "field_mesh": object(),
            })
            dynamic_parameters = build_dynamic_parameters(
                {
                    "dt": 0.1,
                    "dx": 1.0,
                    "dy": 1.0,
                    "dz": 1.0,
                    "Nx": 2,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 2.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "grids": {
                        "vertex": (jnp.zeros(4), jnp.zeros(3), jnp.zeros(3)),
                        "center": (jnp.zeros(4), jnp.zeros(3), jnp.zeros(3)),
                        "tiled_vertex_grid": (jnp.zeros((1, 1, 1, 6)),) * 3,
                        "tiled_center_grid": (jnp.zeros((1, 1, 1, 6)),) * 3,
                    },
                },
                {},
            )
            plotting_parameters = {
                "plotvelocities": jnp.asarray(True),
                "plotchargedensity": jnp.asarray(True),
                "field_map": {"E": (jnp.zeros((1,)),) * 3},
                "particle_species_names": ("electrons", "ions"),
                "particle_species_metadata": (
                    {"name": "electrons", "charge": -1.0},
                    {"name": "ions", "charge": 1.0},
                ),
            }

            dump_parameters_to_toml(
                {"total_time": 1.0},
                static_parameters,
                dynamic_parameters,
                {},
                plotting_parameters,
                particles,
            )

            config = toml.load(os.path.join(tmpdir, "data/output.toml"))

        self.assertNotIn("particle_species_names", config["static_parameters"])
        self.assertNotIn("particle_species_metadata", config["static_parameters"])
        self.assertNotIn("particle_species_names", config.get("plotting", {}))
        self.assertNotIn("particle_species_metadata", config.get("plotting", {}))
        self.assertNotIn("field_map", config.get("plotting", {}))
        self.assertTrue(config["plotting"]["plotvelocities"])
        self.assertTrue(config["plotting"]["plotchargedensity"])
        self.assertEqual(config["particles"][0]["name"], "electrons")
        self.assertEqual(config["particles"][0]["charge"], -1.0)
        self.assertEqual(config["particles"][0]["storage"], "tiled")
        self.assertEqual(config["particles"][0]["active_particles"], 2)
        self.assertEqual(config["particles"][1]["name"], "ions")
        self.assertEqual(config["particles"][1]["charge"], 1.0)
        self.assertEqual(config["particles"][1]["storage"], "tiled")
        self.assertEqual(config["particles"][1]["active_particles"], 1)
        self.assertEqual(config["particles"][0]["tile_shape"], [2, 1, 1])

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

if __name__ == '__main__':
    unittest.main()
