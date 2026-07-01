import unittest
import os
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import toml
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.diagnostics import plotting, vtk as vtk_diagnostics
from PyPIC3D.solvers.yee_tiled import tile_vector_field
from PyPIC3D.utils import (
    print_stats, build_yee_grid, build_collocated_grid, check_stability,
    particle_sanity_check, load_external_fields_from_toml, add_external_fields,
    compute_energy, dump_parameters_to_toml,
)
from PyPIC3D.particles.species_class import particle_species

jax.config.update("jax_enable_x64", True)

class TestUtilsFunctions(unittest.TestCase):
    def setUp(self):
        self.world = {
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
        grid, staggered = build_yee_grid(self.world)
        self.assertEqual(len(grid), 3)
        self.assertEqual(len(staggered), 3)
        self.assertEqual(len(grid[0]), self.world['Nx'] + 2)
        self.assertEqual(len(grid[1]), self.world['Ny'] + 2)
        self.assertEqual(len(grid[2]), self.world['Nz'] + 2)
        self.assertEqual(len(staggered[0]), self.world['Nx'] + 2)
        self.assertEqual(len(staggered[1]), self.world['Ny'] + 2)
        self.assertEqual(len(staggered[2]), self.world['Nz'] + 2)
        #  Check that the grid and staggered arrays have the expected lengths

    def test_check_stability(self):
        # Should not raise
        check_stability(self.plasma_parameters, 0.01)
        # Check that the stability check does not raise an error

    def test_particle_sanity_check(self):
        class DummyParticles:
            def __iter__(self):
                return iter([self])
            def get_position(self):
                N = 5
                return (jnp.zeros(N), jnp.zeros(N), jnp.zeros(N))
            def get_velocity(self):
                N = 5
                return (jnp.zeros(N), jnp.zeros(N), jnp.zeros(N))
            def get_number_of_particles(self):
                return 5
        particles = DummyParticles()
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
        E, B, J, phi, rho = initialize_fields(2, 2, 2)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ex.npy")
            np.save(path, np.ones((2, 2, 2)))
            config = {"field1": {"name": "Ex", "type": 0, "path": path}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config)

        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        self.assertTrue(jnp.allclose(fields[0][interior], 1.0))
        self.assertTrue(jnp.allclose(external_fields[0][0], 0.0))

    def test_load_external_fields_evolve_true_uses_evolved_fields(self):
        E, B, J, phi, rho = initialize_fields(2, 2, 2)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "by.npy")
            np.save(path, np.ones((2, 2, 2)) * 3)
            config = {"field1": {"name": "By", "type": 4, "path": path, "evolve": True}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config)

        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        self.assertTrue(jnp.allclose(fields[4][interior], 3.0))
        self.assertTrue(jnp.allclose(external_fields[1][1], 0.0))

    def test_load_external_fields_evolve_false_uses_external_fields(self):
        E, B, J, phi, rho = initialize_fields(2, 2, 2)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bz.npy")
            np.save(path, np.ones((2, 2, 2)) * 5)
            config = {"field1": {"name": "external Bz", "type": 5, "path": path, "evolve": False}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config)

        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        self.assertTrue(jnp.allclose(fields[5][interior], 0.0))
        self.assertTrue(jnp.allclose(external_fields[1][2][interior], 5.0))

    def test_load_external_fields_rejects_external_current(self):
        E, B, J, phi, rho = initialize_fields(2, 2, 2)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "jx.npy")
            np.save(path, np.ones((2, 2, 2)))
            config = {"field1": {"name": "external Jx", "type": 6, "path": path, "evolve": False}}

            with self.assertRaisesRegex(ValueError, "External-only fields must be electric or magnetic"):
                load_external_fields_from_toml(fields, external_fields, config)

    def test_load_external_fields_preserves_shape_validation(self):
        E, B, J, phi, rho = initialize_fields(2, 2, 2)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wrong.npy")
            np.save(path, np.ones((3, 2, 2)))
            config = {"field1": {"name": "wrong Ex", "type": 0, "path": path, "evolve": False}}

            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                load_external_fields_from_toml(fields, external_fields, config)

    def test_energy_can_include_external_fields(self):
        E, B, J, phi, rho = initialize_fields(1, 1, 1)
        external_E = tuple(jnp.zeros_like(comp) for comp in E)
        external_B = tuple(jnp.zeros_like(comp) for comp in B)
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        external_E = (external_E[0].at[interior].set(2.0), external_E[1], external_E[2])
        external_B = (external_B[0], external_B[1].at[interior].set(3.0), external_B[2])
        total_E, total_B = add_external_fields(E, B, (external_E, external_B))

        world = {"dx": 1.0, "dy": 1.0, "dz": 1.0, "Nx": 1, "Ny": 1, "Nz": 1}
        constants = {"eps": 2.0, "mu": 4.0, "C": 10.0}
        e_energy, b_energy, kinetic_energy = compute_energy([], total_E, total_B, world, constants)

        self.assertTrue(jnp.allclose(e_energy, 4.0))
        self.assertTrue(jnp.allclose(b_energy, 1.125))
        self.assertEqual(kinetic_energy, 0.0)

    def test_compute_energy_ignores_inactive_particles(self):
        E, B, J, phi, rho = initialize_fields(1, 1, 1)
        world = {"dx": 1.0, "dy": 1.0, "dz": 1.0, "Nx": 1, "Ny": 1, "Nz": 1}
        constants = {"eps": 1.0, "mu": 1.0, "C": 10.0}
        species = particle_species(
            name="absorbed",
            N_particles=1,
            charge=1.0,
            mass=1.0,
            weight=1.0,
            T=0.0,
            v1=jnp.array([1.0]),
            v2=jnp.array([0.0]),
            v3=jnp.array([0.0]),
            x1=jnp.array([0.6]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            xwind=1.0,
            ywind=1.0,
            zwind=1.0,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            x_bc="periodic",
        )
        species.boundary_conditions({"particle_boundary_conditions": {"x": 2, "y": 0, "z": 0}})

        _, _, kinetic_energy = compute_energy([species], E, B, world, constants)

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
            simulation_parameters = {
                "output_dir": tmpdir,
                "particle_species_names": ("electrons", "ions"),
                "tile_shape": (2, 1, 1),
            }

            dump_parameters_to_toml(
                {"total_time": 1.0},
                simulation_parameters,
                {},
                {},
                {},
                particles,
            )

            config = toml.load(os.path.join(tmpdir, "data/output.toml"))

        self.assertEqual(config["particles"][0]["name"], "electrons")
        self.assertEqual(config["particles"][0]["storage"], "tiled")
        self.assertEqual(config["particles"][0]["active_particles"], 2)
        self.assertEqual(config["particles"][1]["name"], "ions")
        self.assertEqual(config["particles"][1]["storage"], "tiled")
        self.assertEqual(config["particles"][1]["active_particles"], 1)
        self.assertEqual(config["particles"][0]["tile_shape"], [2, 1, 1])

    def test_plot_positions_flattens_tiled_particles_and_preserves_species_names(self):
        species = particle_species(
            name="beam electrons",
            N_particles=2,
            charge=-1.0,
            mass=1.0,
            weight=2.0,
            T=0.0,
            x1=jnp.array([-0.25, 0.25]),
            x2=jnp.array([0.0, 0.0]),
            x3=jnp.array([0.0, 0.0]),
            v1=jnp.array([0.1, 0.2]),
            v2=jnp.array([0.0, 0.0]),
            v3=jnp.array([0.0, 0.0]),
            xwind=1.0,
            ywind=1.0,
            zwind=1.0,
            dx=0.25,
            dy=0.5,
            dz=1.0,
            dt=0.1,
        )
        world = {
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
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        tiled_particles, species_config = to_tiled_particles([species], world, simulation_parameters)

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
                    world["x_wind"],
                    world["y_wind"],
                    world["z_wind"],
                    tmpdir,
                    species_config=species_config,
                    species_names=("beam electrons",),
                    world=world,
                )

        self.assertEqual(figure.trace_names, ["beam electrons"])

    def test_particles_phase_space_flattens_tiled_particles(self):
        species = particle_species(
            name="electrons",
            N_particles=1,
            charge=-1.0,
            mass=1.0,
            weight=1.0,
            T=0.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([0.2]),
            v2=jnp.array([0.0]),
            v3=jnp.array([0.0]),
            xwind=1.0,
            ywind=1.0,
            zwind=1.0,
            dx=0.25,
            dy=0.5,
            dz=1.0,
            dt=0.1,
        )
        world = {
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
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        tiled_particles, species_config = to_tiled_particles([species], world, simulation_parameters)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "data/phase_space/x"))
            os.makedirs(os.path.join(tmpdir, "data/phase_space/y"))
            os.makedirs(os.path.join(tmpdir, "data/phase_space/z"))
            with unittest.mock.patch.object(plotting.plt, "scatter") as scatter:
                with unittest.mock.patch.object(plotting.plt, "savefig"):
                    plotting.particles_phase_space(
                        tiled_particles,
                        world,
                        0,
                        "electrons",
                        tmpdir,
                        species_config=species_config,
                        species_names=("electrons",),
                    )

        self.assertEqual(scatter.call_count, 3)

    def test_vtk_plot_fields_assembles_tiled_components_at_output_boundary(self):
        world = {
            "Nx": 4,
            "Ny": 2,
            "Nz": 2,
            "dx": 0.25,
            "dy": 0.5,
            "dz": 0.75,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.5,
            "tile_shape": (2, 1, 1),
            "guard_cells": 2,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        field = tuple(jnp.zeros((8, 6, 6)) + component for component in (1.0, 2.0, 3.0))
        tiled_field = tile_vector_field(field, world, world["tile_shape"], num_guard_cells=world["guard_cells"])

        with unittest.mock.patch.object(vtk_diagnostics, "gridToVTK") as grid_to_vtk:
            vtk_diagnostics.plot_fields(
                tiled_field[0],
                tiled_field[1],
                tiled_field[2],
                0,
                "E",
                world["dx"],
                world["dy"],
                world["dz"],
                world=world,
            )

        cell_data = grid_to_vtk.call_args.kwargs["cellData"]
        self.assertEqual(cell_data["E_x"].shape, (6, 4, 4))
        self.assertTrue(jnp.allclose(cell_data["E_y"], 2.0))

if __name__ == '__main__':
    unittest.main()
