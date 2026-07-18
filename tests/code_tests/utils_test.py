import unittest
import os
import tempfile
import importlib.util
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import toml
import PyPIC3D
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from tests.initial_particles import build_tiled_particles, tiled_species
from PyPIC3D.diagnostics import plotting
from PyPIC3D.parameters import build_dynamic_parameters, build_static_parameters
from PyPIC3D.utilities.grids import build_collocated_grid, build_yee_grid
from PyPIC3D.utils import (
    print_stats, check_stability,
    particle_sanity_check, load_external_fields_from_toml, add_external_fields,
    compute_energy, dump_parameters_to_toml,
)
from tests.parameter_helpers import field_initialization_parameters

jax.config.update("jax_enable_x64", True)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, world, tile_shape, num_guard_cells=2):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

    if g != 1:
        field_tiles = jnp.zeros(
            (
                ntx,
                nty,
                ntz,
                tile_nx + 2 * g,
                tile_ny + 2 * g,
                tile_nz + 2 * g,
            ),
            dtype=field.dtype,
        )
        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    ix = 1 + tx * tile_nx
                    iy = 1 + ty * tile_ny
                    iz = 1 + tz * tile_nz
                    interior = field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                    field_tiles = field_tiles.at[tx, ty, tz, g:-g, g:-g, g:-g].set(interior)
        world = dict(world)
        world["tile_shape"] = tuple(int(width) for width in tile_shape)
        world["field_mesh"] = ghost_cells.make_field_mesh((ntx, nty, ntz))
        static_parameters, _ = field_initialization_parameters(world)
        return ghost_cells.update_tiled_ghost_cells(field_tiles, static_parameters, g)

    def tile_at(tx, ty, tz):
        start = (tx * tile_nx, ty * tile_ny, tz * tile_nz)
        size = (tile_nx + 2, tile_ny + 2, tile_nz + 2)
        return jax.lax.dynamic_slice(field, start, size)

    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack([tile_at(tx, ty, tz) for tz in range(ntz)], axis=0)
                    for ty in range(nty)
                ],
                axis=0,
            )
            for tx in range(ntx)
        ],
        axis=0,
    )


def _field_world(Nx, Ny, Nz):
        return {
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "tile_shape": (Nx, Ny, Nz),
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }


def _active_interior(world):
    g = int(world["guard_cells"])
    return (
        0,
        0,
        0,
        slice(g, g + int(world["Nx"])),
        slice(g, g + int(world["Ny"])),
        slice(g, g + int(world["Nz"])),
    )


def _tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


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
        grid, staggered = build_yee_grid(SimpleNamespace(**self.world))
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
        world = _field_world(2, 2, 2)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ex.npy")
            np.save(path, np.ones((2, 2, 2)))
            config = {"field1": {"name": "Ex", "type": 0, "path": path}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

        interior = _active_interior(world)
        self.assertTrue(jnp.allclose(fields[0][interior], 1.0))
        self.assertTrue(jnp.allclose(external_fields[0][0], 0.0))

    def test_load_external_fields_evolve_true_uses_evolved_fields(self):
        world = _field_world(2, 2, 2)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "by.npy")
            np.save(path, np.ones((2, 2, 2)) * 3)
            config = {"field1": {"name": "By", "type": 4, "path": path, "evolve": True}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

        interior = _active_interior(world)
        self.assertTrue(jnp.allclose(fields[4][interior], 3.0))
        self.assertTrue(jnp.allclose(external_fields[1][1], 0.0))

    def test_load_external_fields_evolve_false_uses_external_fields(self):
        world = _field_world(2, 2, 2)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        fields = [component for field in [E, B, J] for component in field]
        external_fields = (tuple(jnp.zeros_like(comp) for comp in E), tuple(jnp.zeros_like(comp) for comp in B))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bz.npy")
            np.save(path, np.ones((2, 2, 2)) * 5)
            config = {"field1": {"name": "external Bz", "type": 5, "path": path, "evolve": False}}

            fields, external_fields = load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters)

        interior = _active_interior(world)
        self.assertTrue(jnp.allclose(fields[5][interior], 0.0))
        self.assertTrue(jnp.allclose(external_fields[1][2][interior], 5.0))

    def test_load_external_fields_rejects_external_current(self):
        world = _field_world(2, 2, 2)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
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
        world = _field_world(2, 2, 2)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
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
        world = _field_world(1, 1, 1)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
        E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        external_E = tuple(jnp.zeros_like(comp) for comp in E)
        external_B = tuple(jnp.zeros_like(comp) for comp in B)
        interior = _active_interior(world)
        external_E = (external_E[0].at[interior].set(2.0), external_E[1], external_E[2])
        external_B = (external_B[0], external_B[1].at[interior].set(3.0), external_B[2])
        total_E, total_B = add_external_fields(E, B, (external_E, external_B))

        world = world | {"dx": 1.0, "dy": 1.0, "dz": 1.0}
        constants = {"eps": 2.0, "mu": 4.0, "C": 10.0}
        static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
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
            update_u=jnp.ones((1, 3), dtype=bool),
        )
        e_energy, b_energy, kinetic_energy = compute_energy(particles, total_E, total_B, static_parameters, dynamic_parameters, species_config=species_config)

        self.assertTrue(jnp.allclose(e_energy, 4.0))
        self.assertTrue(jnp.allclose(b_energy, 1.125))
        self.assertEqual(kinetic_energy, 0.0)

    def test_compute_energy_ignores_inactive_particles(self):
        world = _field_world(1, 1, 1) | {
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
        }
        constants = {"eps": 1.0, "mu": 1.0, "C": 10.0}
        static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
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
            update_u=jnp.ones((1, 3), dtype=bool),
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
        self.assertEqual(config["particles"][0]["name"], "electrons")
        self.assertEqual(config["particles"][0]["charge"], -1.0)
        self.assertEqual(config["particles"][0]["storage"], "tiled")
        self.assertEqual(config["particles"][0]["active_particles"], 2)
        self.assertEqual(config["particles"][1]["name"], "ions")
        self.assertEqual(config["particles"][1]["charge"], 1.0)
        self.assertEqual(config["particles"][1]["storage"], "tiled")
        self.assertEqual(config["particles"][1]["active_particles"], 1)
        self.assertEqual(config["particles"][0]["tile_shape"], [2, 1, 1])

    def test_package_does_not_export_vtk_diagnostics(self):
        self.assertFalse(hasattr(PyPIC3D, "vtk"))
        self.assertIsNone(importlib.util.find_spec("PyPIC3D.diagnostics.vtk"))

    def test_plot_positions_flattens_tiled_particles_and_preserves_species_names(self):
        species = tiled_species(
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
            "tile_shape": (2, 1, 1),
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        tiled_particles, species_config = build_tiled_particles([species], world, simulation_parameters=simulation_parameters)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
        static_parameters.particle_boundary_conditions = (0, 0, 0)

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
        species = tiled_species(
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
            "tile_shape": (2, 1, 1),
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        tiled_particles, species_config = build_tiled_particles([species], world, simulation_parameters=simulation_parameters)
        static_parameters, dynamic_parameters = field_initialization_parameters(world)
        static_parameters.particle_boundary_conditions = (0, 0, 0)

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
