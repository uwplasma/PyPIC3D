import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.supergaussian import (
    SUPERGAUSSIAN_WALLS,
    apply_tiled_supergaussian_absorber,
    build_supergaussian_envelope,
    load_supergaussian_from_toml,
)
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.solvers.yee.first_order_yee import update_B, update_E
from PyPIC3D.utilities.grids import build_yee_grid
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy
from tests.kernel_fixtures import kernel_parameters_from_values, tile_vector_field


def _empty_global_fields(parameter_set):
    shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
    E = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    B = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    J = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    return E, B, J


def _base_parameter_values(nx=24, ny=1, nz=1):
    parameter_set = {
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "dx": 1.0 / nx,
        "dy": 1.0 if ny == 1 else 1.0 / ny,
        "dz": 1.0 if nz == 1 else 1.0 / nz,
        "dt": 0.5 / nx,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }
    center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
    parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
    return parameter_set


def _dynamic_parameters(parameter_set, dynamic_values=None):
    if dynamic_values is None:
        dynamic_values = {}
    return SimpleNamespace(
        Nx=parameter_set["Nx"],
        Ny=parameter_set["Ny"],
        Nz=parameter_set["Nz"],
        dx=parameter_set["dx"],
        dy=parameter_set["dy"],
        dz=parameter_set["dz"],
        dt=parameter_set["dt"],
        x_wind=parameter_set["x_wind"],
        y_wind=parameter_set["y_wind"],
        z_wind=parameter_set["z_wind"],
        C=parameter_set.get("C", dynamic_values.get("C", 1.0)),
        eps=parameter_set.get("eps", dynamic_values.get("eps", 1.0)),
        mu=parameter_set.get("mu", dynamic_values.get("mu", 1.0)),
        alpha=parameter_set.get("alpha", dynamic_values.get("alpha", 1.0)),
        grids=SimpleNamespace(**parameter_set["grids"]),
    )


def _load_supergaussian(raw, parameter_set, dynamic_values=None):
    return load_supergaussian_from_toml(raw, _dynamic_parameters(parameter_set, dynamic_values))


def _empty_config(tmpdir, solver="electrodynamic_yee", supergaussian=None):
    sim = {
        "name": "supergaussian init test",
        "output_dir": tmpdir,
        "solver": solver,
        "Nx": 8,
        "Ny": 1,
        "Nz": 1,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "Nt": 1,
        "dt": 1e-10,
    }
    config = {"simulation_parameters": sim, "plotting": {"plotting": False}}
    if supergaussian is not None:
        config["supergaussian"] = supergaussian
    return config


class TestSupergaussianConfiguration(unittest.TestCase):
    def test_load_supergaussian_accepts_all_six_walls(self):
        raw = [
            {"wall": wall, "width": 2, "order": 4.0, "target_reflection": 1.0e-8}
            for wall in SUPERGAUSSIAN_WALLS
        ]

        active, sg_x, sg_y, sg_z, layers = _load_supergaussian(raw, _base_parameter_values(8, 8, 8), {"C": 3.0})

        self.assertTrue(active)
        self.assertTrue(sg_x)
        self.assertTrue(sg_y)
        self.assertTrue(sg_z)
        self.assertEqual(len(layers), 6)

    def test_load_supergaussian_rejects_invalid_duplicate_and_oversized_walls(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)

        with self.assertRaisesRegex(ValueError, "Invalid supergaussian wall"):
            _load_supergaussian([{"wall": "x+", "width": 2}], parameter_set)

        with self.assertRaisesRegex(ValueError, "Duplicate supergaussian wall"):
            _load_supergaussian(
                [{"wall": "+x", "width": 2}, {"wall": "+x", "width": 2}],
                parameter_set,
            )

        with self.assertRaisesRegex(ValueError, "exceeds active cells"):
            _load_supergaussian([{"wall": "+x", "width": 9}], parameter_set)

    def test_build_envelope_damps_only_requested_wall(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        parameter_set["supergaussian"] = _load_supergaussian(
            [{"wall": "+x", "width": 3, "order": 2.0, "sigma_max": 9.0}],
            parameter_set,
        )
        tile_shape = (8, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)

        envelope = build_supergaussian_envelope(static_parameters, dynamic_parameters, dynamic_parameters.dt)
        assembled = assemble_tiled_vector_field((envelope, envelope, envelope), parameter_set, tile_shape)[0]
        line = assembled[1:-1, 1, 1]

        self.assertTrue(jnp.allclose(line[:5], 1.0))
        self.assertLess(float(line[-1]), float(line[-2]))
        self.assertLess(float(line[-2]), float(line[-3]))
        self.assertGreater(float(line[-3]), 0.0)


class TestSupergaussianInitialization(unittest.TestCase):
    def test_initialize_simulation_rejects_supergaussian_for_electrostatic_solver(self):
        supergaussian = [{"wall": "+x", "width": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "not supported for the electrostatic solver"):
                initialize_simulation(_empty_config(tmpdir, solver="electrostatic", supergaussian=supergaussian))

    def test_initialize_simulation_stores_static_supergaussian_layers(self):
        supergaussian = [{"wall": "+x", "width": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir, supergaussian=supergaussian))

        fields = result[2]
        static_parameters = result[3]
        self.assertEqual(len(fields), 8)
        self.assertTrue(static_parameters.supergaussian_active)
        self.assertEqual(len(static_parameters.supergaussian_layers), 1)
        self.assertEqual(static_parameters.boundary_conditions, (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC))

    def test_initialize_simulation_allows_static_metric_supergaussian_with_constant_radial_bc(self):
        supergaussian = [
            {"wall": "-x", "width": 2, "sigma_max": 1.0},
            {"wall": "+x", "width": 2, "sigma_max": 1.0},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _empty_config(tmpdir, solver="static_metric", supergaussian=supergaussian)
            config["simulation_parameters"].update(
                {
                    "metric": "flat_spherical",
                    "particle_pusher": "hybrid_boris_geodesic",
                    "current_calculation": "GR_direct_deposition",
                    "x_bc": "constant",
                    "y_bc": "periodic",
                    "z_bc": "periodic",
                    "x_min": 1.0,
                    "x_max": 2.0,
                    "y_min": 0.17,
                    "y_max": 1.17,
                }
            )

            result = initialize_simulation(config)

        fields = result[2]
        static_parameters = result[3]
        self.assertEqual(len(fields), 9)
        self.assertTrue(static_parameters.supergaussian_active)
        self.assertEqual(len(static_parameters.supergaussian_layers), 2)
        self.assertEqual(static_parameters.boundary_conditions, (BC_CONSTANT, BC_PERIODIC, BC_PERIODIC))


class TestSupergaussianFDTDBehavior(unittest.TestCase):
    def test_tiled_supergaussian_absorber_damps_E_and_B_but_not_J(self):
        parameter_set = _base_parameter_values(nx=8, ny=1, nz=1)
        parameter_set["supergaussian"] = _load_supergaussian(
            [{"wall": "+x", "width": 3, "order": 2.0, "sigma_max": 20.0}],
            parameter_set,
        )
        tile_shape = (8, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        E, B, J = _empty_global_fields(parameter_set)
        E_tiles = tile_vector_field((E[0], E[1].at[1:-1, 1, 1].set(1.0), E[2]), parameter_set, tile_shape)
        B_tiles = tile_vector_field((B[0], B[1], B[2].at[1:-1, 1, 1].set(1.0)), parameter_set, tile_shape)
        J_tiles = tile_vector_field((J[0], J[1].at[1:-1, 1, 1].set(1.0), J[2]), parameter_set, tile_shape)

        E_after = apply_tiled_supergaussian_absorber(E_tiles, static_parameters, dynamic_parameters, dynamic_parameters.dt)
        B_after = apply_tiled_supergaussian_absorber(B_tiles, static_parameters, dynamic_parameters, dynamic_parameters.dt)

        E_global = assemble_tiled_vector_field(E_after, parameter_set, tile_shape)
        B_global = assemble_tiled_vector_field(B_after, parameter_set, tile_shape)
        J_global = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape)
        self.assertLess(float(E_global[1][-2, 1, 1]), 1.0)
        self.assertLess(float(B_global[2][-2, 1, 1]), 1.0)
        self.assertTrue(jnp.allclose(J_global[1][1:-1, 1, 1], 1.0))

    def test_tiled_supergaussian_absorbs_field_energy_in_particle_free_1d_wave(self):
        parameter_set = _base_parameter_values(nx=40, ny=1, nz=1)
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        parameter_set["supergaussian"] = _load_supergaussian(
            [
                {"wall": "-x", "width": 8, "order": 4.0, "sigma_max": 60.0},
                {"wall": "+x", "width": 8, "order": 4.0, "sigma_max": 60.0},
            ],
            parameter_set,
            dynamic_values,
        )
        parameter_set["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (40, 1, 1)
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J = _empty_global_fields(parameter_set)

        x = parameter_set["grids"]["vertex"][0][1:-1]
        pulse = jnp.exp(-((x + 0.30) / 0.04) ** 2)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(pulse)
        Bz = Bz.at[1:-1, 1, 1].set(pulse)
        E_tiles = tile_vector_field((Ex, Ey, Ez), parameter_set, tile_shape)
        B_tiles = tile_vector_field((Bx, By, Bz), parameter_set, tile_shape)
        J_tiles = tile_vector_field(J, parameter_set, tile_shape)

        empty_particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 0, 0, 3)),
            u=jnp.zeros((1, 1, 1, 0, 0, 3)),
            active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
        )
        empty_species_config = SpeciesConfig(
            charge=jnp.zeros((0,)),
            mass=jnp.zeros((0,)),
            weight=jnp.zeros((0,)),
            update_x=jnp.zeros((0, 3), dtype=bool),
        )

        initial_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )

        def step(E_tiles, B_tiles):
            B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, None)
            E_tiles, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state)
            B_tiles, pml_state = update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state)
            return E_tiles, B_tiles

        step = jax.jit(step)
        for _ in range(60):
            E_tiles, B_tiles = step(E_tiles, B_tiles)

        final_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )
        self.assertTrue(jnp.isfinite(final_energy))
        self.assertLess(float(final_energy), 0.65 * float(initial_energy))


if __name__ == "__main__":
    unittest.main()
