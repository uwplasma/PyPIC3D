import tempfile
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import (
    PML_WALLS,
    build_pml_metadata,
    build_pml_profiles,
    initialize_pml_state,
    parse_pml_config,
)
from PyPIC3D.initialization import initialize_fields, initialize_simulation
from PyPIC3D.solvers.first_order_yee import update_B, update_E
from PyPIC3D.utils import build_yee_grid, compute_energy

jax.config.update("jax_enable_x64", True)


def _base_world(nx=24, ny=1, nz=1):
    world = {
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "dx": 1.0 / nx,
        "dy": 1.0,
        "dz": 1.0,
        "dt": 0.5 / nx,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }
    center_grid, vertex_grid = build_yee_grid(world)
    world["grids"] = {"center": center_grid, "vertex": vertex_grid}
    return world


def _empty_config(tmpdir, solver="fdtd", electrostatic=False, pml=None):
    sim = {
        "name": "pml init test",
        "output_dir": tmpdir,
        "solver": solver,
        "electrostatic": electrostatic,
        "Nx": 8,
        "Ny": 1,
        "Nz": 1,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "Nt": 1,
        "dt": 1e-10,
        "fast_backend": "default",
    }
    config = {"simulation_parameters": sim, "plotting": {"plotting": False}}
    if pml is not None:
        config["pml"] = pml
    return config


class TestPMLConfiguration(unittest.TestCase):
    def test_parse_pml_config_accepts_all_six_walls(self):
        raw = [
            {"wall": wall, "thickness": 2, "order": 3.0, "target_reflection": 1e-8}
            for wall in PML_WALLS
        ]
        parsed = parse_pml_config(raw, _base_world(nx=8, ny=8, nz=8), {"C": 3.0})

        self.assertEqual([entry["wall"] for entry in parsed], PML_WALLS)
        self.assertTrue(all(entry["sigma_max"] > 0.0 for entry in parsed))

    def test_parse_pml_config_rejects_invalid_duplicate_and_oversized_walls(self):
        world = _base_world(nx=8, ny=1, nz=1)

        with self.assertRaisesRegex(ValueError, "Invalid PML wall"):
            parse_pml_config([{"wall": "x+", "thickness": 2}], world, {"C": 3.0})

        with self.assertRaisesRegex(ValueError, "Duplicate PML wall"):
            parse_pml_config(
                [{"wall": "+x", "thickness": 2}, {"wall": "+x", "thickness": 2}],
                world,
                {"C": 3.0},
            )

        with self.assertRaisesRegex(ValueError, "exceeds active cells"):
            parse_pml_config([{"wall": "+x", "thickness": 9}], world, {"C": 3.0})

    def test_build_pml_profiles_ramp_only_on_requested_side(self):
        world = _base_world(nx=8, ny=1, nz=1)
        pml = parse_pml_config(
            [{"wall": "+x", "thickness": 3, "order": 2.0, "sigma_max": 9.0}],
            world,
            {"C": 3.0},
        )

        profiles = build_pml_profiles(world, pml)
        sigma_x = profiles["sigma_x"]

        self.assertEqual(sigma_x.shape, (10, 3, 3))
        self.assertTrue(jnp.allclose(sigma_x[1:5, :, :], 0.0))
        self.assertTrue(jnp.all(sigma_x[-4:-1, :, :] > 0.0))
        self.assertTrue(float(sigma_x[-2, 1, 1]) > float(sigma_x[-4, 1, 1]))
        self.assertTrue(jnp.allclose(profiles["sigma_y"], 0.0))
        self.assertTrue(jnp.allclose(profiles["sigma_z"], 0.0))


class TestPMLInitialization(unittest.TestCase):
    def test_initialize_simulation_rejects_pml_for_non_fdtd_solvers(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "PML is only supported"):
                initialize_simulation(_empty_config(tmpdir, solver="spectral", pml=pml))

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "PML is only supported"):
                initialize_simulation(_empty_config(tmpdir, solver="vector_potential", pml=pml))

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "PML is only supported"):
                initialize_simulation(_empty_config(tmpdir, electrostatic=True, pml=pml))

    def test_initialize_simulation_appends_pml_state_for_fdtd(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir, pml=pml))

        fields = result[2]
        world = result[3]
        self.assertEqual(len(fields), 7)
        self.assertIn("pml", world)
        self.assertTrue(world["pml"]["active"])


class TestPMLFDTDBehavior(unittest.TestCase):
    def test_no_pml_state_preserves_fdtd_update_signature_and_result(self):
        world = _base_world(nx=4, ny=4, nz=4)
        constants = {"C": 2.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        E, B, J, _, _ = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        B = (B[0], B[1], B[2].at[1:-1, 1:-1, 1:-1].set(1.0))

        E_after = update_E(E, B, J, world, constants, lambda *args: None)
        B_after = update_B(E_after, B, world, constants, lambda *args: None)

        self.assertEqual(len(E_after), 3)
        self.assertEqual(len(B_after), 3)

    def test_pml_absorbs_field_energy_in_particle_free_1d_wave(self):
        world = _base_world(nx=80, ny=1, nz=1)
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        world["pml"] = build_pml_metadata(
            [
                {"wall": "-x", "thickness": 12, "order": 3.0, "sigma_max": 40.0},
                {"wall": "+x", "thickness": 12, "order": 3.0, "sigma_max": 40.0},
            ],
            world,
            constants,
        )
        pml_state = initialize_pml_state(world)
        E, B, J, _, _ = initialize_fields(world["Nx"], world["Ny"], world["Nz"])

        x = world["grids"]["vertex"][0][1:-1]
        pulse = jnp.exp(-((x + 0.25) / 0.06) ** 2)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(pulse)
        Bz = Bz.at[1:-1, 1, 1].set(pulse)
        E = (Ex, Ey, Ez)
        B = (Bx, By, Bz)

        initial_energy = sum(compute_energy([], E, B, world, constants)[:2])
        for _ in range(180):
            E, pml_state = update_E(E, B, J, world, constants, lambda *args: None, pml_state)
            B, pml_state = update_B(E, B, world, constants, lambda *args: None, pml_state)

        final_energy = sum(compute_energy([], E, B, world, constants)[:2])
        self.assertTrue(jnp.isfinite(final_energy))
        self.assertLess(float(final_energy), 0.35 * float(initial_energy))


if __name__ == "__main__":
    unittest.main()
