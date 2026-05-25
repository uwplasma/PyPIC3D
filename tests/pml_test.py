import tempfile
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import (
    PML_WALLS,
    build_pml,
    initialize_pml_state,
    load_pml_from_toml,
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
    def test_load_pml_from_toml_accepts_all_six_walls(self):
        raw = [
            {"wall": wall, "thickness": 2, "order": 3.0, "target_reflection": 1e-8}
            for wall in PML_WALLS
        ]

        active, pml_x, pml_y, pml_z, profiles = load_pml_from_toml(
            raw,
            _base_world(nx=8, ny=8, nz=8),
            {"C": 3.0},
        )
        sigma_x, sigma_y, sigma_z = profiles

        self.assertTrue(active)
        self.assertTrue(pml_x)
        self.assertTrue(pml_y)
        self.assertTrue(pml_z)
        self.assertEqual(sigma_x.shape, (10, 10, 10))
        self.assertTrue(jnp.any(sigma_x > 0.0))
        self.assertTrue(jnp.any(sigma_y > 0.0))
        self.assertTrue(jnp.any(sigma_z > 0.0))

    def test_load_pml_from_toml_rejects_invalid_duplicate_and_oversized_walls(self):
        world = _base_world(nx=8, ny=1, nz=1)

        with self.assertRaisesRegex(ValueError, "Invalid PML wall"):
            load_pml_from_toml([{"wall": "x+", "thickness": 2}], world, {"C": 3.0})

        with self.assertRaisesRegex(ValueError, "Duplicate PML wall"):
            load_pml_from_toml(
                [{"wall": "+x", "thickness": 2}, {"wall": "+x", "thickness": 2}],
                world,
                {"C": 3.0},
            )

        with self.assertRaisesRegex(ValueError, "exceeds active cells"):
            load_pml_from_toml([{"wall": "+x", "thickness": 9}], world, {"C": 3.0})

    def test_build_pml_ramp_only_on_requested_side(self):
        world = _base_world(nx=8, ny=1, nz=1)
        pml_layers = (("+x", "x", 3, 2.0, 9.0),)

        sigma_x, sigma_y, sigma_z = build_pml(world, pml_layers)

        self.assertEqual(sigma_x.shape, (10, 3, 3))
        self.assertTrue(jnp.allclose(sigma_x[1:5, :, :], 0.0))
        self.assertTrue(jnp.all(sigma_x[-4:-1, :, :] > 0.0))
        self.assertTrue(float(sigma_x[-2, 1, 1]) > float(sigma_x[-4, 1, 1]))
        self.assertTrue(jnp.allclose(sigma_y, 0.0))
        self.assertTrue(jnp.allclose(sigma_z, 0.0))

    def test_build_pml_ramps_from_interior_interface_to_outer_wall(self):
        world = _base_world(nx=8, ny=1, nz=1)
        pml_layers = (
            ("-x", "x", 3, 2.0, 9.0),
            ("+x", "x", 3, 2.0, 9.0),
        )

        sigma_x, _, _ = build_pml(world, pml_layers)
        sigma_x = sigma_x[:, 1, 1]

        self.assertGreater(float(sigma_x[1]), float(sigma_x[2]))
        self.assertGreater(float(sigma_x[2]), float(sigma_x[3]))
        self.assertGreater(float(sigma_x[-2]), float(sigma_x[-3]))
        self.assertGreater(float(sigma_x[-3]), float(sigma_x[-4]))


class TestPMLInitialization(unittest.TestCase):
    def test_initialize_pml_state_uses_physical_interior_shape(self):
        world = _base_world(nx=8, ny=4, nz=2)

        pml_state = initialize_pml_state(world)
        e_memory, b_memory = pml_state

        self.assertEqual(len(e_memory), 6)
        self.assertEqual(len(b_memory), 6)
        for memory in e_memory:
            self.assertEqual(memory.shape, (8, 4, 2))
        for memory in b_memory:
            self.assertEqual(memory.shape, (8, 4, 2))

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
        active, pml_x, pml_y, pml_z, _ = world["pml"]
        self.assertEqual(len(fields), 7)
        self.assertIn("pml", world)
        self.assertTrue(active)
        self.assertTrue(pml_x)
        self.assertFalse(pml_y)
        self.assertFalse(pml_z)

    def test_initialize_simulation_uses_none_pml_state_without_pml_for_fdtd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir))

        fields = result[2]
        world = result[3]
        active, pml_x, pml_y, pml_z, _ = world["pml"]

        self.assertEqual(len(fields), 7)
        self.assertIsNone(fields[-1])
        self.assertFalse(active)
        self.assertFalse(pml_x)
        self.assertFalse(pml_y)
        self.assertFalse(pml_z)


class TestPMLFDTDBehavior(unittest.TestCase):
    def test_no_pml_state_returns_field_and_none_state(self):
        world = _base_world(nx=4, ny=4, nz=4)
        constants = {"C": 2.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        E, B, J, _, _ = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        B = (B[0], B[1], B[2].at[1:-1, 1:-1, 1:-1].set(1.0))

        E_after, pml_state = update_E(E, B, J, world, constants, lambda *args: None)
        B_after, pml_state = update_B(E_after, B, world, constants, lambda *args: None, pml_state)

        self.assertEqual(len(E_after), 3)
        self.assertEqual(len(B_after), 3)
        self.assertIsNone(pml_state)

    def test_pml_absorbs_field_energy_in_particle_free_1d_wave(self):
        world = _base_world(nx=80, ny=1, nz=1)
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        world["pml"] = load_pml_from_toml(
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
