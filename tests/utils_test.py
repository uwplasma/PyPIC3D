import unittest
import os
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.utils import (
    print_stats, build_yee_grid, build_collocated_grid, check_stability,
    particle_sanity_check, load_external_fields_from_toml, add_external_fields,
    compute_energy,
)

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

if __name__ == '__main__':
    unittest.main()
