import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.particles.tiled_particle_initialization import load_tiled_particles_from_toml
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.particles.species_class import particle_species


jax.config.update("jax_enable_x64", True)


class TestTiledParticleInitialization(unittest.TestCase):
    def _write_array(self, tmpdir, name, values):
        path = os.path.join(tmpdir, name)
        np.save(path, np.asarray(values, dtype=float))
        return path

    def test_to_tiled_particles_packs_existing_particle_species_list(self):
        world = {
            "Nx": 4,
            "Ny": 2,
            "Nz": 1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 4.0,
            "y_wind": 2.0,
            "z_wind": 1.0,
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

        species = particle_species(
            name="ions",
            N_particles=3,
            charge=2.0,
            mass=3.0,
            weight=4.0,
            T=1.0,
            x1=jnp.array([-1.5, 0.5, 1.5]),
            x2=jnp.array([-0.5, 0.5, 0.5]),
            x3=jnp.array([0.0, 0.0, 0.0]),
            v1=jnp.array([0.1, 0.2, 0.3]),
            v2=jnp.array([0.0, 0.0, 0.0]),
            v3=jnp.array([1.0, 2.0, 3.0]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            update_y=False,
            update_vx=False,
            active_mask=jnp.array([True, False, True]),
            dt=world["dt"],
        )

        particles = to_tiled_particles([species], world, simulation_parameters)

        self.assertIsInstance(particles, TiledParticles)
        self.assertEqual(particles.x.shape, (2, 2, 1, 1, 2, 3))
        self.assertEqual(int(jnp.sum(particles.active)), 2)

        self.assertTrue(jnp.allclose(particles.x[0, 0, 0, 0, 0], jnp.array([-1.5, -0.5, 0.0])))
        self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 0], jnp.array([0.5, 0.5, 0.0])))
        self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 1], jnp.array([1.5, 0.5, 0.0])))
        self.assertTrue(jnp.allclose(particles.u[1, 1, 0, 0, 1], jnp.array([0.3, 0.0, 3.0])))

        self.assertTrue(jnp.allclose(particles.charge[particles.x[..., 0] != 0.0], 2.0))
        self.assertTrue(jnp.allclose(particles.mass[particles.x[..., 0] != 0.0], 3.0))
        self.assertTrue(jnp.allclose(particles.weight[particles.x[..., 0] != 0.0], 4.0))

        self.assertTrue(bool(particles.update_x1[1, 1, 0, 0, 1]))
        self.assertFalse(bool(particles.update_x2[1, 1, 0, 0, 1]))
        self.assertTrue(bool(particles.update_x3[1, 1, 0, 0, 1]))
        self.assertFalse(bool(particles.update_u1[1, 1, 0, 0, 1]))
        self.assertTrue(bool(particles.update_u2[1, 1, 0, 0, 1]))
        self.assertTrue(bool(particles.update_u3[1, 1, 0, 0, 1]))

    def test_to_tiled_particles_can_allocate_inactive_capacity_headroom(self):
        world = {
            "Nx": 4,
            "Ny": 1,
            "Nz": 1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
            "particle_tile_capacity_factor": 3.0,
        }
        species = particle_species(
            name="ions",
            N_particles=3,
            charge=1.0,
            mass=1.0,
            weight=1.0,
            T=1.0,
            x1=jnp.array([-1.5, -0.5, 1.5]),
            x2=jnp.zeros(3),
            x3=jnp.zeros(3),
            v1=jnp.zeros(3),
            v2=jnp.zeros(3),
            v3=jnp.zeros(3),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        particles = to_tiled_particles([species], world, simulation_parameters)

        self.assertEqual(particles.active.shape[-1], 6)
        self.assertEqual(int(jnp.sum(particles.active)), 3)

    def test_load_tiled_particles_from_toml_uses_tile_axes_before_species(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = self._write_array(tmpdir, "x.npy", [-1.5, 0.5, 1.5])
            y_path = self._write_array(tmpdir, "y.npy", [-0.5, 0.5, 0.5])
            z_path = self._write_array(tmpdir, "z.npy", [0.0, 0.0, 0.0])
            vx_path = self._write_array(tmpdir, "vx.npy", [0.1, 0.2, 0.3])
            vy_path = self._write_array(tmpdir, "vy.npy", [0.0, 0.0, 0.0])
            vz_path = self._write_array(tmpdir, "vz.npy", [1.0, 2.0, 3.0])

            world = {
                "Nx": 4,
                "Ny": 2,
                "Nz": 1,
                "dx": 1.0,
                "dy": 1.0,
                "dz": 1.0,
                "dt": 0.1,
                "x_wind": 4.0,
                "y_wind": 2.0,
                "z_wind": 1.0,
            }
            constants = {"kb": 1.0, "eps": 1.0}
            simulation_parameters = {
                "ds_per_debye": None,
                "shape_factor": 1,
                "particle_tile_nx": 2,
                "particle_tile_ny": 1,
                "particle_tile_nz": 1,
            }
            config = {
                "particle1": {
                    "name": "electrons",
                    "N_particles": 3,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 4.0,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": y_path,
                    "initial_z": z_path,
                    "initial_vx": vx_path,
                    "initial_vy": vy_path,
                    "initial_vz": vz_path,
                }
            }

            particles = load_tiled_particles_from_toml(config, simulation_parameters, world, constants)

            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(particles.x.shape, (2, 2, 1, 1, 2, 3))
            self.assertEqual(particles.u.shape, (2, 2, 1, 1, 2, 3))
            self.assertEqual(particles.charge.shape, (2, 2, 1, 1, 2))

            self.assertTrue(particles.active[0, 0, 0, 0, 0])
            self.assertTrue(particles.active[1, 1, 0, 0, 0])
            self.assertTrue(particles.active[1, 1, 0, 0, 1])
            self.assertEqual(int(jnp.sum(particles.active)), 3)

            self.assertTrue(jnp.allclose(particles.x[0, 0, 0, 0, 0], jnp.array([-1.5, -0.5, 0.0])))
            self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 0], jnp.array([0.5, 0.5, 0.0])))
            self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 1], jnp.array([1.5, 0.5, 0.0])))
            self.assertTrue(jnp.allclose(particles.u[1, 1, 0, 0, 1], jnp.array([0.3, 0.0, 3.0])))

            self.assertTrue(jnp.allclose(particles.charge[particles.active], -1.0))
            self.assertTrue(jnp.allclose(particles.mass[particles.active], 2.0))
            self.assertTrue(jnp.allclose(particles.weight[particles.active], 4.0))

    def test_load_tiled_particles_from_toml_maps_update_flags_to_active_particles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = self._write_array(tmpdir, "x.npy", [0.0])
            y_path = self._write_array(tmpdir, "y.npy", [0.0])
            z_path = self._write_array(tmpdir, "z.npy", [0.0])

            world = {
                "Nx": 1,
                "Ny": 1,
                "Nz": 1,
                "dx": 1.0,
                "dy": 1.0,
                "dz": 1.0,
                "dt": 0.1,
                "x_wind": 1.0,
                "y_wind": 1.0,
                "z_wind": 1.0,
            }
            constants = {"kb": 1.0, "eps": 1.0}
            simulation_parameters = {
                "ds_per_debye": None,
                "shape_factor": 1,
                "particle_tile_nx": 1,
                "particle_tile_ny": 1,
                "particle_tile_nz": 1,
            }
            config = {
                "particle1": {
                    "name": "partly fixed",
                    "N_particles": 1,
                    "charge": 1.0,
                    "mass": 1.0,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": y_path,
                    "initial_z": z_path,
                    "initial_vx": 0.0,
                    "initial_vy": 0.0,
                    "initial_vz": 0.0,
                    "update_pos": True,
                    "update_x": True,
                    "update_y": False,
                    "update_z": True,
                    "update_v": True,
                    "update_vx": False,
                    "update_vy": True,
                    "update_vz": False,
                }
            }

            particles = load_tiled_particles_from_toml(config, simulation_parameters, world, constants)

            self.assertTrue(bool(particles.update_x1[0, 0, 0, 0, 0]))
            self.assertFalse(bool(particles.update_x2[0, 0, 0, 0, 0]))
            self.assertTrue(bool(particles.update_x3[0, 0, 0, 0, 0]))
            self.assertFalse(bool(particles.update_u1[0, 0, 0, 0, 0]))
            self.assertTrue(bool(particles.update_u2[0, 0, 0, 0, 0]))
            self.assertFalse(bool(particles.update_u3[0, 0, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
