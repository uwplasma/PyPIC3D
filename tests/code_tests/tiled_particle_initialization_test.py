import os
import inspect
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.particles.particle_initialization import load_particles_from_toml
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.tests.tiled_particle_fixtures import particle_species, to_tiled_particles


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

        particles, species_config = to_tiled_particles([species], world, simulation_parameters)

        self.assertIsInstance(particles, TiledParticles)
        self.assertIsInstance(species_config, SpeciesConfig)
        self.assertEqual(particles.x.shape, (2, 2, 1, 1, 2, 3))
        self.assertEqual(int(jnp.sum(particles.active)), 2)
        self.assertFalse(hasattr(particles, "charge"))
        self.assertFalse(hasattr(particles, "mass"))
        self.assertFalse(hasattr(particles, "weight"))

        self.assertTrue(jnp.allclose(particles.x[0, 0, 0, 0, 0], jnp.array([-1.5, -0.5, 0.0])))
        self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 0], jnp.array([0.5, 0.5, 0.0])))
        self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 1], jnp.array([1.5, 0.5, 0.0])))
        self.assertTrue(jnp.allclose(particles.u[1, 1, 0, 0, 1], jnp.array([0.3, 0.0, 3.0])))

        self.assertEqual(species_config.charge.shape, (1,))
        self.assertEqual(species_config.mass.shape, (1,))
        self.assertEqual(species_config.weight.shape, (1,))
        self.assertEqual(species_config.update_x.shape, (1, 3))
        self.assertEqual(species_config.update_u.shape, (1, 3))
        self.assertTrue(jnp.allclose(species_config.charge, jnp.array([2.0])))
        self.assertTrue(jnp.allclose(species_config.mass, jnp.array([3.0])))
        self.assertTrue(jnp.allclose(species_config.weight, jnp.array([4.0])))

        self.assertTrue(bool(species_config.update_x[0, 0]))
        self.assertFalse(bool(species_config.update_x[0, 1]))
        self.assertTrue(bool(species_config.update_x[0, 2]))
        self.assertFalse(bool(species_config.update_u[0, 0]))
        self.assertTrue(bool(species_config.update_u[0, 1]))
        self.assertTrue(bool(species_config.update_u[0, 2]))

    def test_species_metadata_is_not_slot_shaped(self):
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
            "particle_tile_nx": 1,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        species = [
            particle_species(
                name="electrons",
                N_particles=2,
                charge=-1.0,
                mass=2.0,
                weight=0.5,
                T=1.0,
                x1=jnp.array([-1.5, -0.5]),
                x2=jnp.zeros(2),
                x3=jnp.zeros(2),
                v1=jnp.zeros(2),
                v2=jnp.zeros(2),
                v3=jnp.zeros(2),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                dt=world["dt"],
            ),
            particle_species(
                name="ions",
                N_particles=2,
                charge=2.0,
                mass=5.0,
                weight=0.25,
                T=1.0,
                x1=jnp.array([0.5, 1.5]),
                x2=jnp.zeros(2),
                x3=jnp.zeros(2),
                v1=jnp.zeros(2),
                v2=jnp.zeros(2),
                v3=jnp.zeros(2),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                dt=world["dt"],
            ),
        ]

        particles, species_config = to_tiled_particles(species, world, simulation_parameters)

        self.assertEqual(particles.x.shape[:4], (4, 1, 1, 2))
        self.assertEqual(species_config.charge.shape, (2,))
        self.assertEqual(species_config.mass.shape, (2,))
        self.assertEqual(species_config.weight.shape, (2,))
        self.assertEqual(species_config.update_x.shape, (2, 3))
        self.assertEqual(species_config.update_u.shape, (2, 3))

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

        particles, species_config = to_tiled_particles([species], world, simulation_parameters)

        self.assertEqual(particles.active.shape[-1], 6)
        self.assertEqual(int(jnp.sum(particles.active)), 3)
        self.assertEqual(species_config.charge.shape, (1,))

    def test_load_particles_from_toml_uses_tile_axes_before_species(self):
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
                "tile_shape": (2, 1, 1),
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

            particles, species_config, species_names, metadata = load_particles_from_toml(config, simulation_parameters, world, constants)

            self.assertIsInstance(particles, TiledParticles)
            self.assertIsInstance(species_config, SpeciesConfig)
            self.assertEqual(species_names, ("electrons",))
            self.assertEqual(metadata[0]["name"], "electrons")
            self.assertEqual(particles.x.shape, (2, 2, 1, 1, 2, 3))
            self.assertEqual(particles.u.shape, (2, 2, 1, 1, 2, 3))

            self.assertTrue(particles.active[0, 0, 0, 0, 0])
            self.assertTrue(particles.active[1, 1, 0, 0, 0])
            self.assertTrue(particles.active[1, 1, 0, 0, 1])
            self.assertEqual(int(jnp.sum(particles.active)), 3)

            self.assertTrue(jnp.allclose(particles.x[0, 0, 0, 0, 0], jnp.array([-1.5, -0.5, 0.0])))
            self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 0], jnp.array([0.5, 0.5, 0.0])))
            self.assertTrue(jnp.allclose(particles.x[1, 1, 0, 0, 1], jnp.array([1.5, 0.5, 0.0])))
            self.assertTrue(jnp.allclose(particles.u[1, 1, 0, 0, 1], jnp.array([0.3, 0.0, 3.0])))

            self.assertTrue(jnp.allclose(species_config.charge, jnp.array([-1.0])))
            self.assertTrue(jnp.allclose(species_config.mass, jnp.array([2.0])))
            self.assertTrue(jnp.allclose(species_config.weight, jnp.array([4.0])))

    def test_load_particles_from_toml_preserves_interleaved_tile_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            electron_x_path = self._write_array(tmpdir, "electron_x.npy", [-1.5, 0.5, -0.5, 1.5])
            ion_x_path = self._write_array(tmpdir, "ion_x.npy", [1.5, -1.5, 0.5, -0.5])
            y_path = self._write_array(tmpdir, "y.npy", [0.0, 0.0, 0.0, 0.0])
            z_path = self._write_array(tmpdir, "z.npy", [0.0, 0.0, 0.0, 0.0])
            electron_vx_path = self._write_array(tmpdir, "electron_vx.npy", [10.0, 20.0, 30.0, 40.0])
            ion_vx_path = self._write_array(tmpdir, "ion_vx.npy", [100.0, 200.0, 300.0, 400.0])
            vy_path = self._write_array(tmpdir, "vy.npy", [0.0, 0.0, 0.0, 0.0])
            vz_path = self._write_array(tmpdir, "vz.npy", [1.0, 2.0, 3.0, 4.0])

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
                "tile_shape": (2, 1, 1),
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
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 4.0,
                    "temperature": 1.0,
                    "initial_x": electron_x_path,
                    "initial_y": y_path,
                    "initial_z": z_path,
                    "initial_vx": electron_vx_path,
                    "initial_vy": vy_path,
                    "initial_vz": vz_path,
                },
                "particle2": {
                    "name": "ions",
                    "N_particles": 4,
                    "charge": 1.0,
                    "mass": 3.0,
                    "weight": 5.0,
                    "temperature": 1.0,
                    "initial_x": ion_x_path,
                    "initial_y": y_path,
                    "initial_z": z_path,
                    "initial_vx": ion_vx_path,
                    "initial_vy": vy_path,
                    "initial_vz": vz_path,
                },
            }

            particles, species_config, species_names, metadata = load_particles_from_toml(config, simulation_parameters, world, constants)

            self.assertEqual(species_names, ("electrons", "ions"))
            self.assertEqual(tuple(item["name"] for item in metadata), ("electrons", "ions"))
            self.assertEqual(particles.x.shape, (2, 1, 1, 2, 2, 3))
            self.assertEqual(int(jnp.sum(particles.active)), 8)

            self.assertTrue(jnp.allclose(particles.x[0, 0, 0, 0, :, 0], jnp.array([-1.5, -0.5])))
            self.assertTrue(jnp.allclose(particles.u[0, 0, 0, 0, :, 0], jnp.array([10.0, 30.0])))
            self.assertTrue(jnp.allclose(particles.x[1, 0, 0, 0, :, 0], jnp.array([0.5, 1.5])))
            self.assertTrue(jnp.allclose(particles.u[1, 0, 0, 0, :, 0], jnp.array([20.0, 40.0])))

            self.assertTrue(jnp.allclose(particles.x[0, 0, 0, 1, :, 0], jnp.array([-1.5, -0.5])))
            self.assertTrue(jnp.allclose(particles.u[0, 0, 0, 1, :, 0], jnp.array([200.0, 400.0])))
            self.assertTrue(jnp.allclose(particles.x[1, 0, 0, 1, :, 0], jnp.array([1.5, 0.5])))
            self.assertTrue(jnp.allclose(particles.u[1, 0, 0, 1, :, 0], jnp.array([100.0, 300.0])))

            self.assertTrue(jnp.allclose(species_config.charge, jnp.array([-1.0, 1.0])))
            self.assertTrue(jnp.allclose(species_config.mass, jnp.array([2.0, 3.0])))
            self.assertTrue(jnp.allclose(species_config.weight, jnp.array([4.0, 5.0])))

    def test_load_particles_from_toml_does_not_scatter_each_particle_with_jax_at(self):
        source = inspect.getsource(load_particles_from_toml)

        self.assertNotIn(".at[index].set", source)

    def test_load_particles_from_toml_maps_update_flags_to_active_particles(self):
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
                "tile_shape": (1, 1, 1),
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

            particles, species_config, species_names, metadata = load_particles_from_toml(config, simulation_parameters, world, constants)

            self.assertTrue(bool(species_config.update_x[0, 0]))
            self.assertFalse(bool(species_config.update_x[0, 1]))
            self.assertTrue(bool(species_config.update_x[0, 2]))
            self.assertFalse(bool(species_config.update_u[0, 0]))
            self.assertTrue(bool(species_config.update_u[0, 1]))
            self.assertFalse(bool(species_config.update_u[0, 2]))


if __name__ == "__main__":
    unittest.main()
