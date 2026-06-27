import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.esirkepov_tiled import tiled_esirkepov_current
from PyPIC3D.initialization import CURRENT_ESIRKEPOV, initialize_simulation
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.solvers.yee_tiled import (
    assemble_tiled_vector_field,
    empty_tiled_vector_field,
    tile_vector_field,
    update_tiled_E,
)

jax.config.update("jax_enable_x64", True)


class TestTiledEsirkepovCurrent(unittest.TestCase):
    def _build_world(self, Nx=8, Ny=1, Nz=1, dt=0.05):
        x_wind, y_wind, z_wind = 4.0, 1.0, 1.0
        world = {
            "dx": x_wind / Nx,
            "dy": y_wind / Ny,
            "dz": z_wind / Nz,
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "x_wind": x_wind,
            "y_wind": y_wind,
            "z_wind": z_wind,
            "dt": dt,
            "shape_factor": 1,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "current_calculation": CURRENT_ESIRKEPOV,
            "current_guard_cells": 2,
        }
        center_grid = (
            jnp.linspace(-x_wind / 2 - world["dx"] / 2, x_wind / 2 + world["dx"] / 2, Nx + 2),
            jnp.linspace(-y_wind / 2 - world["dy"] / 2, y_wind / 2 + world["dy"] / 2, Ny + 2),
            jnp.linspace(-z_wind / 2 - world["dz"] / 2, z_wind / 2 + world["dz"] / 2, Nz + 2),
        )
        world["grids"] = {"center": center_grid, "vertex": center_grid}
        return world

    def _empty_J(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

    def _species(self, world, x1):
        return particle_species(
            name="electrons",
            N_particles=3,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=x1,
            x2=jnp.zeros_like(x1),
            x3=jnp.zeros_like(x1),
            v1=jnp.array([0.08, -0.06, 0.05]),
            v2=jnp.zeros_like(x1),
            v3=jnp.zeros_like(x1),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

    def test_empty_tiled_vector_field_can_allocate_two_current_guard_cells(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        tile_shape = (2, 1, 1)

        J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=2)

        self.assertEqual(J_tiles[0].shape, (4, 1, 1, 6, 5, 5))
        self.assertTrue(jnp.allclose(J_tiles[0], 0.0))

    def test_two_guard_current_tiles_assemble_to_one_guard_global_current(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        tile_shape = (2, 1, 1)
        J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=2)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[1, 0, 0, 2, 2, 2].set(3.0)

        assembled = assemble_tiled_vector_field((Jx, Jy, Jz), world, tile_shape, num_guard_cells=2)

        self.assertEqual(assembled[0].shape, (10, 3, 3))
        self.assertEqual(float(assembled[0][3, 1, 1]), 3.0)

    def test_update_tiled_E_reads_two_guard_current_interior(self):
        world = self._build_world(Nx=4, Ny=1, Nz=1, dt=0.25)
        constants = {"C": 1.0, "eps": 2.0}
        tile_shape = (2, 1, 1)
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        E_tiles = tile_vector_field(zeros, world, tile_shape)
        B_tiles = tile_vector_field(zeros, world, tile_shape)
        J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=2)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 2:-2, 2:-2, 2:-2].set(4.0)

        E_after = update_tiled_E(E_tiles, B_tiles, (Jx, Jy, Jz), world, constants, None, tile_shape)

        self.assertTrue(jnp.allclose(E_after[0][:, :, :, 1:-1, 1:-1, 1:-1], -0.5))

    def test_tiled_esirkepov_matches_global_1d_periodic_current(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05)
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        x_old = jnp.array([-1.10, -0.10, 1.05])
        old_species = self._species(world, x_old)
        x_new = x_old + old_species.v1 * world["dt"]
        new_species = self._species(world, x_new)

        tiled_particles = to_tiled_particles([old_species], world, simulation_parameters)
        J_reference = Esirkepov_current([new_species], self._empty_J(world), constants, world)
        J_tiles = tiled_esirkepov_current(
            tiled_particles,
            empty_tiled_vector_field(world, tile_shape, num_guard_cells=2),
            constants,
            world,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=2)

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )

    def test_initialize_tiled_yee_esirkepov_uses_two_guard_current_tiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov init smoke",
                    "output_dir": tmpdir,
                    "solver": "tiled_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "none",
                    "fast_backend": "default",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }
            config_path = os.path.join(tmpdir, "tiled_yee_esirkepov.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            _loop, particles, fields, world, *_rest = initialize_simulation(toml.load(config_path))
            E_tiles, _B_tiles, J_tiles, *_ = fields

            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(E_tiles[0].shape[-3:], (4, 3, 3))
            self.assertEqual(J_tiles[0].shape[-3:], (6, 5, 5))
            self.assertEqual(int(world["current_guard_cells"]), 2)
            self.assertEqual(int(world["current_calculation"]), CURRENT_ESIRKEPOV)

    def test_tiled_yee_esirkepov_loop_advances_particles_after_deposition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_initial = np.array([-1.5, -0.5, 0.5, 1.5])
            vx_initial = np.array([0.10, -0.05, 0.07, -0.02])
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, x_initial)
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, vx_initial)

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov step smoke",
                    "output_dir": tmpdir,
                    "solver": "tiled_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "none",
                    "fast_backend": "default",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }
            config_path = os.path.join(tmpdir, "tiled_yee_esirkepov_step.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, _simulation_parameters, constants, _plotting_parameters, _plasma_parameters, \
                solver, _electrostatic, _verbose, _GPUs, _Nt, curl_func, J_func, relativistic, particle_pusher = initialize_simulation(toml.load(config_path))

            particles, fields = loop(
                particles,
                fields,
                world,
                constants,
                curl_func,
                J_func,
                solver,
                tile_shape=tuple(int(width) for width in world["tile_shape"]),
                relativistic=relativistic,
                particle_pusher=particle_pusher,
            )

            active_x = np.asarray(particles.x[..., 0][particles.active])
            expected_x = np.sort(x_initial + vx_initial * float(world["dt"]))
            self.assertTrue(np.allclose(np.sort(active_x), expected_x, rtol=1.0e-12, atol=1.0e-12))
            self.assertEqual(fields[2][0].shape[-3:], (6, 5, 5))
            self.assertFalse(bool(fields[-1]))


if __name__ == "__main__":
    unittest.main()
