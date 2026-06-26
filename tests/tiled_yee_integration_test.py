import functools
import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.evolve import time_loop_electrodynamic
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.direct_deposition_tiled import direct_J_from_tiled_particles
from PyPIC3D.electrodynamic_tiled import time_loop_electrodynamic_tiled
from PyPIC3D.initialization import initialize_simulation, validate_field_solver
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.solvers.yee_tiled import assemble_tiled_vector_field, tile_vector_field
from PyPIC3D.utils import build_yee_grid, compute_energy


jax.config.update("jax_enable_x64", True)


def unused_curl(Ex, Ey, Ez):
    return None


class TestTiledYeeIntegration(unittest.TestCase):
    def _build_world(self):
        world = {
            "Nx": 8,
            "Ny": 1,
            "Nz": 1,
            "dx": 0.5,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.01,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return world

    def _empty_fields(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        zero = jnp.zeros(shape)
        E = (zero, zero, zero)
        B = (zero, zero, zero)
        J = (zero, zero, zero)
        external_fields = (E, B)
        return E, B, J, zero, zero, external_fields, None

    def _species(self, world):
        return particle_species(
            name="electrons",
            N_particles=4,
            charge=-1.0,
            mass=2.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.5, -0.5, 0.5, 1.5]),
            x2=jnp.zeros(4),
            x3=jnp.zeros(4),
            v1=jnp.array([0.10, -0.05, 0.07, -0.02]),
            v2=jnp.zeros(4),
            v3=jnp.zeros(4),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

    def _active_rows(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def test_validate_field_solver_accepts_tiled_yee(self):
        validate_field_solver("tiled_yee")

    def test_initialize_simulation_uses_tiled_particles_and_fields_for_tiled_yee(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee init smoke",
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
                    "current_calculation": "j_from_rhov",
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
            config_path = os.path.join(tmpdir, "tiled_yee.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic_tiled)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(tuple(world["tile_shape"]), (2, 1, 1))

    def test_initialize_simulation_accepts_tiled_yee_digital_current_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee digital filter init smoke",
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
                    "current_calculation": "j_from_rhov",
                    "filter_j": "digital",
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
            config_path = os.path.join(tmpdir, "tiled_yee_digital.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, simulation_parameters, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic_tiled)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(simulation_parameters["filter_j"], "digital")

    def test_initialize_simulation_accepts_tiled_yee_conducting_field_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee conducting init smoke",
                    "output_dir": tmpdir,
                    "solver": "tiled_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "x_bc": "conducting",
                    "y_bc": "conducting",
                    "z_bc": "conducting",
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "j_from_rhov",
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
            config_path = os.path.join(tmpdir, "tiled_yee_conducting.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic_tiled)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(world["boundary_conditions"], {"x": 1, "y": 1, "z": 1})

    def test_tiled_yee_step_matches_standard_periodic_yee_step(self):
        world = self._build_world()
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = self._species(world)
        reference_species = self._species(world)
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)

        reference_particles, reference_fields = time_loop_electrodynamic(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            world,
            constants,
            unused_curl,
            J_func=lambda particles, J, constants, world: J_from_rhov(particles, J, constants, world, filter="none"),
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

        tiled_particles = to_tiled_particles([species], world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], world, tile_shape),
                tile_vector_field(external_fields[1], world, tile_shape),
            ),
            None,
        )
        tiled_particles, tiled_fields = time_loop_electrodynamic_tiled(
            tiled_particles,
            tiled_fields,
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="tiled_yee",
            relativistic=False,
            particle_pusher="boris",
        )

        E_tiles, B_tiles, J_tiles, *_ = tiled_fields
        E_from_tiles = assemble_tiled_vector_field(E_tiles, world, tile_shape)
        B_from_tiles = assemble_tiled_vector_field(B_tiles, world, tile_shape)
        reference_E, reference_B, reference_J, *_ = reference_fields

        flat_x = jnp.stack(reference_particles[0].get_forward_position(), axis=1)
        flat_u = jnp.stack(reference_particles[0].get_velocity(), axis=1)
        flat_order = jnp.lexsort((flat_x[:, 2], flat_x[:, 1], flat_x[:, 0]))
        tiled_x, tiled_u = self._active_rows(tiled_particles)

        self.assertTrue(jnp.allclose(tiled_x, flat_x[flat_order], rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(tiled_u, flat_u[flat_order], rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(reference_E, E_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(reference_B, B_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(reference_J, assemble_tiled_vector_field(J_tiles, world, tile_shape)):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

        reference_energy = compute_energy(reference_particles, reference_E, reference_B, world, constants)
        tiled_energy = compute_energy(tiled_particles, E_tiles, B_tiles, world, constants)
        for reference, tiled in zip(reference_energy, tiled_energy):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_yee_step_uses_digital_filtered_current(self):
        world = self._build_world()
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.6}
        tile_shape = (2, 1, 1)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = self._species(world)
        reference_species = self._species(world)
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)

        _, reference_fields = time_loop_electrodynamic(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            world,
            constants,
            unused_curl,
            J_func=lambda particles, J, constants, world: J_from_rhov(particles, J, constants, world, filter="digital"),
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

        tiled_particles = to_tiled_particles([species], world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], world, tile_shape),
                tile_vector_field(external_fields[1], world, tile_shape),
            ),
            None,
        )
        _, tiled_fields = time_loop_electrodynamic_tiled(
            tiled_particles,
            tiled_fields,
            world,
            constants,
            unused_curl,
            J_func=functools.partial(direct_J_from_tiled_particles, filter="digital"),
            solver="tiled_yee",
            relativistic=False,
            particle_pusher="boris",
        )

        _, _, reference_J, *_ = reference_fields
        _, _, J_tiles, *_ = tiled_fields
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape)

        for reference, tiled in zip(reference_J, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
