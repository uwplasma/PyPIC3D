import functools
import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.evolve import time_loop_electrodynamic
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.current_methods import CURRENT_ESIRKEPOV, CURRENT_J_FROM_RHOV
from PyPIC3D.deposition.direct_deposition_tiled import direct_J_from_tiled_particles
from PyPIC3D.deposition.esirkepov_tiled import tiled_esirkepov_current
from PyPIC3D.electrodynamic_tiled import time_loop_electrodynamic_tiled
from PyPIC3D.initialization import initialize_simulation, validate_field_solver
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.solvers.yee_tiled import assemble_tiled_vector_field, empty_tiled_vector_field, tile_vector_field
from PyPIC3D.utils import build_yee_grid, compute_energy
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions.boundaryconditions import update_ghost_cells


jax.config.update("jax_enable_x64", True)

ROUND_OFF_RTOL = 1.0e-11
ROUND_OFF_ATOL = 1.0e-11


def unused_curl(Ex, Ey, Ez):
    return None


class TestTiledYeeIntegration(unittest.TestCase):
    def _build_world(self, boundary_conditions=None):
        if boundary_conditions is None:
            boundary_conditions = {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}
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
            "guard_cells": 1,
            "boundary_conditions": boundary_conditions,
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

    def _long_two_stream_state(self, current_calculation=CURRENT_J_FROM_RHOV):
        world = {
            "Nx": 16,
            "Ny": 1,
            "Nz": 1,
            "dx": 0.25,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.002,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 2,
            "guard_cells": 2,
            "current_calculation": current_calculation,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}

        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 1, 1)
        Nt = 1500

        x0 = jnp.linspace(
            -0.5 * world["x_wind"],
            0.5 * world["x_wind"],
            world["Nx"],
            endpoint=False,
        ) + 0.5 * world["dx"]
        x0 = x0 + 0.02 * world["dx"] * jnp.sin(2.0 * jnp.pi * (x0 / world["x_wind"] + 0.5))
        zeros = jnp.zeros_like(x0)
        beam_speed = 0.08

        species_kwargs = {
            "N_particles": world["Nx"],
            "mass": 1.0,
            "weight": 1.0,
            "T": 0.0,
            "x1": x0,
            "x2": zeros,
            "x3": zeros,
            "v2": zeros,
            "v3": zeros,
            "xwind": world["x_wind"],
            "ywind": world["y_wind"],
            "zwind": world["z_wind"],
            "dx": world["dx"],
            "dy": world["dy"],
            "dz": world["dz"],
            "dt": world["dt"],
        }
        particles = [
            particle_species(name="electron_left", charge=-1.0, v1=-beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="electron_right", charge=-1.0, v1=beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="ion_background", charge=2.0, v1=zeros, **species_kwargs),
        ]
        reference_particles = [
            particle_species(name="electron_left", charge=-1.0, v1=-beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="electron_right", charge=-1.0, v1=beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="ion_background", charge=2.0, v1=zeros, **species_kwargs),
        ]

        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)
        reference_fields = (E, B, J, rho, phi, external_fields, pml_state)

        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        tiled_particles = to_tiled_particles(particles, world, simulation_parameters)
        tiled_particles = self._pad_tiled_particle_capacity(tiled_particles, min_slots=12)
        g = int(world["guard_cells"])
        if current_calculation == CURRENT_ESIRKEPOV:
            J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=g, dtype=E[0].dtype)
        else:
            J_tiles = tile_vector_field(J, world, tile_shape, num_guard_cells=g)

        tiled_fields = (
            tile_vector_field(E, world, tile_shape, num_guard_cells=g),
            tile_vector_field(B, world, tile_shape, num_guard_cells=g),
            J_tiles,
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], world, tile_shape, num_guard_cells=g),
                tile_vector_field(external_fields[1], world, tile_shape, num_guard_cells=g),
            ),
            None,
        )

        self._assert_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            tile_shape,
            step=0,
        )

        return reference_particles, reference_fields, tiled_particles, tiled_fields, world, constants, tile_shape, Nt

    def _pad_tiled_particle_capacity(self, tiled_particles, min_slots):
        current_slots = tiled_particles.active.shape[-1]
        if current_slots >= min_slots:
            return tiled_particles

        pad_width = [(0, 0)] * tiled_particles.x.ndim
        pad_width[-2] = (0, min_slots - current_slots)
        x = jnp.pad(tiled_particles.x, pad_width)
        u = jnp.pad(tiled_particles.u, pad_width)

        scalar_pad_width = [(0, 0)] * tiled_particles.active.ndim
        scalar_pad_width[-1] = (0, min_slots - current_slots)
        charge = jnp.pad(tiled_particles.charge, scalar_pad_width)
        mass = jnp.pad(tiled_particles.mass, scalar_pad_width)
        weight = jnp.pad(tiled_particles.weight, scalar_pad_width)
        active = jnp.pad(tiled_particles.active, scalar_pad_width)
        update_x1 = jnp.pad(tiled_particles.update_x1, scalar_pad_width)
        update_x2 = jnp.pad(tiled_particles.update_x2, scalar_pad_width)
        update_x3 = jnp.pad(tiled_particles.update_x3, scalar_pad_width)
        update_u1 = jnp.pad(tiled_particles.update_u1, scalar_pad_width)
        update_u2 = jnp.pad(tiled_particles.update_u2, scalar_pad_width)
        update_u3 = jnp.pad(tiled_particles.update_u3, scalar_pad_width)

        return TiledParticles(
            x=x,
            u=u,
            charge=charge,
            mass=mass,
            weight=weight,
            active=active,
            update_x1=update_x1,
            update_x2=update_x2,
            update_x3=update_x3,
            update_u1=update_u1,
            update_u2=update_u2,
            update_u3=update_u3,
        )

    def _standard_species_rows(self, particles, species_index):
        species = particles[species_index]
        active = species.get_active_mask()
        x = jnp.stack(species.get_forward_position(), axis=1)[active]
        u = jnp.stack(species.get_velocity(), axis=1)[active]
        order = jnp.lexsort((u[:, 0], x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def _tiled_species_rows(self, tiled_particles, species_index):
        active = tiled_particles.active[:, :, :, species_index, :]
        x = tiled_particles.x[:, :, :, species_index, :, :].reshape(-1, 3)[active.reshape(-1)]
        u = tiled_particles.u[:, :, :, species_index, :, :].reshape(-1, 3)[active.reshape(-1)]
        order = jnp.lexsort((u[:, 0], x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def _assert_vector_fields_close(self, reference_field, tiled_field, step, name):
        for component_name, reference_component, tiled_component in zip(("x", "y", "z"), reference_field, tiled_field):
            diff = jnp.max(jnp.abs(reference_component - tiled_component))
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: {name}{component_name} max abs diff {diff}",
            )

    def _assert_tiled_state_matches_standard(
        self,
        reference_particles,
        reference_fields,
        tiled_particles,
        tiled_fields,
        world,
        tile_shape,
        step,
    ):
        if len(tiled_fields) > 7:
            self.assertFalse(bool(tiled_fields[-1]), f"step {step}: tiled particle capacity overflowed")

        for species_index in range(len(reference_particles)):
            reference_x, reference_u = self._standard_species_rows(reference_particles, species_index)
            tiled_x, tiled_u = self._tiled_species_rows(tiled_particles, species_index)
            self.assertTrue(
                jnp.allclose(tiled_x, reference_x, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: species {species_index} position mismatch",
            )
            self.assertTrue(
                jnp.allclose(tiled_u, reference_u, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: species {species_index} velocity mismatch",
            )

        reference_E, reference_B, reference_J, *_ = reference_fields
        E_tiles, B_tiles, J_tiles, *_ = tiled_fields
        num_guard_cells = int(world.get("guard_cells", 1))
        self._assert_vector_fields_close(
            reference_E,
            assemble_tiled_vector_field(E_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            step,
            "E",
        )
        self._assert_vector_fields_close(
            reference_B,
            assemble_tiled_vector_field(B_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            step,
            "B",
        )
        self._assert_vector_fields_close(
            reference_J,
            assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            step,
            "J",
        )

    def test_validate_field_solver_accepts_tiled_yee(self):
        validate_field_solver("tiled_yee")

    def test_tiled_yee_field_update_applies_standard_field_filter(self):
        world = self._build_world()
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.6}
        tile_shape = (2, 1, 1)
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))

        x = jnp.arange(world["Nx"], dtype=jnp.float64)[:, None, None]
        seed = jnp.zeros(shape)
        Ex = update_ghost_cells(seed.at[interior].set(0.1 + 0.03 * x), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Ey = update_ghost_cells(seed.at[interior].set(0.2 - 0.02 * x), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Ez = update_ghost_cells(seed.at[interior].set(0.05 * jnp.sin(x + 1.0)), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Bx = update_ghost_cells(seed.at[interior].set(0.07 * jnp.cos(x + 0.5)), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        By = update_ghost_cells(seed.at[interior].set(0.04 + 0.01 * x), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Bz = update_ghost_cells(seed.at[interior].set(0.03 * jnp.sin(2.0 * x)), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        E = (Ex, Ey, Ez)
        B = (Bx, By, Bz)
        J = (seed, seed, seed)
        rho = seed
        phi = seed
        external_fields = ((seed, seed, seed), (seed, seed, seed))

        _, reference_fields = time_loop_electrodynamic(
            [],
            (E, B, J, rho, phi, external_fields, None),
            world,
            constants,
            unused_curl,
            J_func=lambda particles, J, constants, world: J,
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

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
            [],
            tiled_fields,
            world,
            constants,
            unused_curl,
            J_func=lambda particles, J, constants, world, tile_shape=None, g=None: J,
            solver="tiled_yee",
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
            relativistic=False,
            particle_pusher="boris",
        )

        reference_E, reference_B, *_ = reference_fields
        E_tiles, B_tiles, *_ = tiled_fields
        self._assert_vector_fields_close(
            reference_E,
            assemble_tiled_vector_field(E_tiles, world, tile_shape),
            1,
            "E",
        )
        self._assert_vector_fields_close(
            reference_B,
            assemble_tiled_vector_field(B_tiles, world, tile_shape),
            1,
            "B",
        )

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

    def test_initialize_simulation_accepts_tiled_yee_higuera_cary_pusher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee higuera cary init smoke",
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
                    "particle_pusher": "higuera_cary",
                    "relativistic": True,
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
            config_path = os.path.join(tmpdir, "tiled_yee_higuera_cary.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, simulation_parameters, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic_tiled)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(simulation_parameters["particle_pusher"], "higuera_cary")

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

    def test_initialize_simulation_accepts_tiled_yee_bilinear_current_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee bilinear filter init smoke",
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
                    "filter_j": "bilinear",
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
            config_path = os.path.join(tmpdir, "tiled_yee_bilinear.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, simulation_parameters, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic_tiled)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(simulation_parameters["filter_j"], "bilinear")

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
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
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

    def test_tiled_yee_step_matches_standard_conducting_yee_step(self):
        world = self._build_world(
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING}
        )
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

        _, reference_fields = time_loop_electrodynamic(
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
        _, tiled_fields = time_loop_electrodynamic_tiled(
            tiled_particles,
            tiled_fields,
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="tiled_yee",
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
            relativistic=False,
            particle_pusher="boris",
        )

        E_tiles, B_tiles, J_tiles, *_ = tiled_fields
        reference_E, reference_B, reference_J, *_ = reference_fields

        for reference, tiled in zip(reference_E, assemble_tiled_vector_field(E_tiles, world, tile_shape)):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(reference_B, assemble_tiled_vector_field(B_tiles, world, tile_shape)):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(reference_J, assemble_tiled_vector_field(J_tiles, world, tile_shape)):
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
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
            relativistic=False,
            particle_pusher="boris",
        )

        _, _, reference_J, *_ = reference_fields
        _, _, J_tiles, *_ = tiled_fields
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape)

        for reference, tiled in zip(reference_J, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_yee_step_uses_bilinear_filtered_current(self):
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

        _, reference_fields = time_loop_electrodynamic(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            world,
            constants,
            unused_curl,
            J_func=lambda particles, J, constants, world: J_from_rhov(particles, J, constants, world, filter="bilinear"),
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
            J_func=functools.partial(direct_J_from_tiled_particles, filter="bilinear"),
            solver="tiled_yee",
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
            relativistic=False,
            particle_pusher="boris",
        )

        _, _, reference_J, *_ = reference_fields
        _, _, J_tiles, *_ = tiled_fields
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape)

        for reference, tiled in zip(reference_J, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_yee_step_bilinear_quadratic_two_guards_reduced_axes_matches_standard(self):
        world = self._build_world()
        world["shape_factor"] = 2
        world["guard_cells"] = 2
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = particle_species(
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
            v2=jnp.array([0.03, -0.04, 0.02, -0.01]),
            v3=jnp.array([-0.02, 0.01, -0.03, 0.04]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        reference_species = particle_species(
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
            v2=jnp.array([0.03, -0.04, 0.02, -0.01]),
            v3=jnp.array([-0.02, 0.01, -0.03, 0.04]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)

        _, reference_fields = time_loop_electrodynamic(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            world,
            constants,
            unused_curl,
            J_func=lambda particles, J, constants, world: J_from_rhov(particles, J, constants, world, filter="bilinear"),
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

        tiled_particles = to_tiled_particles([species], world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(B, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(J, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], world, tile_shape, num_guard_cells=int(world["guard_cells"])),
                tile_vector_field(external_fields[1], world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            ),
            None,
        )
        _, tiled_fields = time_loop_electrodynamic_tiled(
            tiled_particles,
            tiled_fields,
            world,
            constants,
            unused_curl,
            J_func=functools.partial(direct_J_from_tiled_particles, filter="bilinear"),
            solver="tiled_yee",
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
            relativistic=False,
            particle_pusher="boris",
        )

        self._assert_tiled_state_matches_standard(
            [reference_species],
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            tile_shape,
            1,
        )

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TILED_YEE") == "1",
        "Set RUN_SLOW_TILED_YEE=1 to run the 1500-step tiled Yee two-stream comparison.",
    )
    def test_long_two_stream_tiled_yee_direct_current_quadratic_two_guards_matches_standard_every_step(self):
        (
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            constants,
            tile_shape,
            Nt,
        ) = self._long_two_stream_state(CURRENT_J_FROM_RHOV)

        standard_J = functools.partial(J_from_rhov, filter="none")
        tiled_loop = jax.jit(
            time_loop_electrodynamic_tiled,
            static_argnames=("curl_func", "J_func", "solver", "tile_shape", "g", "relativistic", "particle_pusher"),
        )

        for t in range(Nt):
            reference_particles, reference_fields = time_loop_electrodynamic(
                reference_particles,
                reference_fields,
                world,
                constants,
                unused_curl,
                J_func=standard_J,
                solver="fdtd",
                relativistic=False,
                particle_pusher="boris",
            )
            tiled_particles, tiled_fields = tiled_loop(
                tiled_particles,
                tiled_fields,
                world,
                constants,
                unused_curl,
                J_func=None,
                solver="tiled_yee",
                tile_shape=tile_shape,
                g=int(world["guard_cells"]),
                relativistic=False,
                particle_pusher="boris",
            )

            self._assert_tiled_state_matches_standard(
                reference_particles,
                reference_fields,
                tiled_particles,
                tiled_fields,
                world,
                tile_shape,
                t + 1,
            )

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TILED_YEE") == "1",
        "Set RUN_SLOW_TILED_YEE=1 to run the 1500-step tiled Yee two-stream comparison.",
    )
    def test_long_two_stream_tiled_yee_esirkepov_quadratic_two_guards_matches_standard_every_step(self):
        (
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            constants,
            tile_shape,
            Nt,
        ) = self._long_two_stream_state(CURRENT_ESIRKEPOV)

        tiled_loop = jax.jit(
            time_loop_electrodynamic_tiled,
            static_argnames=("curl_func", "J_func", "solver", "tile_shape", "g", "relativistic", "particle_pusher"),
        )

        for t in range(Nt):
            reference_particles, reference_fields = time_loop_electrodynamic(
                reference_particles,
                reference_fields,
                world,
                constants,
                unused_curl,
                J_func=Esirkepov_current,
                solver="fdtd",
                relativistic=False,
                particle_pusher="boris",
            )
            tiled_particles, tiled_fields = tiled_loop(
                tiled_particles,
                tiled_fields,
                world,
                constants,
                unused_curl,
                J_func=tiled_esirkepov_current,
                solver="tiled_yee",
                tile_shape=tile_shape,
                g=int(world["guard_cells"]),
                relativistic=False,
                particle_pusher="boris",
            )

            self._assert_tiled_state_matches_standard(
                reference_particles,
                reference_fields,
                tiled_particles,
                tiled_fields,
                world,
                tile_shape,
                t + 1,
            )


if __name__ == "__main__":
    unittest.main()
